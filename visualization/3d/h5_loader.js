// keywords: [hdf5, loader, browser, visualization]
/**
 * HDF5 Loader for browser-based visualization
 * Uses h5wasm to load HDF5 files directly in the browser
 */

export class ExperimentH5Loader {
    constructor() {
        this.h5wasm = null;
        this.file = null;
    }

    async initialize() {
        // Load h5wasm library
        if (!this.h5wasm) {
            const { h5wasm } = await import('https://cdn.jsdelivr.net/npm/h5wasm@0.6.0/dist/esm/hdf5_hl.js');
            await h5wasm.ready;
            this.h5wasm = h5wasm;
        }
    }

    async loadFile(url) {
        await this.initialize();
        
        // Fetch the file
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        
        // Load into h5wasm
        this.file = new this.h5wasm.File(new Uint8Array(buffer), 'experiment_data.h5');
        
        return this;
    }

    getMetadata() {
        const metadata = {};
        const attrs = this.file.attrs;
        
        for (const key of attrs.keys()) {
            const value = attrs.get(key).value;
            if (key.startsWith('meta_')) {
                const cleanKey = key.substring(5);
                metadata[cleanKey] = this._parseValue(value);
            } else {
                metadata[key] = this._parseValue(value);
            }
        }
        
        return metadata;
    }

    getConfig() {
        const config = {};
        
        if (this.file.keys().includes('config')) {
            const configGroup = this.file.get('config');
            const attrs = configGroup.attrs;
            
            for (const key of attrs.keys()) {
                config[key] = this._parseValue(attrs.get(key).value);
            }
        }
        
        return config;
    }

    listEpisodes() {
        const episodes = [];
        
        if (this.file.keys().includes('episodes')) {
            const episodesGroup = this.file.get('episodes');
            
            for (const name of episodesGroup.keys()) {
                if (name.startsWith('episode_')) {
                    const episodeId = parseInt(name.split('_')[1]);
                    episodes.push(episodeId);
                }
            }
        }
        
        return episodes.sort((a, b) => a - b);
    }

    getEpisodeData(episodeId) {
        const episodeName = `episode_${String(episodeId).padStart(4, '0')}`;
        const episodePath = `episodes/${episodeName}`;
        
        if (!this.file.get(episodePath)) {
            throw new Error(`Episode ${episodeId} not found`);
        }
        
        const episode = this.file.get(episodePath);
        const data = {
            metadata: {},
            behavior: {},
            rewards: {},
            spikes: {},
            neural_states: {},
            weight_changes: {}
        };
        
        // Get episode attributes
        const attrs = episode.attrs;
        for (const key of attrs.keys()) {
            data.metadata[key] = this._parseValue(attrs.get(key).value);
        }
        
        // Get behavior data
        if (episode.keys().includes('behavior')) {
            const behavior = episode.get('behavior');
            for (const key of behavior.keys()) {
                const dataset = behavior.get(key);
                data.behavior[key] = dataset.value;
            }
        }
        
        // Get rewards data
        if (episode.keys().includes('rewards')) {
            const rewards = episode.get('rewards');
            for (const key of rewards.keys()) {
                const dataset = rewards.get(key);
                data.rewards[key] = dataset.value;
            }
        }
        
        return data;
    }

    getTrajectoryData(episodeId = 0) {
        const config = this.getConfig();
        const episodeData = this.getEpisodeData(episodeId);
        
        // Build visualization-compatible data
        const vizData = {
            metadata: {
                gridSize: config.world_config?.grid_size || 10,
                nRewards: config.world_config?.n_rewards || 5,
                totalSteps: 0,
                totalReward: 0,
                rewardsCollected: 0,
                coverage: 0,
                seed: config.seed || -1,
                episodeId: episodeId
            },
            trajectory: [],
            world: {
                rewardPositions: []
            }
        };
        
        // Extract trajectory from behavior data
        if (episodeData.behavior.positions) {
            const positions = episodeData.behavior.positions;
            const actions = episodeData.behavior.actions || [];
            const observations = episodeData.behavior.observations || [];
            
            vizData.metadata.totalSteps = positions.length - 1;
            
            // Build trajectory
            let cumulativeReward = 0;
            const rewardsCollected = new Array(vizData.metadata.nRewards).fill(false);
            
            for (let i = 0; i < positions.length; i++) {
                const stepData = {
                    step: i,
                    agentPos: positions[i],
                    observation: observations[i] || 0,
                    cumulativeReward: cumulativeReward
                };
                
                if (i > 0 && i - 1 < actions.length) {
                    stepData.action = actions[i - 1];
                }
                
                // Check rewards collected
                if (episodeData.rewards.collected_at_step) {
                    const collectedSteps = episodeData.rewards.collected_at_step;
                    for (let j = 0; j < collectedSteps.length; j++) {
                        if (collectedSteps[j] <= i && collectedSteps[j] > 0) {
                            rewardsCollected[j] = true;
                        }
                    }
                }
                
                stepData.rewardCollected = [...rewardsCollected];
                
                if (episodeData.rewards.reward_per_step && i > 0) {
                    const stepReward = episodeData.rewards.reward_per_step[i - 1] || 0;
                    stepData.reward = stepReward;
                    cumulativeReward += stepReward;
                }
                
                vizData.trajectory.push(stepData);
            }
            
            vizData.metadata.totalReward = cumulativeReward;
            vizData.metadata.rewardsCollected = rewardsCollected.filter(r => r).length;
        }
        
        // Get world data
        if (episodeData.rewards.positions) {
            vizData.world.rewardPositions = episodeData.rewards.positions;
        }
        
        // Calculate coverage if we have unique positions
        if (episodeData.behavior.unique_positions_visited) {
            const uniqueCount = episodeData.behavior.unique_positions_visited;
            const gridArea = vizData.metadata.gridSize * vizData.metadata.gridSize;
            vizData.metadata.coverage = uniqueCount / gridArea;
        }
        
        return vizData;
    }

    _parseValue(value) {
        // Handle different value types
        if (typeof value === 'string') {
            // Try to parse JSON
            if (value.startsWith('{') || value.startsWith('[')) {
                try {
                    return JSON.parse(value);
                } catch (e) {
                    return value;
                }
            }
        }
        return value;
    }

    close() {
        if (this.file) {
            // h5wasm doesn't need explicit closing in browser
            this.file = null;
        }
    }
}

// Alternative loader using server-side API
export class ExperimentAPILoader {
    constructor(apiBaseUrl = '/api') {
        this.apiBaseUrl = apiBaseUrl;
    }

    async loadFile(path) {
        // Request server to load the file
        const response = await fetch(`${this.apiBaseUrl}/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to load file: ${response.statusText}`);
        }
        
        this.filePath = path;
        return this;
    }

    async getTrajectoryData(episodeId = 0) {
        const response = await fetch(`${this.apiBaseUrl}/trajectory?path=${encodeURIComponent(this.filePath)}&episode=${episodeId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to get trajectory: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async listExperiments() {
        const response = await fetch(`${this.apiBaseUrl}/experiments`);
        
        if (!response.ok) {
            throw new Error(`Failed to list experiments: ${response.statusText}`);
        }
        
        return await response.json();
    }
}