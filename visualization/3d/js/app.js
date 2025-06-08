/**
 * keywords: [visualization, app, main, coordinator, scientific, snn]
 * 
 * Main application coordinator for SNN visualization
 */

import { DataManager } from './data-manager.js';
import { Viewport3D } from './viewport-3d.js';
import { NeuralVisualizer } from './neural-visualizer.js';
import { ChartManager } from './chart-manager.js';
import { TimelineController } from './timeline-controller.js';
import { ExportManager } from './export-manager.js';

class SNNVisualizationApp {
    constructor() {
        this.dataManager = new DataManager();
        this.viewport3D = null;
        this.neuralViz = null;
        this.chartManager = null;
        this.timeline = null;
        this.exportManager = new ExportManager();
        
        this.state = {
            currentExperiment: null,
            currentEpisode: null,
            currentTime: 0,
            isPlaying: false,
            playbackSpeed: 1.0,
            data: null
        };
        
        this.animationFrameId = null;
        this.lastFrameTime = 0;
    }
    
    async init() {
        // Initialize UI elements
        this.elements = {
            experimentSelect: document.getElementById('experiment-select'),
            episodeSelect: document.getElementById('episode-select'),
            playPauseBtn: document.getElementById('play-pause-btn'),
            exportBtn: document.getElementById('export-btn'),
            speedControl: document.getElementById('speed-control'),
            speedDisplay: document.getElementById('speed-display'),
            loading: document.getElementById('loading')
        };
        
        // Initialize visualization components
        this.viewport3D = new Viewport3D('viewport');
        this.neuralViz = new NeuralVisualizer('neural-viewer');
        this.chartManager = new ChartManager();
        this.timeline = new TimelineController('timeline', this.onTimelineChange.bind(this));
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load experiments
        await this.loadExperiments();
        
        // Hide loading
        this.setLoading(false);
    }
    
    setupEventListeners() {
        // Experiment selection
        this.elements.experimentSelect.addEventListener('change', async (e) => {
            if (e.target.value) {
                await this.loadExperiment(e.target.value);
            }
        });
        
        // Episode selection
        this.elements.episodeSelect.addEventListener('change', async (e) => {
            if (e.target.value) {
                await this.loadEpisode(parseInt(e.target.value));
            }
        });
        
        // Playback controls
        this.elements.playPauseBtn.addEventListener('click', () => {
            this.togglePlayback();
        });
        
        // Speed control
        this.elements.speedControl.addEventListener('input', (e) => {
            this.state.playbackSpeed = parseFloat(e.target.value);
            this.elements.speedDisplay.textContent = `${this.state.playbackSpeed.toFixed(1)}x`;
        });
        
        // Export
        this.elements.exportBtn.addEventListener('click', () => {
            this.exportData();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    this.togglePlayback();
                    break;
                case 'ArrowLeft':
                    this.stepBackward();
                    break;
                case 'ArrowRight':
                    this.stepForward();
                    break;
                case 'r':
                    this.resetView();
                    break;
            }
        });
    }
    
    async loadExperiments() {
        try {
            const experiments = await this.dataManager.getExperiments();
            
            // Populate dropdown
            this.elements.experimentSelect.innerHTML = '<option value="">Select Experiment...</option>';
            experiments.forEach(exp => {
                const option = document.createElement('option');
                option.value = exp.experiment_id;
                option.textContent = `${exp.experiment_id} (${exp.num_episodes} episodes)`;
                this.elements.experimentSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Failed to load experiments:', error);
            this.showError('Failed to load experiments');
        }
    }
    
    async loadExperiment(experimentId) {
        this.setLoading(true);
        
        try {
            // Get experiment analysis
            const analysis = await this.dataManager.getAnalysis(experimentId);
            this.state.currentExperiment = experimentId;
            
            // Update episode list
            this.updateEpisodeList(analysis.episodes);
            
            // Update charts with aggregate data
            this.chartManager.updateLearningCurve(analysis.aggregate.learning_curve);
            
            // Enable controls
            this.elements.episodeSelect.disabled = false;
            
        } catch (error) {
            console.error('Failed to load experiment:', error);
            this.showError('Failed to load experiment data');
        } finally {
            this.setLoading(false);
        }
    }
    
    async loadEpisode(episodeId) {
        this.setLoading(true);
        
        try {
            // Stop playback
            this.pause();
            
            // Load episode data
            const data = await this.dataManager.getEpisodeData(
                this.state.currentExperiment, 
                episodeId
            );
            
            this.state.currentEpisode = episodeId;
            this.state.data = data;
            this.state.currentTime = 0;
            
            // Initialize visualizations
            this.viewport3D.setData(data);
            this.neuralViz.setData(data.neural);
            this.chartManager.setEpisodeData(data);
            this.timeline.setData(data);
            
            // Update stats
            this.updateStats(data);
            
            // Enable controls
            this.elements.playPauseBtn.disabled = false;
            this.elements.exportBtn.disabled = false;
            
            // Start WebSocket connection for real-time updates
            await this.dataManager.connectWebSocket(this.state.currentExperiment);
            
        } catch (error) {
            console.error('Failed to load episode:', error);
            this.showError('Failed to load episode data');
        } finally {
            this.setLoading(false);
        }
    }
    
    updateEpisodeList(episodes) {
        const listContainer = document.getElementById('episode-list');
        listContainer.innerHTML = '';
        
        episodes.forEach(ep => {
            const item = document.createElement('div');
            item.className = 'episode-item';
            item.style.cssText = `
                padding: 0.8rem;
                margin-bottom: 0.5rem;
                background: #2a2a2a;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.2s;
            `;
            
            item.innerHTML = `
                <div style="font-weight: 500;">Episode ${ep.episode_id}</div>
                <div style="font-size: 0.85rem; color: #888; margin-top: 0.2rem;">
                    Reward: ${ep.total_reward.toFixed(1)} | 
                    Length: ${ep.episode_length}
                </div>
            `;
            
            item.addEventListener('click', () => {
                this.elements.episodeSelect.value = ep.episode_id;
                this.loadEpisode(ep.episode_id);
            });
            
            item.addEventListener('mouseenter', () => {
                item.style.background = '#3a3a3a';
            });
            
            item.addEventListener('mouseleave', () => {
                item.style.background = '#2a2a2a';
            });
            
            listContainer.appendChild(item);
        });
        
        // Update episode dropdown
        this.elements.episodeSelect.innerHTML = '<option value="">Select Episode...</option>';
        episodes.forEach(ep => {
            const option = document.createElement('option');
            option.value = ep.episode_id;
            option.textContent = `Episode ${ep.episode_id}`;
            this.elements.episodeSelect.appendChild(option);
        });
    }
    
    updateStats(data) {
        // Calculate statistics
        const totalReward = data.rewards.reduce((a, b) => a + b, 0);
        const episodeLength = data.trajectory.x.length;
        const avgFiringRate = this.calculateAvgFiringRate(data.neural);
        
        // Update UI
        document.getElementById('stat-total-reward').textContent = totalReward.toFixed(1);
        document.getElementById('stat-episode-length').textContent = episodeLength;
        document.getElementById('stat-firing-rate').textContent = `${avgFiringRate.toFixed(1)} Hz`;
    }
    
    calculateAvgFiringRate(neuralData) {
        if (!neuralData || !neuralData.spikes) return 0;
        
        const totalSpikes = neuralData.spikes.reduce((sum, row) => 
            sum + row.reduce((a, b) => a + b, 0), 0
        );
        const duration = neuralData.spikes.length / 1000; // Convert ms to seconds
        const numNeurons = neuralData.spikes[0].length;
        
        return totalSpikes / (duration * numNeurons);
    }
    
    onTimelineChange(time) {
        this.state.currentTime = time;
        this.updateVisualization();
    }
    
    updateVisualization() {
        if (!this.state.data) return;
        
        const time = Math.floor(this.state.currentTime);
        
        // Update 3D viewport
        this.viewport3D.update(time);
        
        // Update neural visualization
        this.neuralViz.update(time);
        
        // Update charts
        this.chartManager.updateCurrentTime(time);
        
        // Update time display
        this.updateTimeDisplay();
    }
    
    updateTimeDisplay() {
        if (!this.state.data || !this.state.data.trajectory) {
            document.getElementById('time-display').textContent = '0:00';
            document.getElementById('time-total').textContent = '0:00';
            return;
        }
        
        const currentSec = Math.floor(this.state.currentTime / 1000);
        const totalSec = Math.floor(this.state.data.trajectory.x.length / 1000);
        
        const formatTime = (seconds) => {
            const min = Math.floor(seconds / 60);
            const sec = seconds % 60;
            return `${min}:${sec.toString().padStart(2, '0')}`;
        };
        
        document.getElementById('time-display').textContent = formatTime(currentSec);
        document.getElementById('time-total').textContent = formatTime(totalSec);
    }
    
    togglePlayback() {
        if (this.state.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }
    
    play() {
        this.state.isPlaying = true;
        this.elements.playPauseBtn.textContent = '⏸ Pause';
        this.lastFrameTime = performance.now();
        this.animate();
    }
    
    pause() {
        this.state.isPlaying = false;
        this.elements.playPauseBtn.textContent = '▶ Play';
        
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }
    
    animate() {
        if (!this.state.isPlaying) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        
        // Update time based on playback speed
        const timeIncrement = deltaTime * this.state.playbackSpeed;
        this.state.currentTime += timeIncrement;
        
        // Check bounds
        if (this.state.currentTime >= this.state.data.trajectory.x.length) {
            this.state.currentTime = 0; // Loop
        }
        
        // Update timeline
        this.timeline.setTime(this.state.currentTime);
        
        // Update visualization
        this.updateVisualization();
        
        // Continue animation
        this.animationFrameId = requestAnimationFrame(() => this.animate());
    }
    
    stepForward() {
        this.pause();
        this.state.currentTime = Math.min(
            this.state.currentTime + 100,
            this.state.data.trajectory.x.length - 1
        );
        this.timeline.setTime(this.state.currentTime);
        this.updateVisualization();
    }
    
    stepBackward() {
        this.pause();
        this.state.currentTime = Math.max(this.state.currentTime - 100, 0);
        this.timeline.setTime(this.state.currentTime);
        this.updateVisualization();
    }
    
    resetView() {
        this.viewport3D.resetCamera();
    }
    
    async exportData() {
        try {
            const exportData = {
                experiment: this.state.currentExperiment,
                episode: this.state.currentEpisode,
                data: this.state.data,
                analysis: this.chartManager.getAnalysis()
            };
            
            await this.exportManager.exportToFile(exportData, 
                `${this.state.currentExperiment}_episode_${this.state.currentEpisode}`);
                
        } catch (error) {
            console.error('Export failed:', error);
            this.showError('Failed to export data');
        }
    }
    
    setLoading(loading) {
        this.elements.loading.style.display = loading ? 'flex' : 'none';
    }
    
    showError(message) {
        // TODO: Implement proper error notification
        console.error(message);
        alert(message);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new SNNVisualizationApp();
    app.init().catch(console.error);
    
    // Expose to window for debugging
    window.snnApp = app;
});