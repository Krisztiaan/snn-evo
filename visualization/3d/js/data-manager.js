/**
 * keywords: [data, manager, api, websocket, cache, streaming]
 * 
 * Data management module for efficient data loading and caching
 */

export class DataManager {
    constructor() {
        this.baseUrl = 'http://localhost:8080/api';
        this.wsUrl = 'ws://localhost:8080/ws';
        this.cache = new Map();
        this.ws = null;
        this.pendingRequests = new Map();
        this.requestId = 0;
    }
    
    /**
     * Get list of all experiments
     */
    async getExperiments() {
        const cacheKey = 'experiments';
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        try {
            const response = await fetch(`${this.baseUrl}/experiments`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            
            // Cache for 5 minutes
            this.cache.set(cacheKey, data);
            setTimeout(() => this.cache.delete(cacheKey), 5 * 60 * 1000);
            
            return data;
        } catch (error) {
            throw new Error(`Failed to fetch experiments: ${error.message}`);
        }
    }
    
    /**
     * Get analysis for a specific experiment
     */
    async getAnalysis(experimentId) {
        const cacheKey = `analysis_${experimentId}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        try {
            const response = await fetch(`${this.baseUrl}/experiment/${experimentId}/analysis`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.cache.set(cacheKey, data);
            
            return data;
        } catch (error) {
            throw new Error(`Failed to fetch analysis: ${error.message}`);
        }
    }
    
    /**
     * Get episode data with intelligent decimation
     */
    async getEpisodeData(experimentId, episodeId) {
        const cacheKey = `episode_${experimentId}_${episodeId}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        try {
            // Fetch trajectory data
            const trajectoryResponse = await fetch(
                `${this.baseUrl}/experiment/${experimentId}/episode/${episodeId}/trajectory?decimation=1`
            );
            
            if (!trajectoryResponse.ok) throw new Error(`HTTP ${trajectoryResponse.status}`);
            
            const trajectoryData = await trajectoryResponse.json();
            
            // For neural data, we'll use WebSocket streaming
            // This is a placeholder - real neural data comes via WS
            const neuralData = {
                spikes: [],
                weights: [],
                dopamine: []
            };
            
            const episodeData = {
                trajectory: trajectoryData.trajectory,
                rewards: trajectoryData.trajectory.rewards,
                values: trajectoryData.trajectory.values,
                neural: neuralData,
                metadata: trajectoryData.metadata
            };
            
            this.cache.set(cacheKey, episodeData);
            
            return episodeData;
            
        } catch (error) {
            throw new Error(`Failed to fetch episode data: ${error.message}`);
        }
    }
    
    /**
     * Connect WebSocket for real-time data streaming
     */
    async connectWebSocket(experimentId) {
        // Close existing connection
        if (this.ws) {
            this.ws.close();
        }
        
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(`${this.wsUrl}/stream/${experimentId}`);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.startHeartbeat();
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.stopHeartbeat();
                this.ws = null;
            };
            
            // Timeout connection attempt
            setTimeout(() => {
                if (this.ws.readyState === WebSocket.CONNECTING) {
                    this.ws.close();
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    async handleWebSocketMessage(event) {
        try {
            let data;
            
            // Handle binary data (compressed neural data)
            if (event.data instanceof Blob) {
                const arrayBuffer = await event.data.arrayBuffer();
                const decompressed = this.decompressData(arrayBuffer);
                data = msgpack.decode(new Uint8Array(decompressed));
            } else {
                data = JSON.parse(event.data);
            }
            
            // Route message by type
            switch (data.type) {
                case 'metadata':
                    this.handleMetadata(data.data);
                    break;
                    
                case 'neural_chunk':
                    this.handleNeuralChunk(data.data);
                    break;
                    
                case 'trajectory_segments':
                    this.handleTrajectorySegments(data.data);
                    break;
                    
                case 'error':
                    console.error('Server error:', data.message);
                    break;
                    
                case 'heartbeat':
                case 'pong':
                    // Keep connection alive
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
            
            // Resolve pending requests
            if (data.requestId && this.pendingRequests.has(data.requestId)) {
                const { resolve } = this.pendingRequests.get(data.requestId);
                resolve(data);
                this.pendingRequests.delete(data.requestId);
            }
            
        } catch (error) {
            console.error('Failed to handle WebSocket message:', error);
        }
    }
    
    /**
     * Request specific data via WebSocket
     */
    async requestData(dataType, episodeId, options = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }
        
        const requestId = this.requestId++;
        
        return new Promise((resolve, reject) => {
            // Store pending request
            this.pendingRequests.set(requestId, { resolve, reject });
            
            // Send request
            this.ws.send(JSON.stringify({
                type: 'request_data',
                requestId,
                data: {
                    data_type: dataType,
                    episode_id: episodeId,
                    ...options
                }
            }));
            
            // Timeout request
            setTimeout(() => {
                if (this.pendingRequests.has(requestId)) {
                    this.pendingRequests.delete(requestId);
                    reject(new Error('Request timeout'));
                }
            }, 30000);
        });
    }
    
    /**
     * Stream neural data for an episode
     */
    async streamNeuralData(episodeId, onChunk) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }
        
        // Override message handler temporarily
        const originalHandler = this.ws.onmessage;
        
        this.ws.onmessage = async (event) => {
            if (event.data instanceof Blob) {
                const arrayBuffer = await event.data.arrayBuffer();
                const decompressed = this.decompressData(arrayBuffer);
                const chunk = msgpack.decode(new Uint8Array(decompressed));
                onChunk(chunk);
            } else {
                originalHandler.call(this.ws, event);
            }
        };
        
        // Request neural data stream
        await this.requestData('neural', episodeId);
        
        // Restore original handler after streaming
        setTimeout(() => {
            this.ws.onmessage = originalHandler;
        }, 60000); // 1 minute timeout
    }
    
    /**
     * Decompress LZ4 data
     */
    decompressData(compressedData) {
        // Using lz4js library
        const compressed = new Uint8Array(compressedData);
        const decompressed = LZ4.decompress(compressed);
        return decompressed.buffer;
    }
    
    /**
     * WebSocket heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 25000); // Every 25 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    /**
     * Handle metadata updates
     */
    handleMetadata(metadata) {
        // Store metadata for future use
        this.currentMetadata = metadata;
    }
    
    /**
     * Handle neural data chunks
     */
    handleNeuralChunk(chunk) {
        // Emit event for visualization update
        window.dispatchEvent(new CustomEvent('neural-data-chunk', { detail: chunk }));
    }
    
    /**
     * Handle trajectory segments
     */
    handleTrajectorySegments(segments) {
        // Emit event for trajectory visualization
        window.dispatchEvent(new CustomEvent('trajectory-segments', { detail: segments }));
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }
    
    /**
     * Disconnect and cleanup
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
        this.clearCache();
        this.pendingRequests.clear();
    }
}