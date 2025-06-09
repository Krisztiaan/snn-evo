/**
 * keywords: [data, fetch, cache, simple, api]
 * 
 * Simplified data management using fetch and sessionStorage
 */

export class SimpleDataManager {
    constructor() {
        this.baseUrl = 'http://localhost:8080/api';
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }
    
    // Simple cache wrapper
    async fetchWithCache(url, cacheKey) {
        // Check memory cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.data;
            }
        }
        
        // Check sessionStorage
        const stored = sessionStorage.getItem(cacheKey);
        if (stored) {
            const { data, timestamp } = JSON.parse(stored);
            if (Date.now() - timestamp < this.cacheTimeout) {
                this.cache.set(cacheKey, { data, timestamp });
                return data;
            }
        }
        
        // Fetch fresh data
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const cached = { data, timestamp: Date.now() };
        
        // Store in both caches
        this.cache.set(cacheKey, cached);
        try {
            sessionStorage.setItem(cacheKey, JSON.stringify(cached));
        } catch (e) {
            // Ignore quota errors
        }
        
        return data;
    }
    
    async getExperiments() {
        return this.fetchWithCache(
            `${this.baseUrl}/experiments`,
            'experiments'
        );
    }
    
    async getAnalysis(experimentId) {
        return this.fetchWithCache(
            `${this.baseUrl}/experiment/${experimentId}/analysis`,
            `analysis_${experimentId}`
        );
    }
    
    async getEpisodeData(experimentId, episodeId) {
        const trajectoryData = await this.fetchWithCache(
            `${this.baseUrl}/experiment/${experimentId}/episode/${episodeId}/trajectory`,
            `episode_${experimentId}_${episodeId}`
        );
        
        // Simplified data structure
        return {
            trajectory: trajectoryData.trajectory,
            rewards: trajectoryData.trajectory.rewards,
            values: trajectoryData.trajectory.values || [],
            neural: { spikes: [], weights: [] }, // Placeholder - load on demand
            metadata: trajectoryData.metadata
        };
    }
    
    // Remove WebSocket complexity - use Server-Sent Events if needed
    connectSSE(experimentId, onMessage) {
        const eventSource = new EventSource(`${this.baseUrl}/stream/${experimentId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        
        eventSource.onerror = () => {
            console.log('SSE connection lost, will auto-reconnect');
        };
        
        return eventSource;
    }
    
    clearCache() {
        this.cache.clear();
        // Clear only our keys from sessionStorage
        Object.keys(sessionStorage).forEach(key => {
            if (key.startsWith('experiments') || 
                key.startsWith('analysis_') || 
                key.startsWith('episode_')) {
                sessionStorage.removeItem(key);
            }
        });
    }
}