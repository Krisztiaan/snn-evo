/**
 * keywords: [alpine, stores, state, optimized, performance]
 * 
 * Optimized Alpine.js stores for high-performance visualization
 */

document.addEventListener('alpine:init', () => {
    
    // Optimized playback store with frame-based timing
    Alpine.store('playback', {
        time: 0,
        duration: 0,
        playing: false,
        speed: 1,
        targetFPS: 60,
        
        // Internal state
        _rafId: null,
        _lastFrameTime: 0,
        _updateCallbacks: new Set(),
        _frameAccumulator: 0,
        
        get progress() {
            return this.duration > 0 ? (this.time / this.duration) * 100 : 0;
        },
        
        play() {
            if (this.playing) return;
            this.playing = true;
            this._lastFrameTime = performance.now();
            this._startAnimationLoop();
        },
        
        pause() {
            this.playing = false;
            if (this._rafId) {
                cancelAnimationFrame(this._rafId);
                this._rafId = null;
            }
        },
        
        _startAnimationLoop() {
            const targetFrameTime = 1000 / this.targetFPS;
            
            const animate = (currentTime) => {
                if (!this.playing) return;
                
                const deltaTime = Math.min(currentTime - this._lastFrameTime, 100);
                this._lastFrameTime = currentTime;
                
                // Accumulate time based on speed
                this._frameAccumulator += (deltaTime / targetFrameTime) * this.speed;
                
                // Only update on whole frame changes
                if (this._frameAccumulator >= 1) {
                    const frameSteps = Math.floor(this._frameAccumulator);
                    const newTime = this.time + frameSteps;
                    
                    // Loop or clamp
                    if (newTime >= this.duration) {
                        this.time = 0; // Reset to start
                        this._frameAccumulator = 0;
                    } else {
                        this.time = newTime;
                        this._frameAccumulator -= frameSteps;
                    }
                    
                    this._notifyUpdate(this.time);
                }
                
                this._rafId = requestAnimationFrame(animate);
            };
            
            this._rafId = requestAnimationFrame(animate);
        },
        
        seek(time) {
            this.time = Math.max(0, Math.min(time, this.duration - 1));
            this._notifyUpdate(this.time);
        },
        
        reset() {
            this.time = 0;
            this._lastFrameTime = 0;
            this._frameAccumulator = 0;
            this._notifyUpdate(0);
        },
        
        onUpdate(callback) {
            this._updateCallbacks.add(callback);
            return () => this._updateCallbacks.delete(callback);
        },
        
        _notifyUpdate(time) {
            this._updateCallbacks.forEach(cb => cb(time));
        }
    });
    
    // Data store - handles experiments and episodes
    Alpine.store('data', {
        experiments: [],
        episodes: [],
        currentExperiment: '',
        currentEpisode: null,
        episodeData: null
    });
    
    // UI store - handles loading states and preferences
    Alpine.store('ui', {
        loading: false,
        cameraFollow: false,
        showHeatmap: false,
        showPerformance: false,
        quality: 'medium',
        
        init() {
            // Load preferences
            this.cameraFollow = JSON.parse(localStorage.getItem('viz_cameraFollow') || 'false');
            this.showHeatmap = JSON.parse(localStorage.getItem('viz_showHeatmap') || 'false');
            this.showPerformance = JSON.parse(localStorage.getItem('viz_showPerformance') || 'false');
            this.quality = localStorage.getItem('viz_quality') || 'medium';
            
            // Watch for changes with debouncing
            let saveTimeout;
            Alpine.effect(() => {
                clearTimeout(saveTimeout);
                saveTimeout = setTimeout(() => {
                    localStorage.setItem('viz_cameraFollow', JSON.stringify(this.cameraFollow));
                    localStorage.setItem('viz_showHeatmap', JSON.stringify(this.showHeatmap));
                    localStorage.setItem('viz_showPerformance', JSON.stringify(this.showPerformance));
                    localStorage.setItem('viz_quality', this.quality);
                }, 500);
            });
        }
    });
    
    // Stats store - computed statistics
    Alpine.store('stats', {
        get totalReward() {
            const data = Alpine.store('data').episodeData;
            if (!data) return '-';
            return data.rewards.reduce((a, b) => a + b, 0).toFixed(1);
        },
        
        get episodeLength() {
            const data = Alpine.store('data').episodeData;
            return data ? data.trajectory.x.length : '-';
        },
        
        get firingRate() {
            const data = Alpine.store('data').episodeData;
            if (!data || !data.neural || !data.neural.spikes) return '-';
            
            const spikes = data.neural.spikes;
            if (spikes.length === 0) return '0.0 Hz';
            
            const totalSpikes = spikes.flat().reduce((a, b) => a + b, 0);
            const duration = spikes.length / 1000;
            const neurons = spikes[0]?.length || 1;
            
            return `${(totalSpikes / (duration * neurons)).toFixed(1)} Hz`;
        }
    });
    
    // Performance monitoring store
    Alpine.store('performance', {
        fps: 0,
        frameTime: 0,
        memoryUsage: 0,
        
        updateFPS(fps) {
            this.fps = fps;
        }
    });
    
    // Modules store - lazy loaded visualization modules
    Alpine.store('modules', {
        viewport: null,
        charts: null,
        neural: null,
        
        async load() {
            if (this.viewport) return;
            
            const [v3d, cm, nv] = await Promise.all([
                import('./simple-viewport.js'),
                import('./simple-charts.js'),
                import('./simple-neural.js')
            ]);
            
            this.viewport = new v3d.SimpleViewport3D('viewport');
            this.charts = new cm.SimpleCharts();
            this.neural = new nv.SimpleNeuralVisualizer('spike-raster');
        }
    });
    
    // Helper functions as a store
    Alpine.store('utils', {
        formatTime(frames) {
            // Convert frames to seconds (assuming 1000 frames = 1 second for timesteps)
            const seconds = Math.floor(frames / 1000);
            const min = Math.floor(seconds / 60);
            const sec = seconds % 60;
            return `${min}:${sec.toString().padStart(2, '0')}`;
        },
        
        throttle(func, limit) {
            let inThrottle;
            return function(...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },
        
        async fetchData(url) {
            const cached = sessionStorage.getItem(url);
            if (cached) {
                const { data, timestamp } = JSON.parse(cached);
                if (Date.now() - timestamp < 300000) return data;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            
            try {
                sessionStorage.setItem(url, JSON.stringify({ data, timestamp: Date.now() }));
            } catch (e) {
                // Ignore quota errors
            }
            
            return data;
        }
    });
});

// Initialize stores
document.addEventListener('alpine:init', () => {
    Alpine.store('ui').init();
});