/**
 * keywords: [alpine, app, optimized, main, performance]
 * 
 * Optimized main app component with performance enhancements
 */

document.addEventListener('alpine:init', () => {
    
    Alpine.data('app', () => ({
        // Store references
        playback: Alpine.store('playback'),
        data: Alpine.store('data'),
        ui: Alpine.store('ui'),
        stats: Alpine.store('stats'),
        modules: Alpine.store('modules'),
        utils: Alpine.store('utils'),
        performance: Alpine.store('performance'),
        
        // Managers
        dataManager: null,
        updateThrottled: null,
        
        // Performance tracking
        lastUpdateTime: -1,
        frameSkipCounter: 0,
        
        async init() {
            // Load modules
            await this.modules.load();
            
            // Initialize data manager
            const dm = await import('./simple-data-manager.js');
            this.dataManager = new dm.SimpleDataManager();
            
            // Create throttled update function
            this.updateThrottled = this.utils.throttle((time) => {
                this.updateVisualizations(time);
            }, 16); // ~60fps
            
            // Setup optimized playback updates
            this.playback.onUpdate((time) => {
                // Direct update for viewport (most critical)
                if (this.modules.viewport) {
                    this.modules.viewport.update(time);
                }
                
                // Throttle other updates
                const timeInt = Math.floor(time);
                if (timeInt !== this.lastUpdateTime && timeInt % 5 === 0) {
                    this.lastUpdateTime = timeInt;
                    // Update less critical components
                    if (this.modules.charts) this.modules.charts.updateTime(time);
                    if (this.modules.neural) this.modules.neural.updateTime(time);
                }
            });
            
            // Setup viewport controls
            Alpine.effect(() => {
                if (this.modules.viewport) {
                    this.modules.viewport.setQuality(this.ui.quality);
                    this.modules.viewport.followAgent = this.ui.cameraFollow;
                    this.modules.viewport.showHeatmap = this.ui.showHeatmap;
                }
            });
            
            // Setup FPS monitoring after viewport is created
            Alpine.effect(() => {
                if (this.modules.viewport?.container) {
                    this.modules.viewport.container.addEventListener('fps-update', (e) => {
                        this.performance.updateFPS(e.detail);
                    });
                }
            });
            
            // Load initial data
            await this.loadExperiments();
        },
        
        async loadExperiments() {
            try {
                this.ui.loading = true;
                this.data.experiments = await this.dataManager.getExperiments();
            } finally {
                this.ui.loading = false;
            }
        },
        
        async loadExperiment() {
            if (!this.data.currentExperiment) return;
            
            try {
                this.ui.loading = true;
                const analysis = await this.dataManager.getAnalysis(this.data.currentExperiment);
                this.data.episodes = analysis.episodes;
                
                if (this.modules.charts) {
                    this.modules.charts.updateLearningCurve(analysis.aggregate.learning_curve);
                }
            } finally {
                this.ui.loading = false;
            }
        },
        
        async loadEpisode(episodeId) {
            try {
                this.ui.loading = true;
                this.playback.pause();
                
                const data = await this.dataManager.getEpisodeData(
                    this.data.currentExperiment, 
                    episodeId
                );
                
                // Convert to typed arrays for performance
                if (data.trajectory.x && !(data.trajectory.x instanceof Float32Array)) {
                    data.trajectory.x = new Float32Array(data.trajectory.x);
                    data.trajectory.y = new Float32Array(data.trajectory.y);
                    data.rewards = new Float32Array(data.rewards);
                    data.values = new Float32Array(data.values || new Array(data.trajectory.x.length).fill(0));
                }
                
                this.data.currentEpisode = episodeId;
                this.data.episodeData = data;
                this.playback.duration = data.trajectory.x.length;
                this.playback.reset();
                
                // Initialize visualizations
                this.modules.viewport?.setData(data);
                this.modules.charts?.setEpisodeData(data);
                this.modules.neural?.setData(data);
                
            } finally {
                this.ui.loading = false;
            }
        },
        
        updateVisualizations(time) {
            // This method is now unused - updates happen directly in playback callback
            // Kept for compatibility
        },
        
        togglePlayback() {
            if (this.playback.playing) {
                this.playback.pause();
            } else {
                this.playback.play();
            }
        },
        
        step(delta) {
            this.playback.pause();
            this.playback.seek(this.playback.time + delta);
        },
        
        async exportData() {
            const data = {
                experiment: this.data.currentExperiment,
                episode: this.data.currentEpisode,
                performance: {
                    fps: this.performance.fps,
                    quality: this.ui.quality
                },
                data: this.data.episodeData
            };
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.data.currentExperiment}_episode_${this.data.currentEpisode}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    }));
    
    // Simple timeline component
    Alpine.data('timeline', () => ({
        playback: Alpine.store('playback'),
        utils: Alpine.store('utils'),
        
        init() {
            // Removed Alpine.effect to prevent excessive DOM updates
        }
    }));
    
    // Episode list component
    Alpine.data('episodes', () => ({
        data: Alpine.store('data'),
        
        selectEpisode(id) {
            this.$dispatch('load-episode', { id });
        }
    }));
    
    // Performance monitor component
    Alpine.data('performanceMonitor', () => ({
        performance: Alpine.store('performance'),
        ui: Alpine.store('ui'),
        
        get displayFPS() {
            return `${this.performance.fps} FPS`;
        },
        
        get displayFrameTime() {
            return `${this.performance.frameTime.toFixed(1)}ms`;
        },
        
        get qualityColor() {
            return {
                'low': '#ff5252',
                'medium': '#ffeb3b',
                'high': '#00e676'
            }[this.ui.quality];
        }
    }));
});