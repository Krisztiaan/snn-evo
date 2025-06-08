/**
 * keywords: [timeline, navigation, scrubbing, heatmap, events]
 * 
 * Timeline controller for temporal navigation and event visualization
 */

export class TimelineController {
    constructor(containerId, onChangeCallback) {
        this.container = document.getElementById(containerId);
        this.progress = document.getElementById('timeline-progress');
        this.handle = document.getElementById('timeline-handle');
        this.heatmapCanvas = document.getElementById('timeline-heatmap');
        this.heatmapCtx = this.heatmapCanvas.getContext('2d');
        
        this.onChangeCallback = onChangeCallback;
        
        this.data = null;
        this.currentTime = 0;
        this.duration = 0;
        
        // Interaction state
        this.isDragging = false;
        this.isHovering = false;
        
        // Event markers
        this.events = [];
        this.heatmapData = null;
        
        this.init();
    }
    
    init() {
        // Setup event listeners
        this.setupInteraction();
        
        // Setup heatmap canvas
        this.resizeHeatmap();
        window.addEventListener('resize', () => this.resizeHeatmap());
    }
    
    setupInteraction() {
        // Mouse events
        this.container.addEventListener('mousedown', this.onMouseDown.bind(this));
        document.addEventListener('mousemove', this.onMouseMove.bind(this));
        document.addEventListener('mouseup', this.onMouseUp.bind(this));
        
        // Click to seek
        this.container.addEventListener('click', this.onClick.bind(this));
        
        // Touch events for mobile
        this.container.addEventListener('touchstart', this.onTouchStart.bind(this));
        document.addEventListener('touchmove', this.onTouchMove.bind(this));
        document.addEventListener('touchend', this.onTouchEnd.bind(this));
        
        // Hover effects
        this.container.addEventListener('mouseenter', () => {
            this.isHovering = true;
            this.container.style.cursor = 'pointer';
        });
        
        this.container.addEventListener('mouseleave', () => {
            this.isHovering = false;
            if (!this.isDragging) {
                this.container.style.cursor = 'default';
            }
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', this.onKeyDown.bind(this));
    }
    
    setData(data) {
        this.data = data;
        this.duration = data.trajectory.x.length;
        this.currentTime = 0;
        
        // Extract events
        this.extractEvents();
        
        // Generate heatmap
        this.generateHeatmap();
        
        // Update UI
        this.update();
    }
    
    extractEvents() {
        this.events = [];
        
        if (!this.data) return;
        
        // Reward events
        this.data.rewards.forEach((reward, time) => {
            if (reward > 0) {
                this.events.push({
                    time,
                    type: 'reward',
                    value: reward,
                    color: '#ffeb3b'
                });
            }
        });
        
        // High value states
        if (this.data.values) {
            const valueThreshold = ss.quantile(this.data.values, 0.9);
            this.data.values.forEach((value, time) => {
                if (value > valueThreshold) {
                    this.events.push({
                        time,
                        type: 'highValue',
                        value,
                        color: '#00e676'
                    });
                }
            });
        }
        
        // High neural activity
        if (this.data.neural && this.data.neural.spikes) {
            const activityWindow = 50;
            for (let t = 0; t < this.data.neural.spikes.length - activityWindow; t += activityWindow) {
                let spikeCount = 0;
                for (let w = 0; w < activityWindow; w++) {
                    if (this.data.neural.spikes[t + w]) {
                        spikeCount += this.data.neural.spikes[t + w].reduce((a, b) => a + b, 0);
                    }
                }
                
                if (spikeCount > 100) { // Threshold for high activity
                    this.events.push({
                        time: t,
                        type: 'highActivity',
                        value: spikeCount,
                        color: '#ff5252'
                    });
                }
            }
        }
    }
    
    generateHeatmap() {
        if (!this.data) return;
        
        const width = this.heatmapCanvas.width;
        const binSize = Math.ceil(this.duration / width);
        
        // Initialize heatmap data
        this.heatmapData = new Float32Array(width);
        
        // Aggregate data into bins
        for (let bin = 0; bin < width; bin++) {
            const startTime = bin * binSize;
            const endTime = Math.min((bin + 1) * binSize, this.duration);
            
            let value = 0;
            let count = 0;
            
            for (let t = startTime; t < endTime; t++) {
                // Combine multiple signals
                if (this.data.rewards[t] > 0) value += 10;
                if (this.data.values && this.data.values[t]) value += this.data.values[t];
                count++;
            }
            
            this.heatmapData[bin] = count > 0 ? value / count : 0;
        }
        
        // Normalize
        const maxValue = Math.max(...this.heatmapData);
        if (maxValue > 0) {
            for (let i = 0; i < this.heatmapData.length; i++) {
                this.heatmapData[i] /= maxValue;
            }
        }
        
        this.renderHeatmap();
    }
    
    renderHeatmap() {
        const width = this.heatmapCanvas.width;
        const height = this.heatmapCanvas.height;
        
        // Clear canvas
        this.heatmapCtx.clearRect(0, 0, width, height);
        
        // Create gradient
        const gradient = this.heatmapCtx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#4a9eff');
        gradient.addColorStop(0.5, '#00e676');
        gradient.addColorStop(1, '#1a1a1a');
        
        // Draw heatmap bars
        for (let i = 0; i < width; i++) {
            if (this.heatmapData[i] > 0) {
                const barHeight = this.heatmapData[i] * height;
                
                this.heatmapCtx.fillStyle = gradient;
                this.heatmapCtx.fillRect(i, height - barHeight, 1, barHeight);
            }
        }
        
        // Draw event markers
        this.events.forEach(event => {
            const x = (event.time / this.duration) * width;
            
            this.heatmapCtx.strokeStyle = event.color;
            this.heatmapCtx.lineWidth = 2;
            this.heatmapCtx.beginPath();
            this.heatmapCtx.moveTo(x, 0);
            this.heatmapCtx.lineTo(x, height);
            this.heatmapCtx.stroke();
        });
    }
    
    setTime(time) {
        this.currentTime = Math.max(0, Math.min(time, this.duration - 1));
        this.update();
        
        if (this.onChangeCallback) {
            this.onChangeCallback(this.currentTime);
        }
    }
    
    update() {
        if (!this.duration) return;
        
        const progress = this.currentTime / this.duration;
        const percentage = progress * 100;
        
        this.progress.style.width = `${percentage}%`;
        
        // Show tooltip on hover
        if (this.isHovering) {
            this.showTooltip();
        }
    }
    
    showTooltip() {
        const tooltip = document.getElementById('tooltip');
        const rect = this.container.getBoundingClientRect();
        const x = (this.currentTime / this.duration) * rect.width + rect.left;
        const y = rect.top - 30;
        
        // Find nearby events
        const nearbyEvents = this.events.filter(e => 
            Math.abs(e.time - this.currentTime) < 50
        );
        
        let content = `Time: ${Math.floor(this.currentTime)}ms`;
        if (nearbyEvents.length > 0) {
            content += '<br>';
            nearbyEvents.forEach(e => {
                content += `<br>${e.type}: ${e.value.toFixed(2)}`;
            });
        }
        
        tooltip.innerHTML = content;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
        tooltip.classList.add('visible');
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        tooltip.classList.remove('visible');
    }
    
    // Event handlers
    
    onMouseDown(e) {
        if (e.target === this.handle || e.target === this.container) {
            this.isDragging = true;
            this.container.style.cursor = 'grabbing';
            e.preventDefault();
        }
    }
    
    onMouseMove(e) {
        if (!this.isDragging) return;
        
        const rect = this.container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const progress = Math.max(0, Math.min(1, x / rect.width));
        
        this.setTime(progress * this.duration);
    }
    
    onMouseUp() {
        if (this.isDragging) {
            this.isDragging = false;
            this.container.style.cursor = this.isHovering ? 'pointer' : 'default';
        }
    }
    
    onClick(e) {
        if (e.target === this.handle) return;
        
        const rect = this.container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const progress = Math.max(0, Math.min(1, x / rect.width));
        
        this.setTime(progress * this.duration);
    }
    
    onTouchStart(e) {
        const touch = e.touches[0];
        this.onMouseDown({ 
            target: e.target, 
            clientX: touch.clientX, 
            preventDefault: () => e.preventDefault() 
        });
    }
    
    onTouchMove(e) {
        const touch = e.touches[0];
        this.onMouseMove({ clientX: touch.clientX });
    }
    
    onTouchEnd() {
        this.onMouseUp();
    }
    
    onKeyDown(e) {
        if (!this.data) return;
        
        const step = e.shiftKey ? 100 : 10;
        
        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                this.setTime(this.currentTime - step);
                break;
                
            case 'ArrowRight':
                e.preventDefault();
                this.setTime(this.currentTime + step);
                break;
                
            case 'Home':
                e.preventDefault();
                this.setTime(0);
                break;
                
            case 'End':
                e.preventDefault();
                this.setTime(this.duration - 1);
                break;
                
            case 'PageUp':
                e.preventDefault();
                this.jumpToEvent('prev');
                break;
                
            case 'PageDown':
                e.preventDefault();
                this.jumpToEvent('next');
                break;
        }
    }
    
    jumpToEvent(direction) {
        if (this.events.length === 0) return;
        
        let targetEvent = null;
        
        if (direction === 'next') {
            targetEvent = this.events.find(e => e.time > this.currentTime);
            if (!targetEvent) targetEvent = this.events[0]; // Loop to beginning
        } else {
            for (let i = this.events.length - 1; i >= 0; i--) {
                if (this.events[i].time < this.currentTime) {
                    targetEvent = this.events[i];
                    break;
                }
            }
            if (!targetEvent) targetEvent = this.events[this.events.length - 1]; // Loop to end
        }
        
        if (targetEvent) {
            this.setTime(targetEvent.time);
        }
    }
    
    resizeHeatmap() {
        const rect = this.container.getBoundingClientRect();
        this.heatmapCanvas.width = rect.width;
        
        if (this.heatmapData) {
            this.renderHeatmap();
        }
    }
    
    getProgress() {
        return this.duration > 0 ? this.currentTime / this.duration : 0;
    }
    
    getDuration() {
        return this.duration;
    }
    
    getCurrentTime() {
        return this.currentTime;
    }
}