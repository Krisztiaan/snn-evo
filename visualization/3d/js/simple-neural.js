/**
 * keywords: [neural, visualizer, simple, spike, raster]
 * 
 * Simplified neural activity visualizer
 */

export class SimpleNeuralVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.ctx = this.canvas.getContext('2d');
        this.data = null;
        
        // Create offscreen canvas for base rendering
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCtx = this.offscreenCanvas.getContext('2d');
        this.needsBaseRender = true;
        this.lastTimelineX = -1;
        
        // Set canvas size
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }
    
    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = Math.min(200, rect.height);
        
        // Resize offscreen canvas too
        this.offscreenCanvas.width = this.canvas.width;
        this.offscreenCanvas.height = this.canvas.height;
        this.needsBaseRender = true;
    }
    
    setData(episodeData) {
        this.data = episodeData;
        this.needsBaseRender = true;
        this.lastTimelineX = -1;
        
        if (!this.data?.neural?.spikes) {
            this.clear();
            return;
        }
        this.renderBase();
    }
    
    clear() {
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    renderBase() {
        // Clear offscreen canvas
        this.offscreenCtx.fillStyle = '#0a0a0a';
        this.offscreenCtx.fillRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
        
        const spikes = this.data.neural.spikes;
        if (!spikes || spikes.length === 0) return;
        
        const numNeurons = Math.min(spikes[0].length, 100); // Limit neurons for performance
        const numTimesteps = Math.min(spikes.length, 1000); // Limit for performance
        
        const neuronHeight = Math.max(1, this.offscreenCanvas.height / numNeurons);
        const timeWidth = this.offscreenCanvas.width / numTimesteps;
        
        // Draw spikes to offscreen canvas
        this.offscreenCtx.fillStyle = '#4a9eff';
        for (let t = 0; t < numTimesteps; t++) {
            for (let n = 0; n < numNeurons; n++) {
                if (spikes[t][n] > 0) {
                    this.offscreenCtx.fillRect(
                        t * timeWidth,
                        n * neuronHeight,
                        Math.max(1, timeWidth - 1),
                        Math.max(1, neuronHeight - 1)
                    );
                }
            }
        }
        
        this.needsBaseRender = false;
        // Copy to main canvas
        this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    }
    
    updateTime(time) {
        if (!this.data?.neural?.spikes) return;
        
        // Calculate timeline position
        const numTimesteps = Math.min(this.data.neural.spikes.length, 1000);
        const timeWidth = this.canvas.width / numTimesteps;
        const x = Math.min(time, numTimesteps - 1) * timeWidth;
        
        // Only update if position changed significantly
        if (Math.abs(x - this.lastTimelineX) < 1) return;
        this.lastTimelineX = x;
        
        // Render base if needed
        if (this.needsBaseRender) {
            this.renderBase();
        } else {
            // Just copy the base image
            this.ctx.drawImage(this.offscreenCanvas, 0, 0);
        }
        
        // Add time line on top
        this.ctx.strokeStyle = '#ff5252';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(x, 0);
        this.ctx.lineTo(x, this.canvas.height);
        this.ctx.stroke();
    }
}