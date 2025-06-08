/**
 * keywords: [neural, visualizer, spike, raster, connectome, spectrogram]
 * 
 * Neural activity visualizer with spike raster plots and connectivity visualization
 */

export class NeuralVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById('spike-raster');
        this.ctx = this.canvas.getContext('2d');
        
        this.data = null;
        this.currentTime = 0;
        
        // Visualization parameters
        this.params = {
            timeWindow: 1000,      // Show 1 second of data
            neuronHeight: 1,       // Pixels per neuron
            maxNeurons: 100,       // Maximum neurons to display
            colorMap: 'viridis',   // Color scheme
            showConnectivity: true,
            connectivityThreshold: 0.1
        };
        
        // Performance optimization
        this.renderBuffer = null;
        this.needsRedraw = true;
        this.lastDrawTime = 0;
        this.drawInterval = 50; // Minimum ms between redraws
        
        // Connectivity matrix
        this.connectivityCanvas = null;
        this.connectivityCtx = null;
        
        this.init();
    }
    
    init() {
        // Setup canvas
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Create connectivity canvas
        this.connectivityCanvas = document.createElement('canvas');
        this.connectivityCanvas.width = 256;
        this.connectivityCanvas.height = 256;
        this.connectivityCtx = this.connectivityCanvas.getContext('2d');
        
        // Setup render buffer for performance
        this.renderBuffer = document.createElement('canvas');
        this.renderBuffer.width = this.canvas.width;
        this.renderBuffer.height = this.canvas.height;
        this.bufferCtx = this.renderBuffer.getContext('2d');
        
        // Color maps
        this.colorMaps = {
            viridis: this.generateViridisColorMap(),
            plasma: this.generatePlasmaColorMap(),
            inferno: this.generateInfernoColorMap(),
            hot: this.generateHotColorMap()
        };
    }
    
    resizeCanvas() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        if (this.renderBuffer) {
            this.renderBuffer.width = rect.width;
            this.renderBuffer.height = rect.height;
        }
        
        this.needsRedraw = true;
    }
    
    setData(neuralData) {
        this.data = neuralData;
        this.currentTime = 0;
        this.needsRedraw = true;
        
        // Process connectivity data if available
        if (neuralData && neuralData.weights && neuralData.weights.length > 0) {
            this.processConnectivity(neuralData.weights);
        }
        
        // Pre-compute spike density for efficient rendering
        this.computeSpikeDensity();
    }
    
    computeSpikeDensity() {
        if (!this.data || !this.data.spikes || this.data.spikes.length === 0) return;
        
        // Compute spike density in time bins for faster rendering
        const binSize = 10; // 10ms bins
        const numBins = Math.ceil(this.data.spikes.length / binSize);
        const numNeurons = this.data.spikes[0] ? this.data.spikes[0].length : 0;
        
        if (numNeurons === 0) return;
        
        this.spikeDensity = new Float32Array(numBins * numNeurons);
        
        for (let bin = 0; bin < numBins; bin++) {
            const startTime = bin * binSize;
            const endTime = Math.min((bin + 1) * binSize, this.data.spikes.length);
            
            for (let neuron = 0; neuron < numNeurons; neuron++) {
                let spikeCount = 0;
                for (let t = startTime; t < endTime; t++) {
                    if (this.data.spikes[t] && this.data.spikes[t][neuron]) {
                        spikeCount++;
                    }
                }
                this.spikeDensity[bin * numNeurons + neuron] = spikeCount / binSize;
            }
        }
    }
    
    processConnectivity(weights) {
        // Validate input
        if (!weights || weights.length === 0) {
            console.warn('No weight data available for connectivity visualization');
            return;
        }
        
        // Create connectivity matrix visualization
        const numNeurons = Math.min(weights.length, 256);
        if (numNeurons === 0) return;
        
        const imageData = this.connectivityCtx.createImageData(numNeurons, numNeurons);
        const data = imageData.data;
        
        // Find weight range for normalization
        let minWeight = Infinity;
        let maxWeight = -Infinity;
        
        for (let i = 0; i < numNeurons; i++) {
            for (let j = 0; j < numNeurons; j++) {
                if (weights[i] && weights[i][j]) {
                    minWeight = Math.min(minWeight, weights[i][j]);
                    maxWeight = Math.max(maxWeight, weights[i][j]);
                }
            }
        }
        
        // Render connectivity matrix
        for (let i = 0; i < numNeurons; i++) {
            for (let j = 0; j < numNeurons; j++) {
                const idx = (i * numNeurons + j) * 4;
                const weight = weights[i] && weights[i][j] ? weights[i][j] : 0;
                const normalized = (weight - minWeight) / (maxWeight - minWeight);
                
                // Color based on weight strength
                if (Math.abs(weight) < this.params.connectivityThreshold) {
                    // Below threshold - dark
                    data[idx] = 20;
                    data[idx + 1] = 20;
                    data[idx + 2] = 20;
                } else if (weight > 0) {
                    // Excitatory - red
                    data[idx] = 255 * normalized;
                    data[idx + 1] = 50;
                    data[idx + 2] = 50;
                } else {
                    // Inhibitory - blue
                    data[idx] = 50;
                    data[idx + 1] = 50;
                    data[idx + 2] = 255 * Math.abs(normalized);
                }
                data[idx + 3] = 255;
            }
        }
        
        this.connectivityCtx.putImageData(imageData, 0, 0);
    }
    
    update(time) {
        this.currentTime = time;
        
        // Throttle updates for performance
        const now = performance.now();
        if (now - this.lastDrawTime < this.drawInterval) {
            return;
        }
        this.lastDrawTime = now;
        
        this.needsRedraw = true;
        this.render();
    }
    
    render() {
        if (!this.needsRedraw || !this.data) return;
        
        const ctx = this.bufferCtx;
        const width = this.renderBuffer.width;
        const height = this.renderBuffer.height;
        
        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
        
        // Render based on visualization mode
        if (this.data.spikes) {
            this.renderSpikeRaster(ctx);
        }
        
        // Copy buffer to main canvas
        this.ctx.drawImage(this.renderBuffer, 0, 0);
        
        this.needsRedraw = false;
    }
    
    renderSpikeRaster(ctx) {
        const width = this.renderBuffer.width;
        const height = this.renderBuffer.height;
        
        // Check if we have spike data
        if (!this.data.spikes || this.data.spikes.length === 0) {
            // Draw placeholder message
            ctx.fillStyle = '#444';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('No neural data available', width / 2, height / 2);
            return;
        }
        
        // Calculate visible time range
        const startTime = Math.max(0, this.currentTime - this.params.timeWindow / 2);
        const endTime = Math.min(this.data.spikes.length, 
                                this.currentTime + this.params.timeWindow / 2);
        
        // Calculate neuron range to display
        const totalNeurons = this.data.spikes[0] ? this.data.spikes[0].length : 0;
        if (totalNeurons === 0) return;
        
        const displayNeurons = Math.min(totalNeurons, this.params.maxNeurons);
        const neuronStep = Math.ceil(totalNeurons / displayNeurons);
        
        // Calculate pixel dimensions
        const timeScale = width / this.params.timeWindow;
        const neuronHeight = Math.max(1, height / displayNeurons);
        
        // Render spike raster
        for (let nIdx = 0; nIdx < displayNeurons; nIdx++) {
            const neuronId = nIdx * neuronStep;
            const y = nIdx * neuronHeight;
            
            // Draw background gradient based on firing rate
            if (this.spikeDensity) {
                const gradient = ctx.createLinearGradient(0, y, width, y);
                
                for (let t = startTime; t < endTime; t += 50) {
                    const binIdx = Math.floor(t / 10);
                    const density = this.spikeDensity[binIdx * totalNeurons + neuronId] || 0;
                    const color = this.getColorFromDensity(density);
                    const x = (t - startTime) * timeScale / width;
                    gradient.addColorStop(x, color);
                }
                
                ctx.fillStyle = gradient;
                ctx.fillRect(0, y, width, neuronHeight);
            }
            
            // Draw individual spikes
            ctx.fillStyle = '#fff';
            for (let t = startTime; t < endTime; t++) {
                if (this.data.spikes[t] && this.data.spikes[t][neuronId]) {
                    const x = (t - startTime) * timeScale;
                    ctx.fillRect(x, y, 2, neuronHeight - 1);
                }
            }
        }
        
        // Draw time marker
        const markerX = (this.currentTime - startTime) * timeScale;
        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(markerX, 0);
        ctx.lineTo(markerX, height);
        ctx.stroke();
        
        // Draw scale
        this.drawScale(ctx, startTime, endTime, displayNeurons);
    }
    
    drawScale(ctx, startTime, endTime, numNeurons) {
        const width = this.renderBuffer.width;
        const height = this.renderBuffer.height;
        
        ctx.font = '10px monospace';
        ctx.fillStyle = '#888';
        
        // Time scale
        ctx.textAlign = 'left';
        ctx.fillText(`${startTime}ms`, 5, height - 5);
        ctx.textAlign = 'right';
        ctx.fillText(`${endTime}ms`, width - 5, height - 5);
        
        // Neuron scale
        ctx.textAlign = 'left';
        ctx.save();
        ctx.translate(5, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`${numNeurons} neurons`, 0, 0);
        ctx.restore();
    }
    
    getColorFromDensity(density) {
        // Convert spike density to color using selected colormap
        const colorMap = this.colorMaps[this.params.colorMap];
        const index = Math.floor(Math.min(0.99, density * 10) * 255);
        return colorMap[index];
    }
    
    generateViridisColorMap() {
        // Simplified viridis colormap
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const r = Math.floor(68 + 187 * t);
            const g = Math.floor(1 + 206 * t);
            const b = Math.floor(84 + 44 * (1 - t));
            colors.push(`rgb(${r},${g},${b})`);
        }
        return colors;
    }
    
    generatePlasmaColorMap() {
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const r = Math.floor(13 + 242 * t);
            const g = Math.floor(8 + 99 * t * (1 - t * 0.5));
            const b = Math.floor(135 + 92 * (1 - t));
            colors.push(`rgb(${r},${g},${b})`);
        }
        return colors;
    }
    
    generateInfernoColorMap() {
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const r = Math.floor(t * 255);
            const g = Math.floor(t * t * 200);
            const b = Math.floor(Math.pow(t, 3) * 100);
            colors.push(`rgb(${r},${g},${b})`);
        }
        return colors;
    }
    
    generateHotColorMap() {
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255;
            const r = Math.floor(Math.min(255, t * 512));
            const g = Math.floor(Math.max(0, Math.min(255, (t - 0.5) * 512)));
            const b = Math.floor(Math.max(0, (t - 0.75) * 1024));
            colors.push(`rgb(${r},${g},${b})`);
        }
        return colors;
    }
    
    // Public API methods
    
    setTimeWindow(ms) {
        this.params.timeWindow = ms;
        this.needsRedraw = true;
    }
    
    setMaxNeurons(count) {
        this.params.maxNeurons = count;
        this.needsRedraw = true;
    }
    
    setColorMap(name) {
        if (this.colorMaps[name]) {
            this.params.colorMap = name;
            this.needsRedraw = true;
        }
    }
    
    getConnectivityImage() {
        return this.connectivityCanvas.toDataURL();
    }
    
    dispose() {
        // Clean up resources
        this.data = null;
        this.spikeDensity = null;
    }
}