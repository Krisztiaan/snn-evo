/**
 * keywords: [charts, graphs, metrics, analysis, scientific, visualization]
 * 
 * Chart manager for scientific data visualization and analysis
 */

export class ChartManager {
    constructor() {
        this.charts = {};
        this.data = null;
        this.currentTime = 0;
        
        // Chart configurations
        this.chartConfigs = {
            reward: {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Time (ms)', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            title: { display: true, text: 'Cumulative Reward', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    elements: {
                        point: { radius: 0 },
                        line: { borderWidth: 2 }
                    }
                }
            },
            value: {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Time (ms)', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            title: { display: true, text: 'Value Estimate', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    elements: {
                        point: { radius: 0 },
                        line: { borderWidth: 2 }
                    }
                }
            },
            connectivity: {
                type: 'bar',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            title: { display: true, text: 'Connection Strength', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            title: { display: true, text: 'Count', color: '#888' },
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            }
        };
        
        this.init();
    }
    
    init() {
        // Initialize charts
        this.initRewardChart();
        this.initValueChart();
        this.initConnectivityChart();
        
        // Setup custom Chart.js plugins
        this.setupChartPlugins();
    }
    
    initRewardChart() {
        const canvas = document.getElementById('reward-chart');
        const ctx = canvas.getContext('2d');
        
        this.charts.reward = new Chart(ctx, {
            type: this.chartConfigs.reward.type,
            data: {
                datasets: [{
                    label: 'Cumulative Reward',
                    data: [],
                    borderColor: '#4a9eff',
                    backgroundColor: 'rgba(74, 158, 255, 0.1)',
                    fill: true
                }, {
                    label: 'Reward Events',
                    data: [],
                    borderColor: '#ffeb3b',
                    backgroundColor: '#ffeb3b',
                    type: 'scatter',
                    pointRadius: 4
                }]
            },
            options: this.chartConfigs.reward.options
        });
    }
    
    initValueChart() {
        const canvas = document.getElementById('value-chart');
        const ctx = canvas.getContext('2d');
        
        this.charts.value = new Chart(ctx, {
            type: this.chartConfigs.value.type,
            data: {
                datasets: [{
                    label: 'Value Function',
                    data: [],
                    borderColor: '#00e676',
                    backgroundColor: 'rgba(0, 230, 118, 0.1)',
                    fill: true
                }, {
                    label: 'TD Error',
                    data: [],
                    borderColor: '#ff5252',
                    backgroundColor: 'rgba(255, 82, 82, 0.1)',
                    fill: true,
                    hidden: true
                }]
            },
            options: this.chartConfigs.value.options
        });
    }
    
    initConnectivityChart() {
        const canvas = document.getElementById('connectivity-chart');
        const ctx = canvas.getContext('2d');
        
        this.charts.connectivity = new Chart(ctx, {
            type: this.chartConfigs.connectivity.type,
            data: {
                labels: [],
                datasets: [{
                    label: 'Weight Distribution',
                    data: [],
                    backgroundColor: '#9c27b0'
                }]
            },
            options: this.chartConfigs.connectivity.options
        });
    }
    
    setupChartPlugins() {
        // Custom plugin for time marker
        const timeMarkerPlugin = {
            id: 'timeMarker',
            afterDraw: (chart) => {
                if (!this.currentTime || !chart.scales.x) return;
                
                const ctx = chart.ctx;
                const xScale = chart.scales.x;
                const yScale = chart.scales.y;
                const x = xScale.getPixelForValue(this.currentTime);
                
                if (x >= xScale.left && x <= xScale.right) {
                    ctx.save();
                    ctx.strokeStyle = '#4a9eff';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    
                    ctx.beginPath();
                    ctx.moveTo(x, yScale.top);
                    ctx.lineTo(x, yScale.bottom);
                    ctx.stroke();
                    
                    ctx.restore();
                }
            }
        };
        
        // Register plugin globally
        Chart.register(timeMarkerPlugin);
    }
    
    setEpisodeData(data) {
        this.data = data;
        this.currentTime = 0;
        
        // Process and update all charts
        this.updateRewardChart();
        this.updateValueChart();
        this.updateConnectivityChart();
        
        // Compute additional metrics
        this.computeMetrics();
    }
    
    updateRewardChart() {
        if (!this.data || !this.data.rewards) return;
        
        // Calculate cumulative reward
        const cumulative = [];
        let sum = 0;
        
        for (let i = 0; i < this.data.rewards.length; i++) {
            sum += this.data.rewards[i];
            cumulative.push({ x: i, y: sum });
        }
        
        // Find reward events
        const rewardEvents = [];
        this.data.rewards.forEach((reward, i) => {
            if (reward > 0) {
                rewardEvents.push({ x: i, y: sum });
            }
        });
        
        // Update chart
        this.charts.reward.data.datasets[0].data = cumulative;
        this.charts.reward.data.datasets[1].data = rewardEvents;
        
        // Decimate if too many points
        if (cumulative.length > 5000) {
            this.charts.reward.data.datasets[0].data = this.decimateData(cumulative, 1000);
        }
        
        this.charts.reward.update('none');
    }
    
    updateValueChart() {
        if (!this.data || !this.data.values) return;
        
        // Convert to chart format
        const valueData = this.data.values.map((v, i) => ({ x: i, y: v }));
        
        // Calculate TD error if possible
        const tdError = [];
        for (let i = 0; i < this.data.values.length - 1; i++) {
            const reward = this.data.rewards[i] || 0;
            const td = reward + 0.99 * this.data.values[i + 1] - this.data.values[i];
            tdError.push({ x: i, y: td });
        }
        
        // Update chart
        this.charts.value.data.datasets[0].data = this.decimateData(valueData, 1000);
        this.charts.value.data.datasets[1].data = this.decimateData(tdError, 1000);
        
        this.charts.value.update('none');
    }
    
    updateConnectivityChart() {
        if (!this.data || !this.data.neural || !this.data.neural.weights) return;
        
        // Compute weight distribution
        const weights = [];
        const weightMatrix = this.data.neural.weights;
        
        for (let i = 0; i < weightMatrix.length; i++) {
            for (let j = 0; j < weightMatrix[i].length; j++) {
                if (weightMatrix[i][j] !== 0) {
                    weights.push(weightMatrix[i][j]);
                }
            }
        }
        
        // Create histogram
        const histogram = this.createHistogram(weights, 20);
        
        // Update chart
        this.charts.connectivity.data.labels = histogram.labels;
        this.charts.connectivity.data.datasets[0].data = histogram.counts;
        
        this.charts.connectivity.update('none');
    }
    
    updateLearningCurve(learningCurve) {
        // This could be a separate chart or update existing ones
        // For now, we'll store it for analysis
        this.learningCurve = learningCurve;
    }
    
    updateCurrentTime(time) {
        this.currentTime = time;
        
        // Trigger chart redraws to update time marker
        Object.values(this.charts).forEach(chart => {
            chart.update('none');
        });
    }
    
    computeMetrics() {
        if (!this.data) return;
        
        const metrics = {
            totalReward: 0,
            rewardRate: 0,
            explorationCoverage: 0,
            averageValue: 0,
            valueVariance: 0,
            peakFiringRate: 0,
            averageFiringRate: 0
        };
        
        // Reward metrics
        metrics.totalReward = this.data.rewards.reduce((a, b) => a + b, 0);
        metrics.rewardRate = metrics.totalReward / this.data.rewards.length * 1000; // per second
        
        // Value function metrics
        if (this.data.values) {
            metrics.averageValue = ss.mean(this.data.values);
            metrics.valueVariance = ss.variance(this.data.values);
        }
        
        // Exploration metrics
        const positions = new Set();
        for (let i = 0; i < this.data.trajectory.x.length; i++) {
            const key = `${Math.floor(this.data.trajectory.x[i])},${Math.floor(this.data.trajectory.y[i])}`;
            positions.add(key);
        }
        metrics.explorationCoverage = positions.size / 100; // Assuming 10x10 grid
        
        // Neural metrics
        if (this.data.neural && this.data.neural.spikes) {
            const firingRates = this.calculateFiringRates(this.data.neural.spikes);
            metrics.peakFiringRate = ss.max(firingRates);
            metrics.averageFiringRate = ss.mean(firingRates);
        }
        
        this.metrics = metrics;
        
        // Emit metrics update event
        window.dispatchEvent(new CustomEvent('metrics-updated', { detail: metrics }));
    }
    
    calculateFiringRates(spikes) {
        const rates = [];
        const windowSize = 100; // 100ms window
        
        for (let neuron = 0; neuron < spikes[0].length; neuron++) {
            let maxRate = 0;
            
            for (let t = 0; t < spikes.length - windowSize; t += 50) {
                let count = 0;
                for (let w = 0; w < windowSize; w++) {
                    if (spikes[t + w] && spikes[t + w][neuron]) {
                        count++;
                    }
                }
                const rate = count / (windowSize / 1000); // Hz
                maxRate = Math.max(maxRate, rate);
            }
            
            rates.push(maxRate);
        }
        
        return rates;
    }
    
    createHistogram(data, bins) {
        const min = ss.min(data);
        const max = ss.max(data);
        const binWidth = (max - min) / bins;
        
        const labels = [];
        const counts = new Array(bins).fill(0);
        
        // Create bin labels and count
        for (let i = 0; i < bins; i++) {
            const binMin = min + i * binWidth;
            const binMax = binMin + binWidth;
            labels.push(binMin.toFixed(3));
            
            data.forEach(value => {
                if (value >= binMin && value < binMax) {
                    counts[i]++;
                }
            });
        }
        
        return { labels, counts };
    }
    
    decimateData(data, targetPoints) {
        if (data.length <= targetPoints) return data;
        
        const factor = Math.ceil(data.length / targetPoints);
        const decimated = [];
        
        for (let i = 0; i < data.length; i += factor) {
            // Take the average of the window
            let sumX = 0, sumY = 0, count = 0;
            
            for (let j = i; j < Math.min(i + factor, data.length); j++) {
                sumX += data[j].x;
                sumY += data[j].y;
                count++;
            }
            
            decimated.push({
                x: sumX / count,
                y: sumY / count
            });
        }
        
        return decimated;
    }
    
    getAnalysis() {
        return {
            metrics: this.metrics,
            learningCurve: this.learningCurve,
            charts: {
                reward: this.charts.reward.data,
                value: this.charts.value.data,
                connectivity: this.charts.connectivity.data
            }
        };
    }
    
    dispose() {
        // Destroy all charts
        Object.values(this.charts).forEach(chart => {
            chart.destroy();
        });
        this.charts = {};
    }
}