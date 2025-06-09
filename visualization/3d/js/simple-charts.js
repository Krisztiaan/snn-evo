/**
 * keywords: [charts, simple, visualization, reactive]
 * 
 * Simplified chart management using Chart.js reactivity
 */

export class SimpleCharts {
    constructor() {
        this.charts = {};
        this.lastUpdateTime = -1;
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 }, // Disable animations for performance
            scales: {
                x: { 
                    grid: { color: '#333' },
                    ticks: { color: '#888' }
                },
                y: { 
                    grid: { color: '#333' },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: true }
            }
        };
    }
    
    // Single method to create any chart
    createChart(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        // Destroy existing chart
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        this.charts[canvasId] = new Chart(canvas, {
            ...config,
            options: { ...this.defaultOptions, ...config.options }
        });
        
        return this.charts[canvasId];
    }
    
    // Update learning curve
    updateLearningCurve(data) {
        if (!data || data.length === 0) return;
        
        this.createChart('reward-chart', {
            type: 'line',
            data: {
                labels: data.map((_, i) => i + 1),
                datasets: [{
                    data: data,
                    borderColor: '#4a9eff',
                    borderWidth: 2,
                    tension: 0.4
                }]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: 'Episode', color: '#888' } },
                    y: { title: { display: true, text: 'Avg Reward', color: '#888' } }
                }
            }
        });
    }
    
    // Set episode data - create simple value chart
    setEpisodeData(data) {
        if (!data) return;
        
        // Simple cumulative reward chart
        const cumRewards = [];
        let sum = 0;
        for (const reward of data.rewards) {
            sum += reward;
            cumRewards.push(sum);
        }
        
        this.createChart('value-chart', {
            type: 'line',
            data: {
                labels: cumRewards.map((_, i) => i),
                datasets: [{
                    data: cumRewards,
                    borderColor: '#00e676',
                    borderWidth: 1,
                    pointRadius: 0
                }]
            },
            options: {
                scales: {
                    x: { display: false },
                    y: { title: { display: true, text: 'Cumulative Reward', color: '#888' } }
                }
            }
        });
        
        // Store data for time updates
        this.episodeData = data;
        this.cumRewards = cumRewards;
    }
    
    // Update current time marker
    updateTime(time) {
        if (!this.charts['value-chart'] || !this.cumRewards) return;
        
        // Only update if time actually changed significantly
        if (Math.abs(time - this.lastUpdateTime) < 10) return;
        this.lastUpdateTime = time;
        
        // For now, skip the annotation plugin which may be causing issues
        // We can implement a custom overlay later if needed
    }
    
    // Cleanup
    destroy() {
        Object.values(this.charts).forEach(chart => chart.destroy());
        this.charts = {};
    }
}