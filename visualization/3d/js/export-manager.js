/**
 * keywords: [export, data, csv, json, analysis, download]
 * 
 * Export manager for scientific data export and analysis
 */

export class ExportManager {
    constructor() {
        this.exportFormats = ['json', 'csv', 'matlab', 'numpy'];
    }
    
    /**
     * Export data to file
     */
    async exportToFile(data, filename, format = 'json') {
        switch (format) {
            case 'json':
                await this.exportJSON(data, filename);
                break;
            case 'csv':
                await this.exportCSV(data, filename);
                break;
            case 'matlab':
                await this.exportMatlab(data, filename);
                break;
            case 'numpy':
                await this.exportNumpy(data, filename);
                break;
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
    }
    
    /**
     * Export as JSON
     */
    async exportJSON(data, filename) {
        const jsonData = {
            metadata: {
                experiment: data.experiment,
                episode: data.episode,
                exportDate: new Date().toISOString(),
                version: '2.0.0'
            },
            trajectory: {
                x: data.data.trajectory.x,
                y: data.data.trajectory.y,
                timestamps: Array.from({length: data.data.trajectory.x.length}, (_, i) => i)
            },
            rewards: data.data.rewards,
            values: data.data.values,
            neural: {
                summary: this.summarizeNeuralData(data.data.neural)
            },
            analysis: data.analysis
        };
        
        const blob = new Blob([JSON.stringify(jsonData, null, 2)], 
                            { type: 'application/json' });
        this.downloadBlob(blob, `${filename}.json`);
    }
    
    /**
     * Export as CSV
     */
    async exportCSV(data, filename) {
        // Create multiple CSV files for different data types
        
        // Trajectory CSV
        const trajectoryCSV = this.createTrajectoryCSV(data.data);
        this.downloadBlob(
            new Blob([trajectoryCSV], { type: 'text/csv' }),
            `${filename}_trajectory.csv`
        );
        
        // Neural summary CSV
        if (data.data.neural) {
            const neuralCSV = this.createNeuralSummaryCSV(data.data.neural);
            this.downloadBlob(
                new Blob([neuralCSV], { type: 'text/csv' }),
                `${filename}_neural_summary.csv`
            );
        }
        
        // Analysis CSV
        if (data.analysis && data.analysis.metrics) {
            const analysisCSV = this.createAnalysisCSV(data.analysis);
            this.downloadBlob(
                new Blob([analysisCSV], { type: 'text/csv' }),
                `${filename}_analysis.csv`
            );
        }
    }
    
    /**
     * Export as MATLAB format
     */
    async exportMatlab(data, filename) {
        // Create MATLAB-compatible structure
        const matlabCode = `
% SNN Agent Visualization Data Export
% Generated: ${new Date().toISOString()}
% Experiment: ${data.experiment}
% Episode: ${data.episode}

% Trajectory data
trajectory.x = [${data.data.trajectory.x.join(', ')}];
trajectory.y = [${data.data.trajectory.y.join(', ')}];
trajectory.time = 1:${data.data.trajectory.x.length};

% Rewards
rewards = [${data.data.rewards.join(', ')}];

% Value function
values = [${data.data.values.join(', ')}];

% Analysis metrics
${this.generateMatlabMetrics(data.analysis)}

% Plotting example
figure;
subplot(2,2,1);
plot(trajectory.x, trajectory.y);
title('Agent Trajectory');
xlabel('X Position');
ylabel('Y Position');

subplot(2,2,2);
plot(trajectory.time, cumsum(rewards));
title('Cumulative Reward');
xlabel('Time (ms)');
ylabel('Reward');

subplot(2,2,3);
plot(trajectory.time, values);
title('Value Function');
xlabel('Time (ms)');
ylabel('Value Estimate');

subplot(2,2,4);
scatter(trajectory.x(rewards > 0), trajectory.y(rewards > 0), 'r', 'filled');
title('Reward Locations');
xlabel('X Position');
ylabel('Y Position');
`;
        
        const blob = new Blob([matlabCode], { type: 'text/plain' });
        this.downloadBlob(blob, `${filename}.m`);
    }
    
    /**
     * Export as NumPy format
     */
    async exportNumpy(data, filename) {
        // Create Python/NumPy compatible script
        const pythonCode = `
#!/usr/bin/env python3
"""
SNN Agent Visualization Data Export
Generated: ${new Date().toISOString()}
Experiment: ${data.experiment}
Episode: ${data.episode}
"""

import numpy as np
import matplotlib.pyplot as plt

# Trajectory data
trajectory_x = np.array([${data.data.trajectory.x.join(', ')}])
trajectory_y = np.array([${data.data.trajectory.y.join(', ')}])
time = np.arange(len(trajectory_x))

# Rewards
rewards = np.array([${data.data.rewards.join(', ')}])

# Value function
values = np.array([${data.data.values.join(', ')}])

# Save to .npz file
np.savez('${filename}.npz',
         trajectory_x=trajectory_x,
         trajectory_y=trajectory_y,
         time=time,
         rewards=rewards,
         values=values)

# Analysis
total_reward = np.sum(rewards)
reward_rate = total_reward / len(rewards) * 1000  # per second
avg_value = np.mean(values)
exploration_coverage = len(np.unique(np.column_stack([
    np.floor(trajectory_x), 
    np.floor(trajectory_y)
]), axis=0)) / 100  # assuming 10x10 grid

print(f"Total Reward: {total_reward}")
print(f"Reward Rate: {reward_rate:.2f} rewards/sec")
print(f"Average Value: {avg_value:.2f}")
print(f"Exploration Coverage: {exploration_coverage:.2%}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trajectory
axes[0, 0].plot(trajectory_x, trajectory_y, 'b-', alpha=0.6)
axes[0, 0].scatter(trajectory_x[rewards > 0], trajectory_y[rewards > 0], 
                   c='r', s=50, marker='*', label='Rewards')
axes[0, 0].set_title('Agent Trajectory')
axes[0, 0].set_xlabel('X Position')
axes[0, 0].set_ylabel('Y Position')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Cumulative reward
axes[0, 1].plot(time, np.cumsum(rewards))
axes[0, 1].set_title('Cumulative Reward')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].grid(True, alpha=0.3)

# Value function
axes[1, 0].plot(time, values)
axes[1, 0].set_title('Value Function')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Value Estimate')
axes[1, 0].grid(True, alpha=0.3)

# Heatmap of visits
heatmap = np.zeros((10, 10))
for x, y in zip(trajectory_x, trajectory_y):
    ix, iy = int(x), int(y)
    if 0 <= ix < 10 and 0 <= iy < 10:
        heatmap[iy, ix] += 1

im = axes[1, 1].imshow(heatmap, cmap='hot', origin='lower')
axes[1, 1].set_title('Visit Heatmap')
axes[1, 1].set_xlabel('X Position')
axes[1, 1].set_ylabel('Y Position')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('${filename}_analysis.png', dpi=150)
plt.show()
`;
        
        const blob = new Blob([pythonCode], { type: 'text/plain' });
        this.downloadBlob(blob, `${filename}_analysis.py`);
    }
    
    /**
     * Create trajectory CSV
     */
    createTrajectoryCSV(data) {
        let csv = 'time,x,y,reward,value\n';
        
        for (let i = 0; i < data.trajectory.x.length; i++) {
            csv += `${i},${data.trajectory.x[i]},${data.trajectory.y[i]},`;
            csv += `${data.rewards[i] || 0},${data.values[i] || 0}\n`;
        }
        
        return csv;
    }
    
    /**
     * Create neural summary CSV
     */
    createNeuralSummaryCSV(neuralData) {
        const summary = this.summarizeNeuralData(neuralData);
        
        let csv = 'metric,value\n';
        csv += `total_spikes,${summary.totalSpikes}\n`;
        csv += `avg_firing_rate,${summary.avgFiringRate}\n`;
        csv += `peak_firing_rate,${summary.peakFiringRate}\n`;
        csv += `active_neurons,${summary.activeNeurons}\n`;
        csv += `silent_neurons,${summary.silentNeurons}\n`;
        
        return csv;
    }
    
    /**
     * Create analysis CSV
     */
    createAnalysisCSV(analysis) {
        let csv = 'metric,value\n';
        
        if (analysis.metrics) {
            Object.entries(analysis.metrics).forEach(([key, value]) => {
                csv += `${key},${value}\n`;
            });
        }
        
        return csv;
    }
    
    /**
     * Summarize neural data
     */
    summarizeNeuralData(neuralData) {
        if (!neuralData || !neuralData.spikes) {
            return {
                totalSpikes: 0,
                avgFiringRate: 0,
                peakFiringRate: 0,
                activeNeurons: 0,
                silentNeurons: 0
            };
        }
        
        const numNeurons = neuralData.spikes[0].length;
        const duration = neuralData.spikes.length / 1000; // Convert to seconds
        
        let totalSpikes = 0;
        let neuronSpikeCounts = new Array(numNeurons).fill(0);
        
        // Count spikes per neuron
        neuralData.spikes.forEach(timeStep => {
            timeStep.forEach((spike, neuronIdx) => {
                if (spike) {
                    totalSpikes++;
                    neuronSpikeCounts[neuronIdx]++;
                }
            });
        });
        
        // Calculate statistics
        const firingRates = neuronSpikeCounts.map(count => count / duration);
        const activeNeurons = neuronSpikeCounts.filter(count => count > 0).length;
        
        return {
            totalSpikes,
            avgFiringRate: totalSpikes / (numNeurons * duration),
            peakFiringRate: Math.max(...firingRates),
            activeNeurons,
            silentNeurons: numNeurons - activeNeurons
        };
    }
    
    /**
     * Generate MATLAB metrics code
     */
    generateMatlabMetrics(analysis) {
        if (!analysis || !analysis.metrics) return '';
        
        let code = 'metrics = struct();\n';
        
        Object.entries(analysis.metrics).forEach(([key, value]) => {
            code += `metrics.${key} = ${value};\n`;
        });
        
        return code;
    }
    
    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Export screenshot of current visualization
     */
    async exportScreenshot(canvas, filename) {
        canvas.toBlob((blob) => {
            this.downloadBlob(blob, `${filename}.png`);
        }, 'image/png');
    }
    
    /**
     * Create analysis report
     */
    generateReport(data) {
        const report = `
# SNN Agent Visualization Report

## Experiment Details
- **Experiment ID**: ${data.experiment}
- **Episode**: ${data.episode}
- **Export Date**: ${new Date().toLocaleString()}

## Performance Metrics
${this.formatMetrics(data.analysis.metrics)}

## Summary Statistics
- **Episode Length**: ${data.data.trajectory.x.length} timesteps
- **Total Reward**: ${data.data.rewards.reduce((a, b) => a + b, 0)}
- **Exploration Coverage**: ${this.calculateExplorationCoverage(data.data)}%

## Recommendations
${this.generateRecommendations(data)}
        `;
        
        return report;
    }
    
    formatMetrics(metrics) {
        if (!metrics) return 'No metrics available';
        
        return Object.entries(metrics)
            .map(([key, value]) => `- **${this.formatMetricName(key)}**: ${this.formatMetricValue(value)}`)
            .join('\n');
    }
    
    formatMetricName(name) {
        return name.replace(/([A-Z])/g, ' $1')
                  .replace(/^./, str => str.toUpperCase())
                  .trim();
    }
    
    formatMetricValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(3);
        }
        return value;
    }
    
    calculateExplorationCoverage(data) {
        const positions = new Set();
        
        for (let i = 0; i < data.trajectory.x.length; i++) {
            const key = `${Math.floor(data.trajectory.x[i])},${Math.floor(data.trajectory.y[i])}`;
            positions.add(key);
        }
        
        return (positions.size / 100 * 100).toFixed(1); // Assuming 10x10 grid
    }
    
    generateRecommendations(data) {
        const recommendations = [];
        
        // Check exploration
        const coverage = this.calculateExplorationCoverage(data.data);
        if (coverage < 50) {
            recommendations.push('- Low exploration coverage detected. Consider adjusting exploration parameters.');
        }
        
        // Check reward rate
        const rewardRate = data.analysis.metrics.rewardRate;
        if (rewardRate < 1) {
            recommendations.push('- Low reward rate. The agent may need more training or parameter tuning.');
        }
        
        // Check value variance
        if (data.analysis.metrics.valueVariance > 10) {
            recommendations.push('- High value function variance. Consider adjusting learning rate or using value function regularization.');
        }
        
        return recommendations.length > 0 ? recommendations.join('\n') : 'Performance appears optimal.';
    }
}