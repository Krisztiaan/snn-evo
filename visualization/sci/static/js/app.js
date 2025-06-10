// keywords: [visualization, javascript, plotly, cytoscape, interactive, dashboard]

// Global state
let currentEpisodeData = null;
let playbackInterval = null;
let currentTimestep = 0;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('SNN Analytics loaded');
    
    // Check which page we are on by seeing what data is available
    if (typeof experimentsData !== 'undefined') {
        initDashboard();
    }
    if (typeof runData !== 'undefined') {
        initRunDetail();
    }
});

// =============================================================================
// Dashboard Functions
// =============================================================================

function initDashboard() {
    console.log('Initializing dashboard view...');
    
    // Initialize parallel coordinates plot
    if (plotData && plotData.parallel_coords) {
        const trace = {
            type: 'parcoords',
            line: plotData.parallel_coords.line,
            dimensions: plotData.parallel_coords.dimensions.map(dim => ({
                label: dim.label,
                values: dim.values,
                range: dim.range
            }))
        };
        
        const layout = {
            title: 'Parameter-Performance Relationships',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { size: 12 }
        };
        
        Plotly.newPlot('parallel-plot', [trace], layout, {responsive: true});
    }
}

function initScatterPlot() {
    // Create scatter plot matrix for selected metrics
    const metrics = ['avg_reward', 'n_neurons', 'learning_progress'];
    const data = [];
    
    // Extract values for each metric
    const values = {};
    metrics.forEach(metric => {
        values[metric] = experimentsData.map(exp => {
            if (metric === 'n_neurons') {
                return exp.config.neural.n_neurons;
            } else {
                return exp.metrics[metric];
            }
        });
    });
    
    // Create scatter plots for each pair
    for (let i = 0; i < metrics.length; i++) {
        for (let j = 0; j < metrics.length; j++) {
            if (i !== j) {
                data.push({
                    x: values[metrics[j]],
                    y: values[metrics[i]],
                    mode: 'markers',
                    marker: {
                        size: 8,
                        color: values['avg_reward'],
                        colorscale: 'Viridis',
                        showscale: i === 0 && j === 1
                    },
                    text: experimentsData.map(exp => exp.name),
                    hovertemplate: '%{text}<br>X: %{x}<br>Y: %{y}<extra></extra>',
                    xaxis: 'x' + (j + 1),
                    yaxis: 'y' + (i + 1),
                    showlegend: false
                });
            }
        }
    }
    
    const layout = {
        title: 'Metrics Correlation Matrix',
        grid: {
            rows: metrics.length,
            columns: metrics.length,
            pattern: 'independent'
        },
        height: 600
    };
    
    // Add axis labels
    metrics.forEach((metric, i) => {
        layout['xaxis' + (i + 1)] = { title: metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) };
        layout['yaxis' + (i + 1)] = { title: metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) };
    });
    
    Plotly.newPlot('scatter-plot', data, layout, {responsive: true});
}

function initBoxPlot() {
    if (plotData && plotData.box_plot) {
        const layout = {
            title: 'Reward Distribution Across Experiments',
            yaxis: { title: 'Total Reward' },
            xaxis: { title: 'Experiment' },
            showlegend: false
        };
        
        Plotly.newPlot('boxplot-plot', plotData.box_plot, layout, {responsive: true});
    }
}

// =============================================================================
// Run Detail Functions
// =============================================================================

function initRunDetail() {
    console.log('Initializing run detail view for:', runData.name);
    // Tab initialization is handled by openTab function
}

function initProgressionPlots() {
    if (!runData.summary.episode_stats || runData.summary.episode_stats.length === 0) {
        console.warn('No episode stats available');
        return;
    }
    
    const episodeStats = runData.summary.episode_stats;
    const episodes = episodeStats.map(s => s.episode_id);
    
    // Reward progression
    const rewards = episodeStats.map(s => s.total_reward || 0);
    const rewardTrace = {
        x: episodes,
        y: rewards,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Reward',
        line: { color: '#3b82f6' }
    };
    
    // Add moving average
    const windowSize = Math.max(5, Math.floor(rewards.length / 20));
    const movingAvg = rewards.map((val, idx) => {
        const start = Math.max(0, idx - windowSize + 1);
        const window = rewards.slice(start, idx + 1);
        return window.reduce((a, b) => a + b, 0) / window.length;
    });
    
    const avgTrace = {
        x: episodes,
        y: movingAvg,
        type: 'scatter',
        mode: 'lines',
        name: `MA(${windowSize})`,
        line: { color: '#ef4444', dash: 'dash' }
    };
    
    Plotly.newPlot('reward-progression-plot', [rewardTrace, avgTrace], {
        title: 'Reward per Episode',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Total Reward' }
    }, {responsive: true});
    
    // Steps progression
    const steps = episodeStats.map(s => s.steps || 0);
    Plotly.newPlot('steps-progression-plot', [{
        x: episodes,
        y: steps,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#10b981' }
    }], {
        title: 'Episode Length',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Steps' }
    }, {responsive: true});
    
    // Action entropy (if available)
    if (episodeStats[0].action_entropy !== undefined) {
        const entropy = episodeStats.map(s => s.action_entropy || 0);
        Plotly.newPlot('entropy-progression-plot', [{
            x: episodes,
            y: entropy,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#f59e0b' }
        }], {
            title: 'Action Entropy',
            xaxis: { title: 'Episode' },
            yaxis: { title: 'Entropy' }
        }, {responsive: true});
    } else {
        document.getElementById('entropy-progression-plot').style.display = 'none';
    }
    
    // Neural activity (if available)
    if (episodeStats[0].mean_activity !== undefined) {
        const activity = episodeStats.map(s => s.mean_activity || 0);
        Plotly.newPlot('activity-progression-plot', [{
            x: episodes,
            y: activity,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#8b5cf6' }
        }], {
            title: 'Mean Neural Activity',
            xaxis: { title: 'Episode' },
            yaxis: { title: 'Mean Activity' }
        }, {responsive: true});
    } else {
        document.getElementById('activity-progression-plot').style.display = 'none';
    }
}

function initPlayback() {
    // Load first episode
    const firstEpisode = Object.keys(runData.episodes)[0];
    if (firstEpisode) {
        loadEpisode(firstEpisode);
    }
}

function loadEpisode(episodeId) {
    console.log('Loading episode:', episodeId);
    currentEpisodeData = runData.episodes[episodeId];
    
    if (!currentEpisodeData) {
        console.error('Episode data not found:', episodeId);
        return;
    }
    
    // Reset playback
    stopPlayback();
    currentTimestep = 0;
    
    // Update slider
    const slider = document.getElementById('timestep-slider');
    slider.max = currentEpisodeData.timesteps.length - 1;
    slider.value = 0;
    
    // Initialize visualizations
    initRewardTimeline();
    updatePlayback(0);
}

function initRewardTimeline() {
    if (!currentEpisodeData) return;
    
    // Create timeline showing when rewards were collected
    const rewardTimesteps = [];
    const rewardValues = [];
    
    currentEpisodeData.rewards.forEach((reward, idx) => {
        if (reward > 0) {
            rewardTimesteps.push(idx);
            rewardValues.push(reward);
        }
    });
    
    const trace = {
        x: currentEpisodeData.timesteps,
        y: currentEpisodeData.rewards,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Rewards',
        line: { color: '#10b981', width: 2 },
        marker: { size: 8 }
    };
    
    const layout = {
        title: 'Reward Timeline',
        xaxis: { title: 'Timestep' },
        yaxis: { title: 'Reward' },
        height: 250
    };
    
    Plotly.newPlot('reward-timeline', [trace], layout, {responsive: true});
}

function updatePlayback(timestep) {
    if (!currentEpisodeData) return;
    
    timestep = parseInt(timestep);
    currentTimestep = timestep;
    
    // Update timestep display
    document.getElementById('timestep-display').textContent = 
        `Step: ${timestep} / ${currentEpisodeData.timesteps.length - 1}`;
    
    // Update action and state info
    document.getElementById('current-action').textContent = 
        currentEpisodeData.actions[timestep] || '-';
    document.getElementById('current-gradient').textContent = 
        currentEpisodeData.gradients[timestep] || '0';
    document.getElementById('current-reward').textContent = 
        currentEpisodeData.rewards[timestep] || '0';
    
    // Update neural heatmap (if available)
    if (currentEpisodeData.neural_states && currentEpisodeData.neural_states.length > 0) {
        const sampling = currentEpisodeData.neural_states_sampling || 1;
        const nearestIdx = Math.floor(timestep / sampling);
        
        if (nearestIdx < currentEpisodeData.neural_states.length) {
            const neuralData = currentEpisodeData.neural_states[nearestIdx];
            updateNeuralHeatmap(neuralData);
        }
    }
    
    // Update action distribution
    updateActionDistribution(timestep);
    
    // Update reward timeline marker
    updateTimelineMarker(timestep);
}

function updateNeuralHeatmap(neuralData) {
    // Reshape neural data for heatmap
    const n_neurons = neuralData.length;
    const rows = Math.ceil(Math.sqrt(n_neurons));
    const cols = Math.ceil(n_neurons / rows);
    
    // Create 2D array
    const z = [];
    for (let i = 0; i < rows; i++) {
        const row = [];
        for (let j = 0; j < cols; j++) {
            const idx = i * cols + j;
            row.push(idx < n_neurons ? neuralData[idx] : 0);
        }
        z.push(row);
    }
    
    const data = [{
        z: z,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true
    }];
    
    const layout = {
        title: 'Neural Activity',
        xaxis: { showticklabels: false },
        yaxis: { showticklabels: false },
        height: 250
    };
    
    Plotly.newPlot('neural-heatmap', data, layout, {responsive: true});
}

function updateActionDistribution(timestep) {
    // Count actions up to current timestep
    const actionCounts = {};
    for (let i = 0; i <= timestep; i++) {
        const action = currentEpisodeData.actions[i];
        actionCounts[action] = (actionCounts[action] || 0) + 1;
    }
    
    const actions = Object.keys(actionCounts).sort();
    const counts = actions.map(a => actionCounts[a]);
    
    const data = [{
        x: actions,
        y: counts,
        type: 'bar',
        marker: { color: '#3b82f6' }
    }];
    
    const layout = {
        title: 'Action Distribution',
        xaxis: { title: 'Action' },
        yaxis: { title: 'Count' },
        height: 250
    };
    
    Plotly.newPlot('action-distribution', data, layout, {responsive: true});
}

function updateTimelineMarker(timestep) {
    // Add vertical line at current timestep
    const update = {
        shapes: [{
            type: 'line',
            x0: timestep,
            x1: timestep,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: '#ef4444',
                width: 2,
                dash: 'dot'
            }
        }]
    };
    
    Plotly.relayout('reward-timeline', update);
}

function togglePlayback() {
    if (playbackInterval) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    const btn = document.getElementById('play-pause-btn');
    btn.textContent = '⏸ Pause';
    
    const speed = parseInt(document.getElementById('speed-control').value);
    const delay = 1000 / speed;
    
    playbackInterval = setInterval(() => {
        if (currentTimestep >= currentEpisodeData.timesteps.length - 1) {
            stopPlayback();
            return;
        }
        
        currentTimestep++;
        document.getElementById('timestep-slider').value = currentTimestep;
        updatePlayback(currentTimestep);
    }, delay);
}

function stopPlayback() {
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
    }
    
    const btn = document.getElementById('play-pause-btn');
    btn.textContent = '▶ Play';
}

function initNetworkGraph() {
    if (!runData.network || !runData.network.neurons) {
        console.warn('No network data available');
        return;
    }
    
    const elements = [];
    const neurons = runData.network.neurons;
    const connections = runData.network.connections;
    
    // Add neurons as nodes
    neurons.neuron_ids.forEach((id, i) => {
        const neuronType = neurons.neuron_types[i];
        const isExcitatory = neurons.is_excitatory[i];
        
        elements.push({
            data: { 
                id: `n${id}`,
                label: `${id}`,
                type: neuronType,
                excitatory: isExcitatory
            },
            classes: `${neuronType} ${isExcitatory ? 'excitatory' : 'inhibitory'}`
        });
    });
    
    // Add connections as edges (sample for performance)
    const maxEdges = 1000;
    const edgeStep = Math.ceil(connections.source_ids.length / maxEdges);
    
    for (let i = 0; i < connections.source_ids.length; i += edgeStep) {
        elements.push({
            data: {
                id: `e${i}`,
                source: `n${connections.source_ids[i]}`,
                target: `n${connections.target_ids[i]}`
            }
        });
    }
    
    // Initialize Cytoscape
    const cy = cytoscape({
        container: document.getElementById('network-graph'),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'width': 20,
                    'height': 20,
                    'font-size': 8,
                    'text-valign': 'center',
                    'text-halign': 'center'
                }
            },
            {
                selector: '.sensory',
                style: {
                    'background-color': '#10b981',
                    'shape': 'diamond'
                }
            },
            {
                selector: '.processing',
                style: {
                    'background-color': '#3b82f6',
                    'shape': 'ellipse'
                }
            },
            {
                selector: '.readout',
                style: {
                    'background-color': '#ef4444',
                    'shape': 'rectangle'
                }
            },
            {
                selector: '.inhibitory',
                style: {
                    'border-width': 3,
                    'border-color': '#6b7280'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 0.5,
                    'line-color': '#e5e7eb',
                    'target-arrow-color': '#e5e7eb',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0.5
                }
            }
        ],
        layout: {
            name: 'cose',
            nodeRepulsion: 400000,
            idealEdgeLength: 100,
            nodeOverlap: 20,
            animate: true,
            animationDuration: 500
        }
    });
    
    // Store reference for controls
    window.networkCy = cy;
}

function resetNetworkView() {
    if (window.networkCy) {
        window.networkCy.fit();
        window.networkCy.center();
    }
}

function toggleConnections() {
    if (window.networkCy) {
        const showConnections = document.getElementById('show-connections').checked;
        window.networkCy.edges().style('display', showConnections ? 'element' : 'none');
    }
}

// Tab switching for plots
function showPlot(plotName) {
    // Hide all plots
    document.querySelectorAll('.plot-container').forEach(p => {
        p.classList.remove('active-plot');
    });
    document.querySelectorAll('.tab-button').forEach(b => {
        b.classList.remove('active');
    });
    
    // Show selected plot
    document.getElementById(plotName + '-plot').classList.add('active-plot');
    event.target.classList.add('active');
    
    // Initialize plot if needed
    if (plotName === 'scatter' && !window.scatterInitialized) {
        initScatterPlot();
        window.scatterInitialized = true;
    } else if (plotName === 'boxplot' && !window.boxplotInitialized) {
        initBoxPlot();
        window.boxplotInitialized = true;
    }
}