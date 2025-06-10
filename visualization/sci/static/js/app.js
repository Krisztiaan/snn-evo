// keywords: [plotly visualization, cytoscape network, dashboard analytics, episode playback]

document.addEventListener('DOMContentLoaded', () => {
    // --- Dashboard Page Logic ---
    if (document.getElementById('experiments-table')) {
        initDashboard();
    }

    // --- Run Detail Page Logic ---
    if (document.getElementById('run-detail-container')) {
        initRunDetail();
    }
});


// =================================================================================
// DASHBOARD PAGE
// =================================================================================

function initDashboard() {
    // Initialize DataTable
    if (typeof DataTable !== 'undefined') {
        new DataTable('#experiments-table');
    }
    
    // Attach tab event listeners
    document.querySelectorAll('.analysis-container .tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            const tabName = e.target.getAttribute('data-tab');
            
            // Hide all tabs
            document.querySelectorAll('.analysis-container .tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.analysis-container .tab-button').forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            e.target.classList.add('active');

            // Initialize plot on first view
            if (tabName === 'parallel-coordinates' && !window.parallelInitialized) {
                initParallelCoords();
                window.parallelInitialized = true;
            } else if (tabName === 'scatter-matrix' && !window.scatterInitialized) {
                initScatterMatrix();
                window.scatterInitialized = true;
            } else if (tabName === 'reward-distribution' && !window.rewardDistInitialized) {
                initRewardDistribution();
                window.rewardDistInitialized = true;
            }
        });
    });

    // Trigger click on the first tab to initialize it
    const firstTab = document.querySelector('.analysis-container .tab-button.active');
    if (firstTab) firstTab.click();
}

function initParallelCoords() {
    // Fix: Use correct element ID and check for plotsData (not plotData)
    const container = document.getElementById('parallel-coord-plot');
    if (!container) return;
    
    if (!plotsData || !plotsData.parallel_coords || !plotsData.parallel_coords.dimensions || !plotsData.parallel_coords.dimensions.length) {
        container.innerHTML = '<p class="plot-placeholder">No data available for parallel coordinates plot.</p>';
        return;
    }
    const trace = {
        type: 'parcoords',
        line: plotsData.parallel_coords.line,
        dimensions: plotsData.parallel_coords.dimensions
    };
    const layout = {
        title: 'Experiment Parameter & Result Comparison',
        margin: {t: 50, l: 50, r: 50, b: 50}
    };
    Plotly.newPlot(container, [trace], layout);
}

function initScatterMatrix() {
    const container = document.getElementById('scatter-plot');
    if (!container) return;
    
    if (!plotsData || !plotsData.parallel_coords || !plotsData.parallel_coords.dimensions || plotsData.parallel_coords.dimensions.length < 2) {
        container.innerHTML = '<p class="plot-placeholder">Not enough data for scatter matrix plot.</p>';
        return;
    }
    
    // For now, create a simple scatter plot of first two dimensions
    const dim1 = plotsData.parallel_coords.dimensions[0];
    const dim2 = plotsData.parallel_coords.dimensions[1];
    
    const trace = {
        x: dim1.values,
        y: dim2.values,
        mode: 'markers',
        type: 'scatter',
        text: plotsData.parallel_coords.dimensions.map((d, i) => `Exp ${i}`),
        marker: {
            color: plotsData.parallel_coords.line.color,
            colorscale: plotsData.parallel_coords.line.colorscale,
            showscale: true
        }
    };
    const layout = {
        title: `${dim1.label} vs ${dim2.label}`,
        xaxis: { title: dim1.label },
        yaxis: { title: dim2.label },
        margin: {t: 50, l: 60, r: 50, b: 60}
    };
    Plotly.newPlot(container, [trace], layout);
}

function initRewardDistribution() {
    const container = document.getElementById('reward-dist-plot');
    if (!container) return;
    
    if (!plotsData || !plotsData.box_plot || !plotsData.box_plot.length) {
        container.innerHTML = '<p class="plot-placeholder">No reward distribution data available.</p>';
        return;
    }
    const layout = {
        title: 'Reward Distribution Across Experiments',
        yaxis: {
            title: 'Total Reward'
        },
        xaxis: {
            tickangle: -45,
            automargin: true
        },
        margin: {t: 50, l: 70, r: 50, b: 100}
    };
    Plotly.newPlot(container, plotsData.box_plot, layout);
}


// =================================================================================
// RUN DETAIL PAGE
// =================================================================================

function initRunDetail() {
    // Create playback state local to this function
    const playbackState = {
        isPlaying: false,
        intervalId: null,
        currentTimestep: 0,
        currentEpisode: null,
        speed: 50 // ms per frame
    };
    
    // Store playback state globally for debugging
    window.playbackState = playbackState;
    
    // Attach event listeners for tab buttons (remove onclick from HTML)
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            // Get tab name from onclick attribute for backwards compatibility
            const onclickAttr = e.target.getAttribute('onclick');
            if (onclickAttr) {
                const match = onclickAttr.match(/openTab\(event,\s*'([^']+)'\)/);
                if (match) {
                    openTab(e, match[1]);
                }
            }
        });
    });
    
    // Set initial tab content
    openTab({ currentTarget: document.querySelector('.tab-button.active') }, 'summary');

    // Define all functions inside initRunDetail scope
    function initProgressionPlots() {
    const stats = runData.summary.episode_stats || [];
    if (!stats.length) {
        document.getElementById('progression').innerHTML = '<p class="plot-placeholder">No episode statistics available for progression plots.</p>';
        return;
    }

    const episodeIds = stats.map(s => s.episode_id);
    
    // Reward Progression
    Plotly.newPlot('reward-progression-plot', [{
        x: episodeIds,
        y: stats.map(s => s.total_reward),
        mode: 'lines+markers',
        name: 'Total Reward'
    }], { 
        title: 'Reward per Episode',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Total Reward' },
        margin: {t: 40, l: 60, r: 30, b: 40}
    });

    // Steps Progression
    Plotly.newPlot('steps-progression-plot', [{
        x: episodeIds,
        y: stats.map(s => s.steps || s.timesteps),
        mode: 'lines+markers',
        name: 'Steps'
    }], { 
        title: 'Steps per Episode',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Steps' },
        margin: {t: 40, l: 60, r: 30, b: 40}
    });

    // Entropy Progression
    Plotly.newPlot('entropy-progression-plot', [{
        x: episodeIds,
        y: stats.map(s => s.action_entropy),
        mode: 'lines+markers',
        name: 'Action Entropy'
    }], { 
        title: 'Action Entropy per Episode',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Entropy' },
        margin: {t: 40, l: 60, r: 30, b: 40}
    });

    // Activity Progression
    Plotly.newPlot('activity-progression-plot', [{
        x: episodeIds,
        y: stats.map(s => s.mean_neural_activity),
        mode: 'lines+markers',
        name: 'Mean Neural Activity'
    }], { 
        title: 'Mean Neural Activity per Episode',
        xaxis: { title: 'Episode' },
        yaxis: { title: 'Activity' },
        margin: {t: 40, l: 60, r: 30, b: 40}
    });
    }

    function initNetworkGraph() {
    if (!runData.network || !runData.network.neurons.neuron_ids) {
        document.getElementById('network').innerHTML = '<p class="plot-placeholder">No network structure data available.</p>';
        return;
    }

    const elements = [];
    // Neurons
    for (let i = 0; i < runData.network.neurons.neuron_ids.length; i++) {
        const id = runData.network.neurons.neuron_ids[i];
        const type = runData.network.neurons.neuron_types[i];
        let color, size;
        switch (type) {
            case 0: color = '#63b3ed'; size = 20; break; // Sensory
            case 1: color = '#f6e05e'; size = 15; break; // Processing
            case 2: color = '#f56565'; size = 20; break; // Motor
            default: color = '#a0aec0'; size = 15;
        }
        elements.push({
            group: 'nodes',
            data: { 
                id: `n${id}`, 
                label: `${id}`,
                type: type
            },
            style: {
                'background-color': color,
                'width': size,
                'height': size,
                'label': 'data(label)',
                'font-size': '8px',
                'text-valign': 'center',
                'text-halign': 'center'
            }
        });
    }
    // Connections
    for (let i = 0; i < runData.network.connections.source_ids.length; i++) {
        elements.push({
            group: 'edges',
            data: {
                id: `e${i}`,
                source: `n${runData.network.connections.source_ids[i]}`,
                target: `n${runData.network.connections.target_ids[i]}`
            }
        });
    }

    // Initialize Cytoscape without layout (will be run when tab is shown)
    window.cy = cytoscape({
        container: document.getElementById('network-graph'),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'font-size': '8px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': '#fff',
                    'text-outline-width': 2,
                    'text-outline-color': '#888'
                }
            },
            {
                selector: 'node[type=0]',
                style: { 'background-color': '#63b3ed', 'width': 20, 'height': 20 }
            },
            {
                selector: 'node[type=1]',
                style: { 'background-color': '#f6e05e', 'width': 15, 'height': 15 }
            },
            {
                selector: 'node[type=2]',
                style: { 'background-color': '#f56565', 'width': 20, 'height': 20 }
            },
            {
                selector: 'edge',
                style: {
                    'width': 1,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0.5
                }
            }
        ]
    });
    }

    function initPlayback() {
        // Event listeners
        document.getElementById('episode-select').addEventListener('change', (e) => loadEpisode(e.target.value));
        document.getElementById('timestep-slider').addEventListener('input', (e) => updatePlayback(parseInt(e.target.value)));
        document.getElementById('play-pause-btn').addEventListener('click', togglePlayback);
        document.getElementById('speed-control').addEventListener('input', (e) => {
            playbackState.speed = 200 / parseInt(e.target.value);
        });

        // Load first episode
        const firstEpisodeId = document.getElementById('episode-select').value;
        if (firstEpisodeId) {
            loadEpisode(firstEpisodeId);
        }
        }

    function loadEpisode(episodeId) {
    playbackState.currentEpisode = runData.episodes[episodeId];
    if (!playbackState.currentEpisode) {
        console.error('Episode not found:', episodeId);
        return;
    }

    stopPlayback();
    playbackState.currentTimestep = 0;
    
    const slider = document.getElementById('timestep-slider');
    slider.max = playbackState.currentEpisode.timesteps.length - 1;
    slider.value = 0;
    
    // Initial plot renders
    renderRewardTimeline();
    renderActionDistribution();
    updatePlayback(0);
    }

    function updatePlayback(timestep) {
    playbackState.currentTimestep = parseInt(timestep);
    const slider = document.getElementById('timestep-slider');
    if (slider.value != timestep) slider.value = timestep;
    
    const episodeData = playbackState.currentEpisode;
    if (!episodeData) return;

    const totalSteps = episodeData.timesteps.length;
    document.getElementById('timestep-display').textContent = `Step: ${timestep} / ${totalSteps - 1}`;

    // Update Action & State panel
    document.getElementById('current-action').textContent = episodeData.actions[timestep] || 0;
    document.getElementById('current-gradient').textContent = (episodeData.gradients && episodeData.gradients[timestep] || 0).toFixed(3);
    document.getElementById('current-reward').textContent = (episodeData.rewards[timestep] || 0).toFixed(2);
    
    // Update plots
    updateRewardTimelineMarker();
    renderNeuralHeatmap();
    }

    function renderRewardTimeline() {
    const episodeData = playbackState.currentEpisode;
    if (!episodeData) return;
    
    const trace = {
        x: episodeData.timesteps,
        y: episodeData.rewards,
        mode: 'lines',
        name: 'Reward',
        line: { color: 'blue' }
    };
    const layout = {
        title: 'Reward Timeline',
        xaxis: { title: 'Timestep' },
        yaxis: { title: 'Reward' },
        shapes: [{
            type: 'line',
            x0: 0,
            x1: 0,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: 'red', width: 2 }
        }],
        margin: { l: 50, r: 30, t: 50, b: 40 }
    };
    Plotly.newPlot('reward-timeline', [trace], layout, {staticPlot: false});
    }

    function updateRewardTimelineMarker() {
    const update = {
        'shapes[0].x0': playbackState.currentTimestep,
        'shapes[0].x1': playbackState.currentTimestep
    };
    Plotly.relayout('reward-timeline', update);
    }

    function renderNeuralHeatmap() {
    const episodeData = playbackState.currentEpisode;
    if (!episodeData.neural_states || !episodeData.neural_states.length) {
        document.getElementById('neural-heatmap').innerHTML = '<p class="plot-placeholder">No neural state data.</p>';
        return;
    }
    
    const sampling = episodeData.neural_states_sampling || 1;
    const sampledTimestep = Math.floor(playbackState.currentTimestep / sampling);

    if (sampledTimestep >= episodeData.neural_states.length) return;

    const activity = episodeData.neural_states[sampledTimestep];
    const nNeurons = activity.length;
    const side = Math.ceil(Math.sqrt(nNeurons));
    const heatmap = new Array(side).fill(0).map(() => new Array(side).fill(0));
    
    for (let i = 0; i < nNeurons; i++) {
        const row = Math.floor(i / side);
        const col = i % side;
        heatmap[row][col] = activity[i];
    }

    const trace = {
        z: heatmap,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true
    };
    const layout = {
        title: 'Neural Activity Heatmap',
        xaxis: { visible: false },
        yaxis: { visible: false },
        margin: { l: 30, r: 30, t: 50, b: 30 }
    };
    Plotly.newPlot('neural-heatmap', [trace], layout);
    }

    function renderActionDistribution() {
    const episodeData = playbackState.currentEpisode;
    if (!episodeData) return;
    
    const actionLabels = ['Up', 'Down', 'Left', 'Right', 'Stay', 'A5', 'A6', 'A7', 'A8'];
    const counts = new Array(9).fill(0);
    episodeData.actions.forEach(a => { 
        if (a >= 0 && a < 9) counts[a]++; 
    });
    
    const trace = {
        x: actionLabels.slice(0, Math.max(...episodeData.actions) + 1),
        y: counts.slice(0, Math.max(...episodeData.actions) + 1),
        type: 'bar',
        marker: { color: 'lightblue' }
    };
    const layout = {
        title: 'Action Distribution',
        xaxis: { title: 'Action' },
        yaxis: { title: 'Count' },
        margin: { l: 50, r: 30, t: 50, b: 60 }
    };
    Plotly.newPlot('action-distribution', [trace], layout);
    }

    function togglePlayback() {
        if (playbackState.isPlaying) {
            stopPlayback();
        } else {
            startPlayback();
        }
    }

    function startPlayback() {
        playbackState.isPlaying = true;
        document.getElementById('play-pause-btn').textContent = '❚❚ Pause';
        playbackState.intervalId = setInterval(() => {
            let nextStep = playbackState.currentTimestep + 1;
            if (nextStep >= playbackState.currentEpisode.timesteps.length) {
                stopPlayback();
            } else {
                updatePlayback(nextStep);
            }
        }, playbackState.speed);
    }

    function stopPlayback() {
        playbackState.isPlaying = false;
        document.getElementById('play-pause-btn').textContent = '▶ Play';
        if (playbackState.intervalId) {
            clearInterval(playbackState.intervalId);
            playbackState.intervalId = null;
        }
    }

    // Tab functionality for run_detail page
    function openTab(evt, tabName) {
        const tabcontent = document.getElementsByClassName("tab-content");
        for (let i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
            tabcontent[i].classList.remove("active-tab");
        }
        
        const tablinks = document.getElementsByClassName("tab-button");
        for (let i = 0; i < tablinks.length; i++) {
            tablinks[i].classList.remove("active");
        }
        
        const tabElement = document.getElementById(tabName);
        if (tabElement) {
            tabElement.style.display = "block";
            tabElement.classList.add("active-tab");
        }
        evt.currentTarget.classList.add("active");
        
        // Initialize content when tab is first opened
        if (tabName === 'progression' && !window.progressionInitialized) {
            initProgressionPlots();
            window.progressionInitialized = true;
        } else if (tabName === 'playback' && !window.playbackInitialized) {
            initPlayback();
            window.playbackInitialized = true;
        } else if (tabName === 'network' && !window.networkInitialized) {
            initNetworkGraph();
            window.networkInitialized = true;
            // FIX: Run layout after the tab is visible
            setTimeout(() => {
                if (window.cy) {
                    window.cy.layout({
                        name: 'cose',
                        animate: true,
                        animationDuration: 500,
                        nodeRepulsion: 8000,
                        idealEdgeLength: 50,
                        gravity: 0.1,
                        numIter: 1000
                    }).run();
                }
            }, 10);
        }
    }

    // Functions for network view controls
    function resetNetworkView() {
        if (window.cy) {
            window.cy.fit();
        }
    }

    function toggleConnections() {
        if (window.cy) {
            const show = document.getElementById('show-connections').checked;
            window.cy.edges().style('display', show ? 'element' : 'none');
        }
    }
    
    // Attach network control event handlers
    const resetBtn = document.getElementById('reset-network-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetNetworkView);
    }
    
    const showConnCheckbox = document.getElementById('show-connections');
    if (showConnCheckbox) {
        showConnCheckbox.addEventListener('change', toggleConnections);
    }
}

// Tab switching for dashboard plots
window.showPlot = function(plotName) {
    document.querySelectorAll('.plot-container').forEach(p => {
        p.style.display = 'none';
        p.classList.remove('active-plot');
    });
    document.querySelectorAll('.plot-tabs .tab-button').forEach(b => b.classList.remove('active'));
    
    const plotElement = document.getElementById(plotName + '-plot');
    if (plotElement) {
        plotElement.style.display = 'block';
        plotElement.classList.add('active-plot');
    }
    
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    // Initialize plots on first view
    if (plotName === 'parallel' && !window.parallelInitialized) {
        initParallelCoords();
        window.parallelInitialized = true;
    } else if (plotName === 'scatter' && !window.scatterInitialized) {
        initScatterPlot();
        window.scatterInitialized = true;
    } else if (plotName === 'boxplot' && !window.boxplotInitialized) {
        initBoxPlot();
        window.boxplotInitialized = true;
    }
}