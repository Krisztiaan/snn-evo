import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DOM = {
  container: document.getElementById("container"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  modelSelect: document.getElementById("modelSelect"),
  runSelect: document.getElementById("runSelect"),
  episodeSelect: document.getElementById("episodeSelect"),
  loadBtn: document.getElementById("loadBtn"),
  playBtn: document.getElementById("playBtn"),
  resetBtn: document.getElementById("resetBtn"),
  timeline: document.getElementById("timeline"),
  stepDisplay: document.getElementById("stepDisplay"),
  toggleAnalyticsBtn: document.getElementById("toggleAnalytics"),
  analyticsPanel: document.getElementById("analytics"),
  info: {
    step: document.getElementById("info-step"),
    pos: document.getElementById("info-pos"),
    action: document.getElementById("info-action"),
    gradient: document.getElementById("info-gradient"),
    reward: document.getElementById("info-reward"),
    potential: document.getElementById("info-potential"),
    firingRate: document.getElementById("info-firingrate"),
    spikes: document.getElementById("info-spikes"),
    dopamine: document.getElementById("info-dopamine"),
    valueEst: document.getElementById("info-value_est"),
  },
};

const state = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  agent: null,
  trail: null,
  rewards: [],
  grid: null,
  experiments: {},
  currentData: null,
  currentStep: 0,
  isPlaying: false,
  playInterval: null,
  charts: {},
  activeChart: "behavior",
};

const CONFIG = {
  AGENT_COLOR: 0xff6b00,
  REWARD_COLOR: 0x00a8e8,
  REWARD_COLLECTED_COLOR: 0x333333,
  TRAIL_COLOR: 0xff6b00,
  AGENT_SIZE: 0.7,
  REWARD_SIZE: 0.4,
  CHART_PRIMARY: "#FF6B00",
  CHART_SECONDARY: "#00A8E8",
  CHART_TEXT: "#f0f0f0",
  CHART_NEURAL_1: "#E91E63",
  CHART_NEURAL_2: "#4CAF50",
  CHART_LEARNING_1: "#FFC107",
  CHART_LEARNING_2: "#9C27B0",
};

function init() {
  setupScene();
  setupLights();
  setupCharts();
  attachEventListeners();
  animate();
  loadExperimentList();
}

function setupScene() {
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x1a1a1a);

  const aspect = window.innerWidth / window.innerHeight;
  state.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
  state.camera.position.set(10, 20, 30);

  state.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  state.renderer.setSize(window.innerWidth, window.innerHeight);
  DOM.container.appendChild(state.renderer.domElement);

  state.controls = new OrbitControls(state.camera, state.renderer.domElement);
  state.controls.enableDamping = true;
}

function setupLights() {
  state.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(10, 20, 5);
  state.scene.add(dirLight);
}

function setupCharts() {
  ["behavior", "neural", "learning"].forEach((type) => {
    const chartDom = document.getElementById(`${type}Chart`);
    state.charts[type] = echarts.init(chartDom, "dark");
  });
}

function attachEventListeners() {
  window.addEventListener("resize", onWindowResize);
  DOM.modelSelect.addEventListener("change", window.selectModel);
  DOM.runSelect.addEventListener("change", window.selectRun);
  DOM.loadBtn.addEventListener("click", window.loadData);
  DOM.playBtn.addEventListener("click", togglePlay);
  DOM.resetBtn.addEventListener("click", resetPlayback);
  DOM.timeline.addEventListener("input", (e) =>
    setStep(parseInt(e.target.value))
  );
  DOM.analyticsPanel
    .querySelector(".analytics-tabs")
    .addEventListener("click", (e) => {
      if (e.target.tagName === "BUTTON") {
        switchChart(e.target.dataset.chart);
      }
    });
  DOM.toggleAnalyticsBtn.addEventListener("click", toggleAnalytics);
}

async function loadExperimentList() {
  showLoading("Loading compatible experiments...");
  let responseTextForLogging = ""; // To store response text for logging in case of parsing error

  try {
    const response = await fetch("/api/experiments");
    responseTextForLogging = await response.text(); // Assign sooner, in case response.ok itself fails or response.text() fails

    // Log the raw response text immediately for debugging
    console.log("Raw response from /api/experiments:", responseTextForLogging);

    if (!response.ok) {
      console.error(
        `Server error fetching experiments! Status: ${response.status} ${response.statusText}. Response body: ${responseTextForLogging}`
      );
      throw new Error(
        `Server error: ${response.status} ${response.statusText}`
      );
    }

    if (!responseTextForLogging || responseTextForLogging.trim() === "") {
      console.warn(
        "Received empty or whitespace-only response from /api/experiments. Assuming no experiments."
      );
      state.experiments = {};
    } else {
      const data = JSON.parse(responseTextForLogging);
      state.experiments = data.models || {};
    }
  } catch (e) {
    console.error("Error during loadExperimentList. Error object:", e); // Log the whole error object
    let detailedMessage = "Failed to load or parse experiment list. ";
    if (e instanceof Error) {
      detailedMessage += `Type: ${e.name}, Message: ${e.message}`;
    } else {
      detailedMessage += `Unexpected error type: ${typeof e}, Value: ${String(
        e
      )}`;
    }
    console.error(detailedMessage); // More detailed message

    if (e instanceof SyntaxError) {
      console.error(
        "JSON parsing failed. Original response text that was problematic:",
        responseTextForLogging // Already contains the text
      );
    } else if (responseTextForLogging && responseTextForLogging.length > 0) {
      // If it wasn't a SyntaxError but we have responseText, log it as it might be relevant
      console.warn(
        "Response text (which might not have been parsed or caused a non-SyntaxError):",
        responseTextForLogging
      );
    }
    state.experiments = {}; // Ensure experiments are cleared on error
  } finally {
    populateModelSelect(); // Will now handle empty/error state better
    if (typeof hideLoading === "function") {
      hideLoading();
    }
  }
}

window.loadData = async function () {
  const path = DOM.runSelect.value;
  const episodeId = DOM.episodeSelect.value;
  if (!path || !episodeId) return;

  showLoading("Loading episode data...");
  try {
    const response = await fetch(
      `/api/episode-data?path=${encodeURIComponent(path)}&episode=${episodeId}`
    );
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    state.currentData = data;
    setupVisualization();
  } catch (e) {
    showLoading(`Error: ${e.message}`);
  } finally {
    hideLoading();
  }
};

function populateModelSelect() {
  DOM.modelSelect.innerHTML = ""; // Clear previous options

  if (Object.keys(state.experiments).length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No compatible models found or error loading.";
    DOM.modelSelect.appendChild(option);
    // Ensure other dependent dropdowns are hidden and cleared
    DOM.runSelect.innerHTML = "";
    DOM.runSelect.style.display = "none";
    DOM.episodeSelect.innerHTML = "";
    DOM.episodeSelect.style.display = "none";
    DOM.loadBtn.style.display = "none";
    return;
  }

  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "Select a Model";
  DOM.modelSelect.appendChild(defaultOption);

  Object.keys(state.experiments)
    .sort()
    .forEach((modelName) => {
      const option = document.createElement("option");
      option.value = modelName;
      option.textContent = `${modelName} (${state.experiments[modelName].length} runs)`;
      DOM.modelSelect.appendChild(option);
    });

  // Reset and hide dependent dropdowns
  DOM.runSelect.innerHTML = "";
  DOM.runSelect.style.display = "none";
  DOM.episodeSelect.innerHTML = "";
  DOM.episodeSelect.style.display = "none";
  DOM.loadBtn.style.display = "none";
}

window.selectModel = function () {
  const modelName = DOM.modelSelect.value;
  DOM.runSelect.style.display = modelName ? "block" : "none";
  DOM.episodeSelect.style.display = "none";
  DOM.loadBtn.style.display = "none";
  if (!modelName) return;

  DOM.runSelect.innerHTML = '<option value="">Select a Run</option>';
  state.experiments[modelName].forEach((run) => {
    const option = document.createElement("option");
    option.value = run.path;
    option.textContent = run.label;
    DOM.runSelect.appendChild(option);
  });
};

window.selectRun = function () {
  const path = DOM.runSelect.value;
  DOM.episodeSelect.style.display = path ? "block" : "none";
  DOM.loadBtn.style.display = "none";
  if (!path) return;

  const modelName = DOM.modelSelect.value;
  const run = state.experiments[modelName].find((r) => r.path === path);
  DOM.episodeSelect.innerHTML = "";
  run.episodes.forEach((ep) => {
    const option = document.createElement("option");
    option.value = ep;
    option.textContent = `Episode ${ep}`;
    DOM.episodeSelect.appendChild(option);
  });
  DOM.loadBtn.style.display = "block";
};

function setupVisualization() {
  clearScene();
  const { metadata, worldSetup, timeline } = state.currentData;
  const gridSize = metadata.world.gridSize;

  state.grid = createGrid(gridSize);
  state.scene.add(state.grid);
  state.rewards = createRewards(worldSetup.rewardPositions);
  state.agent = createAgent();
  state.trail = createTrail();

  state.camera.position.set(gridSize / 2, gridSize, gridSize * 1.5);
  state.controls.target.set(gridSize / 2, 0, gridSize / 2);

  DOM.timeline.max = timeline.length - 1;
  setStep(0);

  prepareChartData();
  renderAllCharts();
}

function clearScene() {
  if (state.grid) state.scene.remove(state.grid);
  if (state.agent) state.scene.remove(state.agent);
  if (state.trail) state.scene.remove(state.trail);
  state.rewards.forEach((r) => state.scene.remove(r.mesh));
  state.rewards = [];
}

function createGrid(size) {
  const group = new THREE.Group();
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(size, size),
    new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.8 })
  );
  floor.rotation.x = -Math.PI / 2;
  group.add(floor);
  group.add(new THREE.GridHelper(size, size, 0x444444, 0x444444));
  group.position.set(size / 2, 0, size / 2);
  return group;
}

function createRewards(positions) {
  return positions.map((pos) => {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(CONFIG.REWARD_SIZE, 16, 8),
      new THREE.MeshStandardMaterial({
        color: CONFIG.REWARD_COLOR,
        emissive: CONFIG.REWARD_COLOR,
        emissiveIntensity: 0.5,
      })
    );
    mesh.position.set(pos[0] + 0.5, CONFIG.REWARD_SIZE, pos[1] + 0.5);
    state.scene.add(mesh);
    return { mesh, collected: false };
  });
}

function createAgent() {
  const agent = new THREE.Mesh(
    new THREE.ConeGeometry(CONFIG.AGENT_SIZE / 2, CONFIG.AGENT_SIZE, 16),
    new THREE.MeshStandardMaterial({
      color: CONFIG.AGENT_COLOR,
      emissive: CONFIG.AGENT_COLOR,
      emissiveIntensity: 0.5,
    })
  );
  agent.position.y = CONFIG.AGENT_SIZE / 2;
  state.scene.add(agent);
  return agent;
}

function createTrail() {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.LineBasicMaterial({
    color: CONFIG.TRAIL_COLOR,
    linewidth: 2,
  });
  const trail = new THREE.Line(geometry, material);
  state.scene.add(trail);
  return trail;
}

function setStep(step) {
  if (!state.currentData) return;
  state.currentStep = Math.max(
    0,
    Math.min(step, state.currentData.timeline.length - 1)
  );
  DOM.timeline.value = state.currentStep;
  updateScene();
  updateUI();
  updateCharts();
}

function togglePlay() {
  state.isPlaying = !state.isPlaying;
  DOM.playBtn.textContent = state.isPlaying ? "❚❚ Pause" : "▶ Play";
  if (state.isPlaying) {
    state.playInterval = setInterval(() => {
      if (state.currentStep < state.currentData.timeline.length - 1) {
        setStep(state.currentStep + 1);
      } else {
        togglePlay();
      }
    }, 50);
  } else {
    clearInterval(state.playInterval);
  }
}

function resetPlayback() {
  if (state.isPlaying) togglePlay();
  setStep(0);
}

function updateScene() {
  const { timeline } = state.currentData;
  const stepData = timeline[state.currentStep];

  const pos = stepData.position;
  state.agent.position.set(pos[0] + 0.5, CONFIG.AGENT_SIZE / 2, pos[1] + 0.5);

  if (state.currentStep > 0) {
    const prevPos = timeline[state.currentStep - 1].position;
    const dx = pos[0] - prevPos[0];
    const dz = pos[1] - prevPos[1];
    if (Math.abs(dx) > 0.01 || Math.abs(dz) > 0.01) {
      state.agent.rotation.y = Math.atan2(dx, dz);
    }
  }

  const trailPositions = timeline
    .slice(0, state.currentStep + 1)
    .map(
      (s) => new THREE.Vector3(s.position[0] + 0.5, 0.1, s.position[1] + 0.5)
    );
  state.trail.geometry.setFromPoints(trailPositions);

  const collectedAtThisStep = {};
  if (stepData.reward > 0.5) {
    state.currentData.worldSetup.rewardPositions.forEach((rPos, i) => {
      if (Math.hypot(pos[0] - rPos[0], pos[1] - rPos[1]) < 1) {
        collectedAtThisStep[i] = true;
      }
    });
  }

  state.rewards.forEach((r, i) => {
    if (state.rewards[i].collected || collectedAtThisStep[i]) {
      r.collected = true;
    }
    r.mesh.material.color.set(
      r.collected ? CONFIG.REWARD_COLLECTED_COLOR : CONFIG.REWARD_COLOR
    );
    r.mesh.material.emissive.set(r.collected ? 0x000000 : CONFIG.REWARD_COLOR);
  });
}

function updateUI() {
  const stepData = state.currentData.timeline[state.currentStep];
  DOM.stepDisplay.textContent = `Step: ${stepData.step} / ${
    state.currentData.metadata.totalSteps - 1
  }`;
  DOM.info.step.textContent = stepData.step;
  DOM.info.pos.textContent = `${stepData.position[0].toFixed(
    1
  )}, ${stepData.position[1].toFixed(1)}`;
  DOM.info.action.textContent =
    ["Up", "Right", "Down", "Left"][stepData.action] || "N/A";
  DOM.info.gradient.textContent = stepData.gradient.toFixed(3);
  DOM.info.reward.textContent = stepData.reward.toFixed(2);

  const neural = stepData.neural;
  DOM.info.potential.textContent = neural
    ? neural.mean_potential.toFixed(2)
    : "N/A";
  DOM.info.firingRate.textContent = neural
    ? neural.firing_rate_hz.toFixed(2)
    : "N/A";
  DOM.info.spikes.textContent = neural ? neural.spike_count : "N/A";
  DOM.info.dopamine.textContent = neural?.dopamine?.toFixed(3) ?? "N/A";
  DOM.info.valueEst.textContent = neural?.value_estimate?.toFixed(3) ?? "N/A";
}

function onWindowResize() {
  state.camera.aspect = window.innerWidth / window.innerHeight;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(window.innerWidth, window.innerHeight);
  Object.values(state.charts).forEach((chart) => chart.resize());
}

function animate() {
  requestAnimationFrame(animate);
  state.controls.update();
  state.renderer.render(state.scene, state.camera);
}

function toggleAnalytics() {
  const isHidden = DOM.analyticsPanel.style.display === "none";
  DOM.analyticsPanel.style.display = isHidden ? "block" : "none";
  DOM.toggleAnalyticsBtn.textContent = isHidden
    ? "📊 Hide Analytics"
    : "📊 Show Analytics";
  if (isHidden) {
    Object.values(state.charts).forEach((chart) => chart.resize());
  }
}

function switchChart(type) {
  state.activeChart = type;
  document
    .querySelectorAll(".analytics-tab")
    .forEach((tab) => tab.classList.remove("active"));
  document.querySelector(`[data-chart="${type}"]`).classList.add("active");
  document
    .querySelectorAll(".chart-container")
    .forEach((c) => (c.style.display = "none"));
  document.getElementById(`${type}Chart`).style.display = "block";
}

function prepareChartData() {
  const { timeline } = state.currentData;
  const steps = timeline.map((s) => s.step);

  const cumulativeRewards = timeline.reduce((acc, s) => {
    acc.push((acc.length > 0 ? acc[acc.length - 1] : 0) + s.reward);
    return acc;
  }, []);

  const chartOptions = (title, legend, yAxis, series) => ({
    title: { text: title, textStyle: { color: CONFIG.CHART_TEXT } },
    tooltip: { trigger: "axis" },
    legend: { data: legend, textStyle: { color: CONFIG.CHART_TEXT } },
    grid: { left: "10%", right: "10%", bottom: "15%" },
    xAxis: {
      type: "category",
      data: steps,
      axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } },
    },
    yAxis: yAxis,
    series: series,
    dataZoom: [{ type: "inside" }, { type: "slider" }],
    backgroundColor: "transparent",
  });

  state.charts.behavior.setOption(
    chartOptions(
      "Behavioral Metrics",
      ["Cumulative Reward", "Gradient"],
      [
        {
          type: "value",
          name: "Reward",
          axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } },
        },
        {
          type: "value",
          name: "Gradient",
          axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } },
        },
      ],
      [
        {
          name: "Cumulative Reward",
          type: "line",
          data: cumulativeRewards,
          itemStyle: { color: CONFIG.CHART_PRIMARY },
        },
        {
          name: "Gradient",
          type: "line",
          yAxisIndex: 1,
          data: timeline.map((s) => s.gradient),
          itemStyle: { color: CONFIG.CHART_SECONDARY },
        },
      ]
    ),
    true
  );

  state.charts.neural.setOption(
    chartOptions(
      "Neural Dynamics",
      ["Firing Rate (Hz)", "Mean Potential (mV)"],
      [
        {
          type: "value",
          name: "Hz",
          axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } },
        },
        {
          type: "value",
          name: "mV",
          axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } },
        },
      ],
      [
        {
          name: "Firing Rate (Hz)",
          type: "line",
          data: timeline.map((s) => s.neural?.firing_rate_hz),
          itemStyle: { color: CONFIG.CHART_NEURAL_1 },
        },
        {
          name: "Mean Potential (mV)",
          type: "line",
          yAxisIndex: 1,
          data: timeline.map((s) => s.neural?.mean_potential),
          itemStyle: { color: CONFIG.CHART_NEURAL_2 },
        },
      ]
    ),
    true
  );

  state.charts.learning.setOption(
    chartOptions(
      "Learning Signals",
      ["Dopamine", "Value Estimate"],
      { type: "value", axisLine: { lineStyle: { color: CONFIG.CHART_TEXT } } },
      [
        {
          name: "Dopamine",
          type: "line",
          data: timeline.map((s) => s.neural?.dopamine),
          itemStyle: { color: CONFIG.CHART_LEARNING_1 },
        },
        {
          name: "Value Estimate",
          type: "line",
          data: timeline.map((s) => s.neural?.value_estimate),
          itemStyle: { color: CONFIG.CHART_LEARNING_2 },
        },
      ]
    ),
    true
  );
}

function renderAllCharts() {
  Object.values(state.charts).forEach((chart) => chart.resize());
  updateCharts();
}

function updateCharts() {
  const markLineOpt = {
    symbol: "none",
    lineStyle: { type: "solid", color: CONFIG.CHART_PRIMARY, width: 2 },
    data: [{ xAxis: state.currentStep }],
    silent: true,
  };
  Object.values(state.charts).forEach((chart) => {
    chart.setOption({
      series: [{ markLine: markLineOpt }, { markLine: markLineOpt }],
    });
  });
}

function showLoading(message) {
  DOM.loadingOverlay.textContent = message;
  DOM.loadingOverlay.style.display = "flex";
}

function hideLoading() {
  DOM.loadingOverlay.style.display = "none";
}

init();
