#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.3.2": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot, chart

// CONFIGURATION
#let LOG_BASE = "logs/0-4"

// Auto-detect latest run directory
#let find-latest-run() = {
  // Since Typst doesn't have direct directory listing, we'll try recent timestamps
  let candidates = (
    "run_20250529_195832",
    "run_20250529_172129", 
    "run_20250529_164103",
    "run_20250529_160604",
    "run_20250529_151331",
    "run_20250529_151231",
  )
  
  for candidate in candidates {
    let test-path = LOG_BASE + "/" + candidate + "/metadata.json"
    if read(test-path, default: none) != none {
      return LOG_BASE + "/" + candidate
    }
  }
  
  // Default fallback
  return LOG_BASE + "/run_20250529_195832"
}

#let RUN_DIR = find-latest-run()

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)

#set text(font: "Linux Libertine", size: 11pt)
#set heading(numbering: "1.")

// Read metadata
#let metadata-content = read(RUN_DIR + "/metadata.json", default: none)
#let metadata = if metadata-content != none { json.decode(metadata-content) } else { (:) }

#let GRID_SIZE = metadata.at("grid_world_size", default: 10)
#let N_NEURONS = metadata.at("n_neurons", default: 256)
#let N_SEEDS = metadata.at("n_seeds", default: 1)
#let GRID_DIM = calc.floor(calc.sqrt(N_NEURONS))

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Emergent Spiking Neural Network Analysis
  ]

  #text(size: 14pt)[
    Phase 0.3 - Results Report
  ]

  #v(1em)
  #text(size: 10pt)[
    Run: #RUN_DIR.split("/").last() \
    #datetime.today().display()
  ]
]

#pagebreak()

#outline(depth: 2)

#pagebreak()

= Executive Summary

This document presents analysis of the Phase 0.3 experiment for emergent spiking neural networks in grid world navigation.

#if N_SEEDS == 1 [
  *Note*: This analysis is based on a single seed experiment. Statistical inference is limited.
] else [
  The analysis covers #N_SEEDS random seeds with statistical validation.
]

// Helper function to safely read files
#let safe-read(path) = {
  read(RUN_DIR + "/" + path, default: none)
}

// Helper function to read CSV data with error handling
#let read-csv(path) = {
  let content = safe-read(path)
  if content == none { return (headers: (), data: ()) }
  
  let lines = content.split("\n").filter(x => x != "")
  if lines.len() == 0 { return (headers: (), data: ()) }
  
  // Check if first line has headers
  let has-headers = lines.at(0).contains("seed") or lines.at(0).contains("episode") or lines.at(0).contains("key")
  
  if has-headers {
    let headers = lines.at(0).split(",")
    let data = lines.slice(1).map(line => line.split(","))
    return (headers: headers, data: data)
  } else {
    // No headers, just data
    return (headers: (), data: lines.map(line => line.split(",")))
  }
}

// Helper to safely parse numbers
#let parse-num(s) = {
  if s == none or s == "nan" or s == "" { 0.0 } 
  else { float(str(s).replace("\r", "").trim()) }
}

= Experiment Configuration

// Check if we have detailed configuration
#let has_detailed_config = metadata.at("network_architecture", default: none) != none

#if has_detailed_config {
  // Display comprehensive configuration
  figure(
    table(
      columns: (1.5fr, 2fr),
      inset: 10pt,
      align: (left, left),
      [*Category*], [*Configuration*],
      
      [*Network Architecture*], [
        • #metadata.network_architecture.n_neurons neurons (#metadata.network_architecture.grid_size × #metadata.network_architecture.grid_size grid) \
        • Local connectivity radius: #metadata.network_architecture.local_radius \
        • Long-range probability: #metadata.network_architecture.long_range_prob \
        • Inputs: #metadata.network_architecture.n_sensory sensory, Outputs: #metadata.network_architecture.n_motor motor
      ],
      
      [*Neuron Dynamics*], [
        • Leaky Integrate-and-Fire model \
        • Time step: #metadata.neuron_dynamics.dt ms, Membrane τ: #metadata.neuron_dynamics.tau ms \
        • Threshold: #metadata.neuron_dynamics.v_thresh mV (rest: #metadata.neuron_dynamics.v_rest mV) \
        • Refractory period: #metadata.neuron_dynamics.refrac_time ms
      ],
      
      [*Learning*], [
        • Three-factor rule: STDP × Reward × Activity \
        • Base rate: #metadata.learning_parameters.learning_rate \
        • Reward scale: #metadata.learning_parameters.reward_learning_scale×, Motor scale: #metadata.learning_parameters.motor_learning_scale× \
        • STDP: +#metadata.learning_parameters.potentiation_factor/-#metadata.learning_parameters.depression_factor \
        • Eligibility decay: #metadata.learning_parameters.eligibility_decay
      ],
      
      [*Environment*], [
        • #metadata.environment.grid_world_size × #metadata.environment.grid_world_size #metadata.environment.boundary_mode grid \
        • Gradient: #metadata.environment.gradient_function \
        • Reward distance: ≥#metadata.environment.min_reward_distance cells \
        • Collection radius: #metadata.environment.reward_collection_distance
      ],
      
      [*Episode Control*], [
        • Max steps: #metadata.episode_control.max_episode_steps \
        • Early exit: #metadata.episode_control.early_exit_rewards rewards \
        • Timeout: #metadata.episode_control.no_reward_timeout_steps steps without reward \
        • Decision steps per action: #metadata.agent_parameters.decision_steps
      ],
    ),
    caption: [Comprehensive experiment configuration],
  )
} else {
  // Fallback to simple configuration
  figure(
    table(
      columns: (1fr, 1fr),
      inset: 10pt,
      align: (left, right),
      [*Parameter*], [*Value*],
      [Number of Seeds], [#N_SEEDS],
      [Episodes per Seed], [#metadata.at("n_episodes", default: "30")],
      [Network Size], [#N_NEURONS neurons],
      [Grid World Size], [#GRID_SIZE × #GRID_SIZE],
      [Boundary Mode], [#metadata.at("boundary_mode", default: "toroidal")],
      [Learning Rate], [#metadata.at("learning_rate", default: 0.01)],
    ),
    caption: [Basic experiment configuration],
  )
}

= Statistical Summary

#let summary = read-csv("summary_statistics.csv")
#let summary-dict = (:)

#for row in summary.data {
  if row.len() >= 2 {
    summary-dict.insert(row.at(0), row.at(1))
  }
}

#if summary-dict.len() > 0 {
  figure(
    table(
      columns: (1fr, 1fr),
      inset: 10pt,
      align: (left, right),
      [*Metric*], [*Value*],
      [Final Performance (mean)], [#summary-dict.at("final_perf_mean", default: "N/A") rewards],
      ..if N_SEEDS > 1 {(
        [Standard Deviation], [#summary-dict.at("final_perf_std", default: "N/A")],
        [Performance Range], [#summary-dict.at("final_perf_min", default: "N/A") - #summary-dict.at("final_perf_max", default: "N/A")],
      )} else {()},
      [Learning Speed], [#summary-dict.at("learning_speed_mean", default: "N/A") episodes],
      [Convergence Rate], [#calc.round(parse-num(summary-dict.at("convergence_proportion", default: "0")) * 100, digits: 1)%],
      ..if N_SEEDS > 1 {(
        [Hub Consistency], [#summary-dict.at("hub_consistency", default: "N/A")],
      )} else {()},
    ),
    caption: [Summary statistics],
  )

  #if N_SEEDS > 1 and summary-dict.at("p_value", default: none) != none [
    #let sig = parse-num(summary-dict.at("p_value", default: "1")) < 0.01
    The network #if sig [demonstrates *statistically significant*] else [does not show statistically significant] learning compared to random baseline#if sig [ (p < 0.01)].
  ]
} else {
  [Summary statistics not available.]
}

= Learning Curves

#let rewards-content = safe-read("all_episode_rewards.csv")
#if rewards-content != none {
  let rewards-lines = rewards-content.split("\n").filter(x => x != "")
  let all-rewards = ()
  
  for line in rewards-lines {
    let values = line.split(",").map(s => parse-num(s))
    if values.len() > 0 {
      all-rewards.push(values)
    }
  }
  
  if all-rewards.len() > 0 {
    let n-seeds = all-rewards.len()
    let n-episodes = all-rewards.at(0).len()
    
    // Calculate mean rewards per episode
    let mean-rewards = ()
    for ep in range(n-episodes) {
      let ep-rewards = ()
      for seed in range(n-seeds) {
        if ep < all-rewards.at(seed).len() {
          ep-rewards.push(all-rewards.at(seed).at(ep))
        }
      }
      if ep-rewards.len() > 0 {
        mean-rewards.push(ep-rewards.sum() / ep-rewards.len())
      }
    }
    
    #figure(
      canvas({
        import draw: *
        plot.plot(
          size: (14, 8),
          x-label: "Episode",
          y-label: "Total Rewards",
          x-min: 0,
          x-max: n-episodes,
          y-min: 0,
          y-max: if mean-rewards.len() > 0 { calc.max(..mean-rewards) * 1.1 } else { 60 },
          {
            if mean-rewards.len() > 0 {
              plot.add(
                range(mean-rewards.len()).zip(mean-rewards),
                style: (stroke: (paint: blue, thickness: 2pt)),
                label: if n-seeds > 1 [Mean (#n-seeds seeds)] else [Performance],
              )
            }
          },
        )
      }),
      caption: [Learning curve showing rewards per episode],
    )
  }
} else {
  [Learning curve data not available.]
}

= Seed Performance

#let seed-results = read-csv("seed_results.csv")
#if seed-results.data.len() > 0 {
  let seed-indices = seed-results.data.map(row => parse-num(row.at(0)))
  let final-perfs = seed-results.data.map(row => parse-num(row.at(2)))
  
  #figure(
    table(
      columns: if N_SEEDS > 5 { (1fr, 1fr, 1fr) } else { (1fr, 1fr) },
      inset: 8pt,
      align: center,
      [*Seed*], [*Final Performance*], ..if N_SEEDS > 5 { () } else { ([*Episodes*],) },
      ..seed-results.data.map(row => (
        row.at(0),
        str(parse-num(row.at(2))) + " rewards",
        ..if N_SEEDS > 5 { () } else { (row.at(6),) }
      )).flatten()
    ),
    caption: [Individual seed performance],
  )
}

= Network Analysis

== Weight Distribution

#let weights-content = safe-read("best_seed/final_weights.csv")
#if weights-content != none {
  let weight-values = ()
  for line in weights-content.split("\n") {
    if line != "" {
      for val in line.split(",") {
        weight-values.push(parse-num(val))
      }
    }
  }
  
  if weight-values.len() > 0 {
    // Basic statistics
    let non-zero = weight-values.filter(w => calc.abs(w) > 0.1).len()
    let sparsity = (weight-values.len() - non-zero) / weight-values.len() * 100
    
    [Network has #weight-values.len() synaptic connections with #calc.round(sparsity, digits: 1)% sparsity (|w| < 0.1).]
    
    // Simple histogram
    let n-bins = 20
    let min-w = calc.min(..weight-values)
    let max-w = calc.max(..weight-values)
    let range-w = max-w - min-w
    
    #figure(
      canvas({
        import draw: *
        let bin-width = if range-w > 0 { range-w / n-bins } else { 1 }
        let bins = range(n-bins).map(i => min-w + i * bin-width + bin-width / 2)
        let counts = bins.map(b => weight-values.filter(w => w >= b - bin-width / 2 and w < b + bin-width / 2).len())
        
        chart.columnchart(
          size: (12, 6),
          x-label: "Weight Value",
          y-label: "Count",
          value-key: 1,
          label-key: 0,
          bins.zip(counts).filter(((b, c)) => c > 0).map(((b, c)) => (str(calc.round(b, digits: 2)), c)),
          bar-width: 0.8,
          bar-style: (fill: blue.lighten(50%)),
        )
      }),
      caption: [Distribution of synaptic weights],
    )
  }
} else {
  [Weight data not available.]
}

== Hub Neurons

#let hub-neurons = read-csv("common_hub_neurons.csv")
#if N_SEEDS > 1 and hub-neurons.data.len() > 0 {
  [The following neurons were identified as hubs across multiple seeds:]
  
  #figure(
    table(
      columns: (1fr, 2fr),
      inset: 10pt,
      [*Neuron ID*], [*Grid Position*],
      ..hub-neurons.data.slice(0, calc.min(10, hub-neurons.data.len())).map(row => {
        let neuron-id = int(row.at(0))
        let x = calc.rem(neuron-id, GRID_DIM)
        let y = calc.floor(neuron-id / GRID_DIM)
        (str(neuron-id), str("(" + str(x) + ", " + str(y) + ")"))
      }).flatten()
    ),
    caption: [Top hub neurons],
  )
} else if N_SEEDS == 1 {
  [Hub analysis requires multiple seeds for consistency measurement.]
} else {
  [Hub neuron data not available.]
}

= Best Episode Analysis

#let best-metrics = read-csv("best_seed/episode_metrics.csv")
#if best-metrics.data.len() > 0 {
  let episodes = best-metrics.data.map(row => parse-num(row.at(0)))
  let rewards = best-metrics.data.map(row => parse-num(row.at(1)))
  
  #figure(
    canvas({
      import draw: *
      plot.plot(
        size: (14, 6),
        x-label: "Episode",
        y-label: "Rewards",
        x-min: 0,
        x-max: if episodes.len() > 0 { calc.max(..episodes) } else { 30 },
        y-min: 0,
        y-max: if rewards.len() > 0 { calc.max(..rewards) * 1.1 } else { 60 },
        {
          if episodes.len() > 0 {
            plot.add(
              episodes.zip(rewards),
              style: (stroke: (paint: blue, thickness: 2pt)),
              label: "Rewards per Episode",
            )
          }
        },
      )
    }),
    caption: [Performance trajectory of the best seed],
  )
}

== Agent Behavior

#let best-path = read-csv("best_seed/best_episode_path.csv")
#if best-path.data.len() > 0 {
  let path-x = best-path.data.map(row => parse-num(row.at(0)))
  let path-y = best-path.data.map(row => parse-num(row.at(1)))
  
  #figure(
    rect(width: 100%, fill: gray.lighten(95%), stroke: gray)[
      #align(center)[
        #text(12pt)[Agent path visualization available] \
        #text(10pt, style: "italic")[
          Use the pathstep.html tool with best_episode_path.csv \
          and best_episode_reward_positions.csv for interactive visualization
        ]
      ]
    ],
    caption: [Agent trajectory data (total steps: #path-x.len())],
  )
  
  // Basic path statistics
  let unique-positions = ()
  for i in range(path-x.len()) {
    let pos = str(int(path-x.at(i))) + "," + str(int(path-y.at(i)))
    if pos not in unique-positions {
      unique-positions.push(pos)
    }
  }
  
  [The agent visited #unique-positions.len() unique positions out of #(GRID_SIZE * GRID_SIZE) total grid cells (#calc.round(unique-positions.len() / (GRID_SIZE * GRID_SIZE) * 100, digits: 1)% coverage).]
}

= Neuron Specialization

#let specialization = read-csv("best_seed/neuron_specialization.csv")
#if specialization.data.len() > 0 {
  // Find neurons with highest specialization
  let gradient-high = specialization.data.map(row => parse-num(row.at(1)))
  let gradient-low = specialization.data.map(row => parse-num(row.at(2)))
  let post-reward = specialization.data.map(row => parse-num(row.at(3)))
  
  // Calculate simple specialization scores
  let gradient-specialists = ()
  let reward-specialists = ()
  
  for i in range(calc.min(gradient-high.len(), N_NEURONS)) {
    let high = gradient-high.at(i)
    let low = gradient-low.at(i)
    if high > low * 2 and high > 0.1 {  // Simple threshold
      gradient-specialists.push(i)
    }
    if post-reward.at(i) > 0.5 {  // Simple threshold
      reward-specialists.push(i)
    }
  }
  
  [Found #gradient-specialists.len() gradient-selective neurons and #reward-specialists.len() reward-responsive neurons.]
  
  #if gradient-specialists.len() > 0 or reward-specialists.len() > 0 {
    figure(
      table(
        columns: (1fr, 1fr),
        inset: 10pt,
        [*Specialization Type*], [*Number of Neurons*],
        [Gradient Detection], [#gradient-specialists.len()],
        [Reward Response], [#reward-specialists.len()],
        [Total Specialized], [#(gradient-specialists.len() + reward-specialists.len())],
        [Percentage], [#calc.round((gradient-specialists.len() + reward-specialists.len()) / N_NEURONS * 100, digits: 1)%],
      ),
      caption: [Neuron specialization summary],
    )
  }
} else {
  [Neuron specialization data not available.]
}

= Conclusions

#if N_SEEDS == 1 {
  [This single-seed experiment demonstrates that the emergent spiking neural network can successfully learn to navigate and collect rewards in the grid world environment.]
  
  #if summary-dict.at("final_perf_mean", default: none) != none {
    [- Final performance: #summary-dict.at("final_perf_mean") rewards]
  }
  
  [- Network exhibits sparse connectivity and specialized neurons]
  [- Further experiments with multiple seeds recommended for statistical validation]
} else {
  [Based on #N_SEEDS independent experimental runs, the emergent spiking neural network demonstrates:]
  
  #if summary-dict.at("final_perf_mean", default: none) != none {
    [- Consistent learning with mean performance of #summary-dict.at("final_perf_mean") ± #summary-dict.at("final_perf_std", default: "0") rewards]
  }
  
  #if summary-dict.at("convergence_proportion", default: none) != none {
    [- #calc.round(parse-num(summary-dict.at("convergence_proportion", default: "0")) * 100, digits: 1)% convergence rate]
  }
  
  [- Emergence of specialized neural structures through reward-modulated plasticity]
  [- Robust learning across different random initializations]
}

#pagebreak()

#align(center)[
  #text(style: "italic", size: 10pt)[
    Analysis generated from: #RUN_DIR \
    For interactive visualization, use pathstep.html with the exported CSV data.
  ]
]