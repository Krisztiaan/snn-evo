# keywords: [visualization, report generator, static site, analytics, jinja2]
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from jinja2 import Environment, FileSystemLoader

from data_parser import get_all_experiment_data, calculate_experiment_metrics, _get_nested


# --- CONFIGURATION ---
EXPERIMENTS_DIR = Path("../../experiments")
OUTPUT_DIR = Path("./")
TEMPLATE_DIR = Path("./templates")


def prepare_header_stats(experiments):
    """Calculate the high-level stats for the dashboard header."""
    if not experiments:
        return {
            "total_experiments": 0,
            "best_reward": {"value": 0, "name": "N/A"},
            "agent_types": 0
        }

    total_experiments = len(experiments)
    
    best_reward_val = -np.inf
    best_reward_name = "N/A"
    for exp in experiments:
        mean_reward = _get_nested(exp, ['summary', 'reward_stats', 'mean'], -np.inf)
        if mean_reward > best_reward_val:
            best_reward_val = mean_reward
            best_reward_name = exp['name']

    agent_types = len(set(exp.get('agent', 'unknown') for exp in experiments))

    return {
        "total_experiments": total_experiments,
        "best_reward": {"value": f"{best_reward_val:.2f}", "name": best_reward_name},
        "agent_types": agent_types
    }


def prepare_dashboard_data(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pre-processes data specifically for the dashboard plots."""
    # Extract key parameters and results for parallel coordinates
    dimensions = []
    
    # Add result metrics first
    result_keys = [
        ('avg_reward', 'metrics.avg_reward'),
        ('std_reward', 'metrics.std_reward'),
        ('learning_progress', 'metrics.learning_progress'),
    ]
    
    for label, path in result_keys:
        values = []
        for exp in experiments:
            value = exp
            for key in path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = 0
                    break
            values.append(float(value) if value is not None else 0)
        
        if any(v != 0 for v in values):  # Only add if there's actual data
            dimensions.append({
                'label': label.replace('_', ' ').title(),
                'values': values,
                'range': [min(values), max(values)] if values else [0, 1]
            })
    
    # Add more comprehensive parameters
    param_keys = [
        ('N Neurons', 'config.neural.n_neurons'),
        ('Connection Prob', 'config.neural.connection_probability'),
        ('Excitatory Ratio', 'config.neural.excitatory_ratio'),
        ('STDP Potentiation LR', 'config.plasticity.stdp_lr_potentiation'),
        ('STDP Depression LR', 'config.plasticity.stdp_lr_depression'),
        ('Homeostasis LR', 'config.plasticity.homeostasis_lr'),
        ('Dopamine Boost', 'config.plasticity.dopamine_reward_boost'),
        ('Grid Size', 'config.world.grid_size'),
        ('Max Timesteps', 'config.world.max_timesteps')
    ]

    for label, path in param_keys:
        values = []
        path_keys = path.split('.')
        for exp in experiments:
            value = exp
            try:
                for key in path_keys:
                    # Handle both dicts and objects with attributes
                    if isinstance(value, dict):
                        value = value[key]
                    else:
                        value = getattr(value, key)
                values.append(float(value))
            except (KeyError, AttributeError, TypeError):
                # If a parameter doesn't exist for an experiment, use 0
                values.append(0)
        
        # Only add the dimension if it has non-zero values
        if any(v != 0 for v in values):
            dimensions.append({
                'label': label,
                'values': values,
                'range': [min(values), max(values)] if values else [0, 1]
            })
    
    # Data for box plot (reward distributions)
    box_data = []
    for exp in experiments:
        if 'episode_stats' in exp['summary'] and exp['summary']['episode_stats']:
            rewards = [ep.get('total_reward', 0) for ep in exp['summary']['episode_stats']]
            if rewards:  # Only add if there are rewards
                box_data.append({
                    'name': exp['name'][:30] + '...' if len(exp['name']) > 30 else exp['name'],
                    'y': rewards,
                    'type': 'box',
                    'boxmean': True
                })
    
    return {
        'parallel_coords': {
            'dimensions': dimensions,
            'line': {
                'color': list(range(len(experiments))),
                'colorscale': 'Viridis'
            }
        },
        'box_plot': box_data
    }


def extract_experiment_date(exp_name: str) -> str:
    """Extract date from experiment name if present."""
    # Look for patterns like 20250610_123456
    import re
    date_match = re.search(r'(\d{8})_(\d{6})', exp_name)
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"
    return "Unknown"


def prepare_experiment_for_template(exp_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare experiment data for template rendering."""
    # Calculate metrics
    metrics = calculate_experiment_metrics(exp_data)
    
    # Extract date
    date = extract_experiment_date(exp_data['name'])
    
    # Add computed fields
    exp_data['metrics'] = metrics
    exp_data['date'] = date
    
    return exp_data


def main():
    """Main generation function."""
    print("🚀 Starting report generation...")
    
    # 1. Setup output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "runs").mkdir(exist_ok=True)
    
    # Static files are already in place, no need to copy
    
    # 2. Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    def safe_average(generator):
        lst = list(generator)  # Convert generator to list
        return sum(lst) / len(lst) if lst else 0
    env.filters['average'] = safe_average
    
    dashboard_template = env.get_template("dashboard.html")
    run_detail_template = env.get_template("run_detail.html")
    
    # 3. Load and parse all experiment data
    print(f"📊 Loading experiments from {EXPERIMENTS_DIR.resolve()}...")
    all_experiments = get_all_experiment_data(str(EXPERIMENTS_DIR))
    
    if not all_experiments:
        print("❌ No experiments found. Exiting.")
        return
    
    print(f"✅ Found {len(all_experiments)} experiments")
    
    # 4. Prepare experiments for templates
    for exp in all_experiments:
        prepare_experiment_for_template(exp)
    
    # Sort by average reward (descending)
    all_experiments.sort(key=lambda x: x['metrics']['avg_reward'], reverse=True)
    
    # 5. Calculate dashboard statistics
    print("📊 Calculating dashboard statistics...")
    header_stats = prepare_header_stats(all_experiments)
    dashboard_plot_data = prepare_dashboard_data(all_experiments)
    
    # 6. Generate dashboard
    print("📝 Generating dashboard...")
    
    dashboard_html = dashboard_template.render(
        experiments=all_experiments,
        experiments_json=json.dumps(all_experiments),
        plots_data=json.dumps(dashboard_plot_data),  # Use the new prepare_dashboard_data result
        header_stats=header_stats,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        url_prefix=""
    )
    
    with open(OUTPUT_DIR / "index.html", "w") as f:
        f.write(dashboard_html)
    
    # 7. Generate individual run pages
    print("📄 Generating individual run pages...")
    for exp_data in all_experiments:
        try:
            run_html = run_detail_template.render(
                exp=exp_data,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                url_prefix="../"
            )
            
            with open(OUTPUT_DIR / "runs" / f"{exp_data['name']}.html", "w") as f:
                f.write(run_html)
                
            print(f"  ✓ Generated {exp_data['name']}.html")
        except Exception as e:
            print(f"  ✗ Failed to generate {exp_data['name']}.html: {e}")
    
    # 8. Create placeholder static files if they don't exist
    css_dir = OUTPUT_DIR / "static" / "css"
    js_dir = OUTPUT_DIR / "static" / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal CSS if not exists
    css_file = css_dir / "style.css"
    if not css_file.exists():
        print("📝 Creating default CSS...")
        with open(css_file, "w") as f:
            f.write("""/* Basic styling - replace with your own */
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; }
header { background: #1e293b; color: white; padding: 1rem; }
main { padding: 2rem; }
.tab-button { padding: 0.5rem 1rem; margin-right: 0.5rem; border: none; background: #e2e8f0; cursor: pointer; }
.tab-button.active { background: #3b82f6; color: white; }
.tab-content { display: none; padding: 1rem; }
.tab-content.active-tab { display: block; }
.plot-container { width: 100%; height: 400px; margin: 1rem 0; }
.stats-panel { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-card { flex: 1; padding: 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }
""")
    
    # Create minimal JS if not exists
    js_file = js_dir / "app.js"
    if not js_file.exists():
        print("📝 Creating default JavaScript...")
        with open(js_file, "w") as f:
            f.write("""// Placeholder - will be replaced with full implementation
console.log('SNN Analytics loaded');
""")
    
    print(f"\n✅ Report generated successfully!")
    print(f"📂 Output directory: {OUTPUT_DIR.resolve()}")
    print(f"🌐 Open '{OUTPUT_DIR.resolve()}/index.html' in your browser to view.")


if __name__ == "__main__":
    main()