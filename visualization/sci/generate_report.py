# keywords: [visualization, report generator, static site, analytics, jinja2]
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from jinja2 import Environment, FileSystemLoader

from data_parser import get_all_experiment_data, calculate_experiment_metrics


# --- CONFIGURATION ---
EXPERIMENTS_DIR = Path("../../experiments")
OUTPUT_DIR = Path("./")
TEMPLATE_DIR = Path("./templates")


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
    
    # Add some common parameters if they exist
    # Check if n_neurons exists in any experiment
    if any('neural' in exp.get('config', {}) and 'n_neurons' in exp['config']['neural'] for exp in experiments):
        values = []
        for exp in experiments:
            n_neurons = 0
            if 'neural' in exp.get('config', {}) and 'n_neurons' in exp['config']['neural']:
                n_neurons = exp['config']['neural']['n_neurons']
            values.append(n_neurons)
        dimensions.append({
            'label': 'N Neurons',
            'values': values,
            'range': [min(values), max(values)]
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
    best_reward = max(exp['metrics']['avg_reward'] for exp in all_experiments)
    best_experiment = next(exp['name'] for exp in all_experiments 
                          if exp['metrics']['avg_reward'] == best_reward)
    agent_types = list(set(exp['agent'] for exp in all_experiments))
    
    # 6. Generate dashboard
    print("📝 Generating dashboard...")
    dashboard_plot_data = prepare_dashboard_data(all_experiments)
    
    dashboard_html = dashboard_template.render(
        experiments=all_experiments,
        experiments_json=json.dumps(all_experiments),
        plot_data=json.dumps(dashboard_plot_data),
        best_reward=best_reward,
        best_experiment=best_experiment,
        agent_types=agent_types,
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