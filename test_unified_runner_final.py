#!/usr/bin/env python3
"""Test the final unified model runner implementation."""

import subprocess
import json
from pathlib import Path
import shutil

def run_test(cmd, description):
    """Run a test command and check output."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Success!")
        
        # Extract experiment directory from output
        for line in result.stdout.split('\n'):
            if "Experiment directory:" in line:
                exp_dir = line.split(":", 1)[1].strip()
                print(f"  Experiment directory: {exp_dir}")
                return True, exp_dir
            elif "Experiment output:" in line:
                exp_dir = line.split(":", 1)[1].strip()
                print(f"  Experiment output: {exp_dir}")
                return True, exp_dir
                
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, None

def check_experiment_dir(exp_dir):
    """Check that experiment directory contains expected files."""
    if not exp_dir:
        return False
        
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"✗ Experiment directory not found: {exp_dir}")
        return False
        
    expected_files = [
        "metadata.json",
        "results.json",
        "summary.json",
        "command_args.json"
    ]
    
    print(f"\nChecking experiment directory: {exp_dir}")
    found_files = []
    missing_files = []
    
    for file in expected_files:
        if (exp_path / file).exists():
            found_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"  Found files: {', '.join(found_files)}")
    if missing_files:
        print(f"  Missing files: {', '.join(missing_files)}")
    
    # Check summary content
    summary_path = exp_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Agent: {summary.get('agent', 'N/A')}")
        print(f"  Episodes: {summary.get('performance', {}).get('total_time', 'N/A'):.1f}s")
        print(f"  Mean reward: {summary.get('rewards', {}).get('mean', 'N/A'):.1f}")
    
    return len(missing_files) == 0

def main():
    print("Testing Final Unified Model Runner")
    print("="*60)
    
    # Clean up any previous test experiments
    if Path("experiments").exists():
        for item in Path("experiments").glob("*_test_*"):
            if item.is_dir():
                shutil.rmtree(item)
    
    tests = [
        # Basic random agent test
        (["python", "-m", "models.runner", "random", "--n-episodes", "2", "--quiet"], 
         "Random agent basic run", True),
        
        # Phase 0.13 with small grid
        (["python", "-m", "models.runner", "phase_0_13", "--grid-size", "10", "--n-episodes", "2", "--quiet"],
         "Phase 0.13 small grid", True),
        
        # With neural config
        (["python", "-m", "models.runner", "phase_0_13", "--neural-config", "configs/neural/small.json", "--n-episodes", "1", "--quiet"],
         "Phase 0.13 with small neural config", True),
        
        # With learning rules config
        (["python", "-m", "models.runner", "phase_0_13", "--learning-rules-config", "configs/learning_rules/minimal.json", "--n-episodes", "1", "--quiet"],
         "Phase 0.13 with minimal learning rules", True),
        
        # No export mode
        (["python", "-m", "models.runner", "random", "--n-episodes", "2", "--no-export", "--quiet"],
         "Random agent with no export", False),
    ]
    
    passed = 0
    failed = 0
    
    for cmd, description, check_dir in tests:
        success, exp_dir = run_test(cmd, description)
        
        if success:
            if check_dir and exp_dir:
                if check_experiment_dir(exp_dir):
                    passed += 1
                else:
                    failed += 1
                    success = False
            else:
                passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        
        # Show example usage
        print("\n" + "="*60)
        print("EXAMPLE USAGE")
        print("="*60)
        print("# Run with default settings:")
        print("python -m models.runner phase_0_13")
        print("\n# Run with custom configurations:")
        print("python -m models.runner phase_0_13 \\")
        print("    --neural-config configs/neural/large.json \\")
        print("    --learning-config configs/learning/stable.json \\")
        print("    --learning-rules-config configs/learning_rules/no_dale.json \\")
        print("    --grid-size 50 --n-episodes 100")
        print("\n# Check results:")
        print("ls experiments/phase_0_13_0001/")
    else:
        print(f"\n✗ {failed} tests failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())