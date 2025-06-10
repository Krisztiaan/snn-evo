#!/usr/bin/env python3
"""Test the unified model runner interface."""

import subprocess
import json
from pathlib import Path

def run_test(cmd, description):
    """Run a test command and report results."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Success!")
        if result.stdout:
            print("\nOutput preview:")
            lines = result.stdout.strip().split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("Testing Unified Model Runner")
    print("="*60)
    
    tests = [
        # Basic functionality
        (["python", "-m", "models.runner", "random", "--n-episodes", "2", "--quiet"], 
         "Random agent basic run"),
        
        (["python", "-m", "models.runner", "phase_0_13", "--n-episodes", "2", "--quiet"],
         "Phase 0.13 agent basic run"),
        
        # Custom parameters
        (["python", "-m", "models.runner", "phase_0_13", "--grid-size", "10", "--n-episodes", "1", "--quiet"],
         "Custom grid size"),
        
        # Configuration files
        (["python", "-m", "models.runner", "phase_0_13", "--neural-config", "configs/neural/small.json", "--n-episodes", "1", "--quiet"],
         "Small neural config"),
        
        (["python", "-m", "models.runner", "phase_0_13", "--learning-config", "configs/learning/fast.json", "--n-episodes", "1", "--quiet"],
         "Fast learning config"),
        
        # Output to JSON
        (["python", "-m", "models.runner", "random", "--n-episodes", "2", "--output-json", "test_output.json", "--quiet"],
         "JSON output"),
        
        # No export mode
        (["python", "-m", "models.runner", "phase_0_13", "--n-episodes", "2", "--no-export", "--quiet"],
         "No export mode (max performance)"),
    ]
    
    passed = 0
    failed = 0
    
    for cmd, description in tests:
        if run_test(cmd, description):
            passed += 1
        else:
            failed += 1
    
    # Check JSON output
    if Path("test_output.json").exists():
        print("\n" + "="*60)
        print("Checking JSON output...")
        with open("test_output.json") as f:
            data = json.load(f)
        print(f"✓ Valid JSON with {len(data)} top-level keys")
        print(f"  Agent: {data.get('agent', 'N/A')}")
        print(f"  Episodes: {len(data.get('summaries', []))}")
        print(f"  Mean reward: {data.get('rewards', {}).get('mean', 'N/A'):.1f}")
        Path("test_output.json").unlink()  # Clean up
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} tests failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())