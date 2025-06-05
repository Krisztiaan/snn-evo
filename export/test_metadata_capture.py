# keywords: [test, metadata, capture, validation, hdf5, json]
"""Comprehensive test script for metadata capture functionality."""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

# Add export module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export.exporter import DataExporter
from export.loader import ExperimentLoader


def test_metadata_capture():
    """Test all metadata capture features."""
    
    print("=" * 80)
    print("TESTING METADATA CAPTURE FUNCTIONALITY")
    print("=" * 80)
    
    test_dir = Path("test_metadata_output")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Create exporter with all metadata features
    print("\n1. Creating DataExporter and testing metadata capture...")
    
    with DataExporter(
        experiment_name="metadata_test",
        output_base_dir=str(test_dir),
        neural_sampling_rate=100,
        validate_data=True,
        compression='gzip',
        compression_level=4
    ) as exporter:
        
        print(f"   ✓ Exporter created at: {exporter.output_dir}")
        
        # Test 2: save_config()
        print("\n2. Testing save_config()...")
        config = {
            "model_version": "0.4",
            "n_neurons": 256,
            "dt": 0.1,
            "tau_membrane": 20.0,
            "learning_rate": 0.001,
            "environment": {
                "grid_size": 10,
                "reward_locations": [[5, 5], [8, 8]],
                "obstacle_map": None
            },
            "training": {
                "episodes": 100,
                "max_steps": 1000,
                "batch_size": 32
            },
            "nested_list": [1, [2, 3], {"key": "value"}]
        }
        
        exporter.save_config(config)
        print("   ✓ Configuration saved")
        
        # Verify config files exist
        json_path = exporter.output_dir / 'config.json'
        assert json_path.exists(), "config.json not created"
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)
        assert loaded_config == config, "Config mismatch in JSON"
        print("   ✓ config.json verified")
        
        # Test 3: save_metadata()
        print("\n3. Testing save_metadata()...")
        metadata = {
            "experiment_title": "Metadata Capture Test Suite",
            "author": "Test Script",
            "email": "test@example.com", 
            "institution": "Test Lab",
            "description": "Comprehensive test of all metadata capture features",
            "keywords": ["test", "metadata", "validation"],
            "references": {
                "paper1": "Smith et al. 2024",
                "paper2": "Jones et al. 2023"
            },
            "funding": "Test Grant #12345",
            "version": "1.0.0",
            "custom_field": {"nested": {"data": [1, 2, 3]}}
        }
        
        exporter.save_metadata(metadata)
        print("   ✓ Metadata saved")
        
        # Verify metadata JSON
        metadata_json_path = exporter.output_dir / 'metadata.json'
        assert metadata_json_path.exists(), "metadata.json not created"
        
        with open(metadata_json_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata, "Metadata mismatch in JSON"
        print("   ✓ metadata.json verified")
        
        # Test 4: save_runtime_info() (automatically called in __init__)
        print("\n4. Testing save_runtime_info()...")
        runtime_json_path = exporter.output_dir / 'runtime_info.json'
        assert runtime_json_path.exists(), "runtime_info.json not created"
        
        with open(runtime_json_path, 'r') as f:
            runtime_info = json.load(f)
        
        # Check expected fields
        expected_fields = ['python_version', 'platform', 'hostname', 'numpy_version', 
                          'h5py_version', 'working_directory', 'output_directory']
        for field in expected_fields:
            assert field in runtime_info, f"Missing field: {field}"
            print(f"   ✓ {field}: {runtime_info[field][:50]}..." if len(str(runtime_info[field])) > 50 else f"   ✓ {field}: {runtime_info[field]}")
        
        # Test 5: save_code_snapshot()
        print("\n5. Testing save_code_snapshot()...")
        
        # Save this test file
        saved_files = exporter.save_code_snapshot([__file__])
        print(f"   ✓ Saved {len(saved_files)} code files")
        
        # Verify code snapshot directory
        code_dir = exporter.output_dir / 'code_snapshot'
        assert code_dir.exists(), "code_snapshot directory not created"
        
        # Verify this file was saved
        test_file_name = Path(__file__).name
        saved_test_file = code_dir / test_file_name
        assert saved_test_file.exists(), f"{test_file_name} not saved in code_snapshot"
        
        # Verify content matches
        with open(__file__, 'r') as f:
            original_content = f.read()
        with open(saved_test_file, 'r') as f:
            saved_content = f.read()
        assert saved_content == original_content, "Code content mismatch"
        print(f"   ✓ Code content verified for {test_file_name}")
        
        # Test 6: save_git_info()
        print("\n6. Testing save_git_info()...")
        git_info = exporter.save_git_info()
        
        if git_info:
            print("   ✓ Git info captured:")
            for key, value in git_info.items():
                if key == 'uncommitted_files' and isinstance(value, list):
                    print(f"     - {key}: {len(value)} files")
                else:
                    print(f"     - {key}: {value}")
            
            # Verify git_info.json
            git_json_path = exporter.output_dir / 'git_info.json'
            assert git_json_path.exists(), "git_info.json not created"
        else:
            print("   ⚠ Git info not available (not in a git repository)")
        
        # Test 7: Verify HDF5 file structure
        print("\n7. Verifying HDF5 file structure...")
        
        # Check that HDF5 file exists
        assert exporter.h5_path.exists(), "HDF5 file not created"
        print(f"   ✓ HDF5 file exists: {exporter.h5_path}")
        
        # Check HDF5 attributes
        print("   ✓ HDF5 root attributes:")
        for key, value in exporter.h5_file.attrs.items():
            print(f"     - {key}: {value}")
        
        # Check HDF5 groups
        print("   ✓ HDF5 groups:")
        for group_name in exporter.h5_file.keys():
            print(f"     - /{group_name}")
            if group_name == 'config':
                print("       Config attributes:")
                for key in exporter.h5_file['config'].attrs.keys():
                    print(f"         - {key}")
        
        # Add some dummy episode data to test complete functionality
        print("\n8. Adding dummy episode data...")
        ep = exporter.start_episode()
        
        # Log a few timesteps
        for t in range(10):
            exporter.log(
                timestep=t,
                neural_state={"test_state": np.random.randn(10)},
                behavior={"x": t * 0.1, "y": t * 0.2},
                reward=0.1 if t == 5 else 0.0
            )
        
        exporter.end_episode(success=True, summary={"test_complete": True})
        print("   ✓ Dummy episode added")
    
    print("\n" + "=" * 80)
    print("TESTING DATA RETRIEVAL WITH LOADER")
    print("=" * 80)
    
    # Test 9: Load and verify all metadata with ExperimentLoader
    print("\n9. Loading metadata with ExperimentLoader...")
    
    with ExperimentLoader(exporter.output_dir) as loader:
        # Test get_metadata()
        print("\n   Testing get_metadata():")
        loaded_metadata = loader.get_metadata()
        
        # Check system metadata
        assert 'experiment_name' in loaded_metadata
        assert 'timestamp' in loaded_metadata
        assert 'schema_version' in loaded_metadata
        print("     ✓ System metadata loaded")
        
        # Check user metadata (with 'meta_' prefix removed)
        for key in metadata.keys():
            assert key in loaded_metadata, f"Missing user metadata: {key}"
            if isinstance(metadata[key], dict):
                assert loaded_metadata[key] == metadata[key], f"Metadata mismatch for {key}"
        print("     ✓ User metadata loaded and verified")
        
        # Test get_config()
        print("\n   Testing get_config():")
        loaded_config = loader.get_config()
        assert loaded_config == config, "Config mismatch from loader"
        print("     ✓ Configuration loaded and verified")
        
        # Test get_runtime_info()
        print("\n   Testing get_runtime_info():")
        loaded_runtime = loader.get_runtime_info()
        for field in expected_fields:
            assert field in loaded_runtime, f"Missing runtime field: {field}"
        print("     ✓ Runtime info loaded")
        
        # Test get_git_info()
        print("\n   Testing get_git_info():")
        loaded_git_info = loader.get_git_info()
        if git_info:
            assert loaded_git_info is not None, "Git info not loaded"
            for key in git_info.keys():
                assert key in loaded_git_info, f"Missing git field: {key}"
            print("     ✓ Git info loaded and verified")
        else:
            print("     ⚠ Git info not available")
        
        # Test get_code_snapshot()
        print("\n   Testing get_code_snapshot():")
        code_files = loader.get_code_snapshot()
        assert len(code_files) > 0, "No code files loaded"
        assert test_file_name in code_files, f"{test_file_name} not in code snapshot"
        
        # Verify content
        loaded_code = code_files[test_file_name]
        with open(__file__, 'r') as f:
            current_code = f.read()
        assert loaded_code == current_code, "Code content mismatch from loader"
        print(f"     ✓ Code snapshot loaded and verified ({len(code_files)} files)")
        
        # Test episode data
        print("\n   Testing episode data retrieval:")
        episodes = loader.list_episodes()
        assert len(episodes) == 1, f"Expected 1 episode, got {len(episodes)}"
        
        episode = loader.get_episode(0)
        ep_metadata = episode.get_metadata()
        assert ep_metadata['success'] == True
        assert ep_metadata['total_timesteps'] == 10
        print("     ✓ Episode data loaded")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    
    print(f"\nTest output directory: {exporter.output_dir}")
    print("\nMetadata capture system is fully functional.")
    
    return exporter.output_dir


def verify_json_files(output_dir):
    """Additional verification of JSON files."""
    print("\n" + "=" * 80)
    print("VERIFYING JSON FILES")
    print("=" * 80)
    
    output_path = Path(output_dir)
    
    # List all JSON files
    json_files = list(output_path.glob('*.json'))
    print(f"\nFound {len(json_files)} JSON files:")
    
    for json_file in json_files:
        print(f"\n{json_file.name}:")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                print(f"  - {len(data)} top-level keys")
                for key in list(data.keys())[:5]:  # Show first 5 keys
                    value_type = type(data[key]).__name__
                    print(f"    • {key}: {value_type}")
                if len(data) > 5:
                    print(f"    • ... and {len(data) - 5} more")
            else:
                print(f"  - Type: {type(data).__name__}")
                
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")


def verify_hdf5_metadata(output_dir):
    """Direct verification of HDF5 metadata."""
    print("\n" + "=" * 80)
    print("VERIFYING HDF5 METADATA")
    print("=" * 80)
    
    h5_path = Path(output_dir) / 'experiment_data.h5'
    
    with h5py.File(h5_path, 'r') as f:
        print("\nRoot attributes:")
        for key, value in f.attrs.items():
            print(f"  - {key}: {value}")
        
        print("\nGroups and their attributes:")
        for group_name in f.keys():
            print(f"\n/{group_name}:")
            group = f[group_name]
            
            # Show attributes
            if len(group.attrs) > 0:
                print("  Attributes:")
                for key, value in group.attrs.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    - {key}: [string, {len(value)} chars]")
                    else:
                        print(f"    - {key}: {value}")
            
            # Show subgroups/datasets
            if len(group.keys()) > 0:
                print("  Contents:")
                for item in list(group.keys())[:10]:
                    if isinstance(group[item], h5py.Group):
                        print(f"    - {item}/ (group)")
                    else:
                        print(f"    - {item} (dataset, shape: {group[item].shape})")


if __name__ == "__main__":
    # Run the main test
    output_dir = test_metadata_capture()
    
    # Additional verifications
    verify_json_files(output_dir)
    verify_hdf5_metadata(output_dir)
    
    print("\n✅ All metadata capture features are working correctly!")
    print(f"\nYou can explore the test output at: {output_dir}")