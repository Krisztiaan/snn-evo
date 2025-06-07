import numpy as np

# Test empty array properties
empty = np.array([])
print(f"Empty array shape: {empty.shape}")
print(f"Empty array ndim: {empty.ndim}")
print(f"Empty array size: {empty.size}")

# What shape do we get when we add dimension?
print(f"Shape with added dimension: {(0,) + empty.shape}")

# Test with empty 2D array
empty2d = np.array([]).reshape(0, 5)
print(f"\nEmpty 2D array shape: {empty2d.shape}")
print(f"Shape with added dimension: {(0,) + empty2d.shape}")

# Test creating HDF5 dataset with empty array
import h5py
import tempfile

with tempfile.NamedTemporaryFile(suffix='.h5') as f:
    with h5py.File(f.name, 'w') as h5f:
        # Try different approaches
        try:
            # Direct creation
            h5f.create_dataset('empty1', data=empty)
            print("\nDirect creation of empty dataset: SUCCESS")
        except Exception as e:
            print(f"\nDirect creation failed: {e}")
        
        try:
            # With explicit shape
            h5f.create_dataset('empty2', shape=(0,), dtype=np.float64)
            print("Creation with shape=(0,): SUCCESS")
        except Exception as e:
            print(f"Creation with shape=(0,) failed: {e}")
        
        try:
            # With chunks
            h5f.create_dataset('empty3', shape=(0,), dtype=np.float64, chunks=(1,))
            print("Creation with chunks=(1,): SUCCESS")
        except Exception as e:
            print(f"Creation with chunks=(1,) failed: {e}")
        
        try:
            # Resizable empty
            h5f.create_dataset('empty4', shape=(0,), maxshape=(None,), dtype=np.float64)
            print("Creation with maxshape=(None,): SUCCESS")
            
            # Try to resize and add data
            ds = h5f['empty4']
            ds.resize((1,))
            ds[0] = 42.0
            print(f"After resize and write: shape={ds.shape}, value={ds[0]}")
        except Exception as e:
            print(f"Resizable empty failed: {e}")