# Export System Technical Notes

## Current State

The export system is functional but has some inefficiencies:

1. **Compression**: Need to pass compression kwargs to each dataset creation
2. **Resizing**: Currently resizes by 1 element at a time (inefficient)
3. **Sparse Storage**: Spike storage repeats timestep for each spike

## Recommended Improvements (Post-MVP)

1. **Batch Resizing**: Resize in chunks (e.g., 1000 elements)
2. **Better Sparse Format**: Store (timestep, count) + neuron_ids separately
3. **Compression**: Add proper compression to all datasets
4. **Error Handling**: Add try/except and cleanup

## What Works Well

- Simple API
- HDF5 single file storage
- Validation system
- Clear organization
- Efficient loading

## Decision: Keep It Simple

For now, the system works and is good enough for experiments. Optimizations can come later when/if performance becomes an issue. The current implementation prioritizes:

1. Correctness
2. Simplicity  
3. Ease of use

Over premature optimization.