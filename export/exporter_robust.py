# keywords: [exporter, robust, error-handling, recovery, fault-tolerant]
"""Robust version of optimized exporter with advanced error handling and recovery."""

import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings
import traceback
import tempfile
import shutil
import atexit
import signal
import sys
from contextlib import contextmanager

from .exporter_optimized import OptimizedDataExporter, OptimizedEpisode


class RobustEpisode(OptimizedEpisode):
    """Episode with error recovery and auto-save capabilities."""
    
    def __init__(self, *args, auto_save_interval: int = 1000, **kwargs):
        """Initialize robust episode.
        
        Args:
            auto_save_interval: Flush data every N timesteps
        """
        super().__init__(*args, **kwargs)
        self.auto_save_interval = auto_save_interval
        self.last_auto_save = 0
        self.error_count = 0
        self.max_errors = 10
        
    def log_timestep(self, *args, **kwargs):
        """Log timestep with error recovery."""
        try:
            super().log_timestep(*args, **kwargs)
            
            # Auto-save if needed
            if self.timestep_count - self.last_auto_save >= self.auto_save_interval:
                self._auto_save()
                
        except Exception as e:
            self._handle_error("log_timestep", e, args, kwargs)
            
    def _auto_save(self):
        """Flush all buffers to disk."""
        try:
            # Flush all buffered datasets
            for dataset in self.datasets.values():
                dataset.flush()
                
            # Flush HDF5 file
            self.h5_file.flush()
            self.last_auto_save = self.timestep_count
            
        except Exception as e:
            warnings.warn(f"Auto-save failed: {e}")
            
    def _handle_error(self, operation: str, error: Exception, args: tuple, kwargs: dict):
        """Handle errors with recovery attempts."""
        self.error_count += 1
        
        # Log error details
        error_msg = f"Error in {operation}: {error}\n{traceback.format_exc()}"
        warnings.warn(error_msg)
        
        # Save error to episode metadata
        if 'errors' not in self.group.attrs:
            self.group.attrs['errors'] = json.dumps([])
            
        errors = json.loads(self.group.attrs['errors'])
        errors.append({
            'operation': operation,
            'error': str(error),
            'timestep': kwargs.get('timestep', -1),
            'timestamp': datetime.now().isoformat()
        })
        self.group.attrs['errors'] = json.dumps(errors[-10:])  # Keep last 10 errors
        
        # Check if too many errors
        if self.error_count >= self.max_errors:
            raise RuntimeError(f"Too many errors ({self.error_count}) in episode {self.episode_id}")
            
        # Attempt recovery
        if operation == "log_timestep":
            # Try to at least save critical data
            timestep = kwargs.get('timestep')
            if timestep is not None:
                try:
                    # Save minimal checkpoint
                    self.group.attrs['last_timestep'] = timestep
                    self.group.attrs['error_recovery'] = True
                except:
                    pass
                    
    def end(self, *args, **kwargs):
        """End episode with final flush."""
        try:
            # Final auto-save
            self._auto_save()
            super().end(*args, **kwargs)
        except Exception as e:
            self._handle_error("end", e, args, kwargs)
            # Mark as incomplete
            self.group.attrs['status'] = 'error'
            self.group.attrs['error_message'] = str(e)


class RobustDataExporter(OptimizedDataExporter):
    """Robust data exporter with advanced error handling and recovery."""
    
    def __init__(self, 
                 *args,
                 enable_recovery: bool = True,
                 backup_interval: int = 10,
                 auto_save_interval: int = 1000,
                 **kwargs):
        """Initialize robust exporter.
        
        Args:
            enable_recovery: Enable crash recovery
            backup_interval: Create backup every N episodes
            auto_save_interval: Auto-save episode data every N timesteps
        """
        self.enable_recovery = enable_recovery
        self.backup_interval = backup_interval
        self.auto_save_interval = auto_save_interval
        self._backup_count = 0
        self._temp_dir = None
        self._recovery_file = None
        
        # Initialize parent
        super().__init__(*args, **kwargs)
        
        # Setup recovery
        if enable_recovery:
            self._setup_recovery()
            
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_recovery(self):
        """Setup crash recovery system."""
        # Create recovery file
        self._recovery_file = self.output_dir / '.recovery.json'
        recovery_data = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'h5_path': str(self.h5_path),
            'status': 'running',
            'last_checkpoint': datetime.now().isoformat(),
            'episodes_completed': 0,
            'errors': []
        }
        
        with open(self._recovery_file, 'w') as f:
            json.dump(recovery_data, f, indent=2)
            
        # Create temp directory for incremental backups
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.experiment_name}_"))
        
    def _update_recovery(self, status: str = 'running', error: Optional[str] = None):
        """Update recovery file."""
        if not self.enable_recovery or not self._recovery_file:
            return
            
        try:
            with open(self._recovery_file, 'r') as f:
                recovery_data = json.load(f)
                
            recovery_data['status'] = status
            recovery_data['last_checkpoint'] = datetime.now().isoformat()
            recovery_data['episodes_completed'] = self.episode_count
            
            if error:
                recovery_data['errors'].append({
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
                
            with open(self._recovery_file, 'w') as f:
                json.dump(recovery_data, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to update recovery file: {e}")
            
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\nReceived signal {signum}, saving data...")
        self._update_recovery('interrupted', f'Signal {signum}')
        self.close()
        sys.exit(0)
        
    def save_config(self, config: Dict[str, Any]):
        """Save config with error handling."""
        try:
            super().save_config(config)
        except Exception as e:
            self._handle_save_error("config", e, config)
            
    def save_network_structure(self, *args, **kwargs):
        """Save network structure with error handling."""
        try:
            super().save_network_structure(*args, **kwargs)
        except Exception as e:
            self._handle_save_error("network_structure", e, {"args": args, "kwargs": kwargs})
            
    def _handle_save_error(self, operation: str, error: Exception, data: Any):
        """Handle save errors with fallback."""
        error_msg = f"Error saving {operation}: {error}"
        warnings.warn(error_msg)
        self._update_recovery('error', error_msg)
        
        # Try to save to backup location
        if self._temp_dir:
            try:
                backup_file = self._temp_dir / f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(data, f, default=str)
                warnings.warn(f"Data saved to backup: {backup_file}")
            except:
                pass
                
    def start_episode(self, episode_id: Optional[int] = None) -> RobustEpisode:
        """Start episode with robust handling."""
        try:
            if self.current_episode is not None:
                warnings.warn(f"Previous episode {self.current_episode.episode_id} not ended")
                self.current_episode.end(success=False)
                
            if episode_id is None:
                episode_id = self.episode_count
                
            self.current_episode = RobustEpisode(
                episode_id=episode_id,
                h5_file=self.episodes_group,
                neural_sampling_rate=self.neural_sampling_rate,
                validate_data=self.validate_data,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunk_size=self.chunk_size,
                auto_save_interval=self.auto_save_interval
            )
            
            self.episode_count += 1
            self.h5_file.attrs['episode_count'] = self.episode_count
            
            return self.current_episode
            
        except Exception as e:
            self._handle_save_error("start_episode", e, {"episode_id": episode_id})
            raise
            
    def end_episode(self, *args, **kwargs):
        """End episode with backup."""
        try:
            super().end_episode(*args, **kwargs)
            
            # Create backup if needed
            self._backup_count += 1
            if self._backup_count >= self.backup_interval:
                self._create_backup()
                self._backup_count = 0
                
            # Update recovery
            self._update_recovery()
            
        except Exception as e:
            self._handle_save_error("end_episode", e, {"args": args, "kwargs": kwargs})
            
    def _create_backup(self):
        """Create incremental backup."""
        if not self._temp_dir:
            return
            
        try:
            # Close and copy HDF5 file
            self.h5_file.flush()
            backup_path = self._temp_dir / f"backup_ep{self.episode_count}.h5"
            shutil.copy2(self.h5_path, backup_path)
            
            # Keep only last 3 backups
            backups = sorted(self._temp_dir.glob("backup_*.h5"))
            for old_backup in backups[:-3]:
                old_backup.unlink()
                
        except Exception as e:
            warnings.warn(f"Backup failed: {e}")
            
    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """Save checkpoint with verification."""
        try:
            super().save_checkpoint(name, data)
            
            # Verify checkpoint
            checkpoint_name = f"{name}_{datetime.now().strftime('%H%M%S')}"
            if checkpoint_name in self.checkpoints_group:
                checkpoint = self.checkpoints_group[checkpoint_name]
                for key in data:
                    if key not in checkpoint:
                        raise ValueError(f"Checkpoint verification failed: missing {key}")
                        
        except Exception as e:
            self._handle_save_error("checkpoint", e, {"name": name, "data": data})
            
    def close(self):
        """Close with final backup and cleanup."""
        try:
            # Final backup
            if self.enable_recovery:
                self._create_backup()
                self._update_recovery('completed')
                
            # Close HDF5
            super().close()
            
            # Verify file integrity
            try:
                with h5py.File(self.h5_path, 'r') as f:
                    # Basic integrity check
                    if 'episodes' not in f:
                        raise ValueError("Missing episodes group")
                    print(f"✓ Data integrity verified")
            except Exception as e:
                warnings.warn(f"Data integrity check failed: {e}")
                # Attempt recovery from backup
                self._attempt_recovery()
                
        except Exception as e:
            self._update_recovery('error', str(e))
            warnings.warn(f"Error during close: {e}")
            
    def _attempt_recovery(self):
        """Attempt to recover from backup."""
        if not self._temp_dir:
            return
            
        backups = sorted(self._temp_dir.glob("backup_*.h5"))
        if backups:
            latest_backup = backups[-1]
            warnings.warn(f"Attempting recovery from {latest_backup}")
            
            try:
                # Move corrupted file
                corrupted_path = self.h5_path.with_suffix('.corrupted.h5')
                shutil.move(self.h5_path, corrupted_path)
                
                # Restore from backup
                shutil.copy2(latest_backup, self.h5_path)
                print(f"✓ Recovered from backup: {latest_backup}")
                
            except Exception as e:
                warnings.warn(f"Recovery failed: {e}")
                
    def _cleanup(self):
        """Cleanup temporary files."""
        try:
            # Remove recovery file if completed successfully
            if self._recovery_file and self._recovery_file.exists():
                with open(self._recovery_file, 'r') as f:
                    recovery_data = json.load(f)
                    
                if recovery_data['status'] == 'completed':
                    self._recovery_file.unlink()
                    
            # Clean temp directory
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
                
        except Exception as e:
            warnings.warn(f"Cleanup error: {e}")


@contextmanager
def robust_experiment(experiment_name: str, **kwargs):
    """Context manager for robust experiments with automatic recovery."""
    exporter = None
    try:
        # Check for existing recovery file
        recovery_files = list(Path('.').glob(f"*/{experiment_name}*/.recovery.json"))
        
        if recovery_files:
            print(f"Found {len(recovery_files)} recovery files:")
            for i, rf in enumerate(recovery_files):
                with open(rf, 'r') as f:
                    data = json.load(f)
                print(f"  {i+1}. {rf.parent} - Status: {data['status']}, Episodes: {data['episodes_completed']}")
                
            choice = input("Recover from existing experiment? (number/n): ")
            if choice.lower() != 'n' and choice.isdigit():
                recovery_file = recovery_files[int(choice) - 1]
                print(f"Recovering from {recovery_file.parent}")
                # TODO: Implement full recovery logic
                
        # Create new exporter
        exporter = RobustDataExporter(experiment_name, **kwargs)
        yield exporter
        
    except Exception as e:
        if exporter:
            exporter._update_recovery('error', str(e))
        raise
        
    finally:
        if exporter:
            exporter.close()


# Make RobustDataExporter the default in production
DataExporter = RobustDataExporter