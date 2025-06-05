# keywords: [benchmark, performance, testing, exporter, comparison]
"""Comprehensive benchmarking suite for data exporters."""

import time
import numpy as np
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd

from export.exporter import DataExporter
from export.exporter_optimized import OptimizedDataExporter


class ExporterBenchmark:
    """Benchmark suite for comparing exporter implementations."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def benchmark_scenario(self,
                          name: str,
                          n_episodes: int = 10,
                          timesteps_per_episode: int = 10000,
                          n_neurons: int = 1000,
                          spike_rate: float = 0.1,
                          weight_change_rate: float = 0.01,
                          neural_sampling_rate: int = 100) -> Dict[str, Any]:
        """Run a benchmark scenario."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        print(f"Episodes: {n_episodes}")
        print(f"Timesteps per episode: {timesteps_per_episode}")
        print(f"Neurons: {n_neurons}")
        print(f"Spike rate: {spike_rate}")
        print(f"Weight change rate: {weight_change_rate}")
        
        # Generate test data
        neurons = {
            "neuron_ids": np.arange(n_neurons),
            "neuron_types": np.array(["excitatory"] * int(n_neurons * 0.8) + 
                                   ["inhibitory"] * int(n_neurons * 0.2)),
            "positions": np.random.rand(n_neurons, 2)
        }
        
        n_connections = n_neurons * 10  # Sparse connectivity
        connections = {
            "source_ids": np.random.randint(0, n_neurons, n_connections),
            "target_ids": np.random.randint(0, n_neurons, n_connections),
        }
        initial_weights = np.random.randn(n_connections) * 0.1
        
        # Benchmark standard exporter
        print("\nTesting standard exporter...")
        std_results = self._run_exporter(
            DataExporter,
            "standard",
            name,
            neurons,
            connections,
            initial_weights,
            n_episodes,
            timesteps_per_episode,
            n_neurons,
            spike_rate,
            weight_change_rate,
            neural_sampling_rate
        )
        
        # Benchmark optimized exporter
        print("\nTesting optimized exporter...")
        opt_results = self._run_exporter(
            OptimizedDataExporter,
            "optimized",
            name,
            neurons,
            connections,
            initial_weights,
            n_episodes,
            timesteps_per_episode,
            n_neurons,
            spike_rate,
            weight_change_rate,
            neural_sampling_rate
        )
        
        # Compare results
        comparison = {
            'scenario': name,
            'parameters': {
                'n_episodes': n_episodes,
                'timesteps_per_episode': timesteps_per_episode,
                'n_neurons': n_neurons,
                'spike_rate': spike_rate,
                'weight_change_rate': weight_change_rate
            },
            'standard': std_results,
            'optimized': opt_results,
            'speedup': {
                'total_time': std_results['total_time'] / opt_results['total_time'],
                'write_speed': opt_results['write_speed'] / std_results['write_speed'],
                'file_size_reduction': 1 - (opt_results['file_size'] / std_results['file_size'])
            }
        }
        
        self.results.append(comparison)
        self._print_comparison(comparison)
        
        return comparison
        
    def _run_exporter(self,
                     exporter_class,
                     exporter_name: str,
                     scenario_name: str,
                     neurons: Dict,
                     connections: Dict,
                     initial_weights: np.ndarray,
                     n_episodes: int,
                     timesteps_per_episode: int,
                     n_neurons: int,
                     spike_rate: float,
                     weight_change_rate: float,
                     neural_sampling_rate: int) -> Dict[str, Any]:
        """Run a single exporter benchmark."""
        
        # Setup
        experiment_name = f"{scenario_name}_{exporter_name}"
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create exporter
        start_time = time.time()
        
        with exporter_class(
            experiment_name=experiment_name,
            output_base_dir=str(self.output_dir),
            neural_sampling_rate=neural_sampling_rate,
            compression='gzip',
            compression_level=4
        ) as exporter:
            
            # Save initial data
            exporter.save_config({
                "n_neurons": n_neurons,
                "spike_rate": spike_rate,
                "weight_change_rate": weight_change_rate
            })
            exporter.save_network_structure(neurons, connections, initial_weights)
            
            # Track metrics
            timestep_times = []
            episode_times = []
            memory_usage = []
            
            # Run episodes
            total_timesteps = 0
            
            for episode in range(n_episodes):
                episode_start = time.time()
                
                # Start episode
                ep = exporter.start_episode()
                
                # Initialize state
                v = np.random.randn(n_neurons) * 10 - 65
                agent_pos = np.array([0.5, 0.5])
                
                # Run timesteps
                for t in range(timesteps_per_episode):
                    timestep_start = time.time()
                    
                    # Neural dynamics
                    v += np.random.randn(n_neurons) * 2
                    
                    # Generate spikes
                    spike_prob = np.random.rand(n_neurons)
                    spikes = spike_prob < spike_rate
                    v[spikes] = -65
                    
                    # Agent movement
                    agent_pos += np.random.randn(2) * 0.01
                    agent_pos = np.clip(agent_pos, 0, 1)
                    
                    # Reward
                    distance = np.linalg.norm(agent_pos - 0.5)
                    reward = 1.0 - distance if distance < 0.5 else 0
                    
                    # Log data
                    exporter.log(
                        timestep=t,
                        neural_state={"membrane_potentials": v} if t % neural_sampling_rate == 0 else None,
                        spikes=spikes,
                        behavior={"x": agent_pos[0], "y": agent_pos[1]},
                        reward=reward
                    )
                    
                    # Weight changes
                    if np.random.rand() < weight_change_rate:
                        n_changes = np.random.poisson(5)
                        for _ in range(n_changes):
                            idx = np.random.randint(len(initial_weights))
                            old_w = initial_weights[idx]
                            new_w = old_w + np.random.randn() * 0.001
                            initial_weights[idx] = new_w
                            
                            exporter.log(
                                timestep=t,
                                synapse_id=idx,
                                old_weight=old_w,
                                new_weight=new_w
                            )
                    
                    timestep_times.append(time.time() - timestep_start)
                    total_timesteps += 1
                    
                    # Sample memory usage
                    if t % 1000 == 0:
                        memory_usage.append(process.memory_info().rss / 1024 / 1024)
                
                # End episode
                exporter.end_episode(success=True)
                episode_times.append(time.time() - episode_start)
                
                print(f"  Episode {episode + 1}/{n_episodes} complete "
                      f"({episode_times[-1]:.2f}s, "
                      f"{timesteps_per_episode / episode_times[-1]:.0f} steps/s)")
        
        # Calculate metrics
        total_time = time.time() - start_time
        file_path = exporter.h5_path
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        
        end_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(memory_usage) if memory_usage else end_memory
        
        results = {
            'total_time': total_time,
            'total_timesteps': total_timesteps,
            'write_speed': total_timesteps / total_time,
            'avg_timestep_time': np.mean(timestep_times),
            'std_timestep_time': np.std(timestep_times),
            'avg_episode_time': np.mean(episode_times),
            'file_size': file_size,
            'file_size_per_timestep': file_size / total_timesteps * 1000,  # KB
            'start_memory': start_memory,
            'peak_memory': peak_memory,
            'memory_growth': peak_memory - start_memory,
            'output_path': str(file_path)
        }
        
        return results
        
    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison results."""
        print(f"\n{'='*60}")
        print(f"Results for: {comparison['scenario']}")
        print(f"{'='*60}")
        
        std = comparison['standard']
        opt = comparison['optimized']
        speedup = comparison['speedup']
        
        print(f"\nPerformance Metrics:")
        print(f"  Total time:")
        print(f"    Standard:  {std['total_time']:.2f}s")
        print(f"    Optimized: {opt['total_time']:.2f}s")
        print(f"    Speedup:   {speedup['total_time']:.2f}x")
        
        print(f"\n  Write speed:")
        print(f"    Standard:  {std['write_speed']:.0f} timesteps/s")
        print(f"    Optimized: {opt['write_speed']:.0f} timesteps/s")
        print(f"    Speedup:   {speedup['write_speed']:.2f}x")
        
        print(f"\n  File size:")
        print(f"    Standard:  {std['file_size']:.2f} MB")
        print(f"    Optimized: {opt['file_size']:.2f} MB")
        print(f"    Reduction: {speedup['file_size_reduction']*100:.1f}%")
        
        print(f"\n  Memory usage:")
        print(f"    Standard:  {std['peak_memory']:.2f} MB (growth: {std['memory_growth']:.2f} MB)")
        print(f"    Optimized: {opt['peak_memory']:.2f} MB (growth: {opt['memory_growth']:.2f} MB)")
        
    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite."""
        # Scenario 1: Small scale, high frequency
        self.benchmark_scenario(
            name="small_high_freq",
            n_episodes=5,
            timesteps_per_episode=10000,
            n_neurons=100,
            spike_rate=0.2,
            weight_change_rate=0.05
        )
        
        # Scenario 2: Medium scale, normal frequency
        self.benchmark_scenario(
            name="medium_normal",
            n_episodes=10,
            timesteps_per_episode=5000,
            n_neurons=500,
            spike_rate=0.1,
            weight_change_rate=0.01
        )
        
        # Scenario 3: Large scale, sparse activity
        self.benchmark_scenario(
            name="large_sparse",
            n_episodes=3,
            timesteps_per_episode=10000,
            n_neurons=2000,
            spike_rate=0.05,
            weight_change_rate=0.001
        )
        
        # Scenario 4: Stress test
        self.benchmark_scenario(
            name="stress_test",
            n_episodes=2,
            timesteps_per_episode=20000,
            n_neurons=1000,
            spike_rate=0.3,
            weight_change_rate=0.1
        )
        
    def plot_results(self):
        """Generate comparison plots."""
        if not self.results:
            print("No results to plot")
            return
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Data Exporter Performance Comparison', fontsize=16)
        
        scenarios = [r['scenario'] for r in self.results]
        
        # Plot 1: Write speed
        ax = axes[0, 0]
        std_speeds = [r['standard']['write_speed'] for r in self.results]
        opt_speeds = [r['optimized']['write_speed'] for r in self.results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        ax.bar(x - width/2, std_speeds, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, opt_speeds, width, label='Optimized', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Write Speed (timesteps/s)')
        ax.set_title('Write Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: File size
        ax = axes[0, 1]
        std_sizes = [r['standard']['file_size'] for r in self.results]
        opt_sizes = [r['optimized']['file_size'] for r in self.results]
        
        ax.bar(x - width/2, std_sizes, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, opt_sizes, width, label='Optimized', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('File Size (MB)')
        ax.set_title('Storage Efficiency')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Speedup factors
        ax = axes[1, 0]
        time_speedups = [r['speedup']['total_time'] for r in self.results]
        write_speedups = [r['speedup']['write_speed'] for r in self.results]
        
        ax.plot(scenarios, time_speedups, 'o-', label='Time speedup', markersize=8)
        ax.plot(scenarios, write_speedups, 's-', label='Write speedup', markersize=8)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Performance Improvements')
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Memory usage
        ax = axes[1, 1]
        std_memory = [r['standard']['memory_growth'] for r in self.results]
        opt_memory = [r['optimized']['memory_growth'] for r in self.results]
        
        ax.bar(x - width/2, std_memory, width, label='Standard', alpha=0.8)
        ax.bar(x + width/2, opt_memory, width, label='Optimized', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Memory Growth (MB)')
        ax.set_title('Memory Efficiency')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_results.png', dpi=150)
        plt.show()
        
    def save_summary(self):
        """Save benchmark summary."""
        if not self.results:
            return
            
        # Create summary DataFrame
        rows = []
        for result in self.results:
            row = {
                'scenario': result['scenario'],
                'timesteps': result['parameters']['timesteps_per_episode'] * 
                            result['parameters']['n_episodes'],
                'neurons': result['parameters']['n_neurons'],
                'std_time': result['standard']['total_time'],
                'opt_time': result['optimized']['total_time'],
                'time_speedup': result['speedup']['total_time'],
                'std_speed': result['standard']['write_speed'],
                'opt_speed': result['optimized']['write_speed'],
                'write_speedup': result['speedup']['write_speed'],
                'std_size': result['standard']['file_size'],
                'opt_size': result['optimized']['file_size'],
                'size_reduction': result['speedup']['file_size_reduction'] * 100,
                'std_memory': result['standard']['memory_growth'],
                'opt_memory': result['optimized']['memory_growth']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'benchmark_summary.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"\nAverage Performance Improvements:")
        print(f"  Time speedup:        {df['time_speedup'].mean():.2f}x")
        print(f"  Write speed speedup: {df['write_speedup'].mean():.2f}x")
        print(f"  File size reduction: {df['size_reduction'].mean():.1f}%")
        print(f"  Memory efficiency:   {(1 - df['opt_memory'].sum() / df['std_memory'].sum()) * 100:.1f}% reduction")
        
        print(f"\nBest improvements:")
        print(f"  Max time speedup:    {df['time_speedup'].max():.2f}x ({df.loc[df['time_speedup'].idxmax(), 'scenario']})")
        print(f"  Max write speedup:   {df['write_speedup'].max():.2f}x ({df.loc[df['write_speedup'].idxmax(), 'scenario']})")
        print(f"  Max size reduction:  {df['size_reduction'].max():.1f}% ({df.loc[df['size_reduction'].idxmax(), 'scenario']})")
        

def main():
    """Run full benchmark suite."""
    print("Starting comprehensive data exporter benchmarks...")
    
    benchmark = ExporterBenchmark()
    
    # Run all benchmarks
    benchmark.run_all_benchmarks()
    
    # Generate plots
    benchmark.plot_results()
    
    # Save summary
    benchmark.save_summary()
    
    print("\nBenchmarks complete! Results saved to benchmarks/")
    

if __name__ == "__main__":
    main()