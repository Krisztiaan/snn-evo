# keywords: [grid world, jax environment, simple navigation, toroidal grid]
"""Simple grid world implementation with JAX optimization."""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random, jit, Array

from .types import WorldState, Observation, StepResult, WorldConfig


class SimpleGridWorld:
    """A minimal grid world for navigation tasks.
    
    Features:
    - Toroidal (wraparound) grid
    - Single agent navigating to collect rewards
    - Gradient-based observation (distance to nearest reward)
    - JAX-optimized for performance
    """
    
    # World metadata
    NAME = "SimpleGridWorld"
    VERSION = "0.0.2"
    DESCRIPTION = "A toroidal grid navigation environment with dynamic reward respawning"
    
    def __init__(self, config: WorldConfig = None, grid_size: int = None):
        if config is None:
            config = WorldConfig(grid_size=grid_size if grid_size is not None else 100)
        elif grid_size is not None:
            config = config._replace(grid_size=grid_size)
        self.config = config
    
    def get_metadata(self) -> dict:
        """Get world metadata."""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "description": self.DESCRIPTION,
            "config": self.config._asdict()
        }
        
    def reset(self, key: random.PRNGKey) -> Tuple[WorldState, Observation]:
        """Reset the world to initial state."""
        key1, key2 = random.split(key)
        
        # Agent starts at center
        agent_pos = (self.config.grid_size // 2, self.config.grid_size // 2)
        
        # Generate reward positions (clustered for biological realism)
        reward_positions = self._generate_rewards(key1, agent_pos)
        reward_collected = jnp.zeros(self.config.n_rewards, dtype=bool)
        
        state = WorldState(
            agent_pos=agent_pos,
            reward_positions=reward_positions,
            reward_collected=reward_collected,
            total_reward=0.0,
            timestep=0
        )
        
        observation = self._get_observation(state)
        return state, observation
    
    def step(self, state: WorldState, action: int, key: random.PRNGKey = None) -> StepResult:
        """Take a step in the world."""
        # Update position based on action (0=up, 1=right, 2=down, 3=left)
        new_pos = self._move_agent(state.agent_pos, action)
        
        # Use provided key or generate deterministic key from timestep
        if key is None:
            respawn_key = random.PRNGKey(state.timestep)
        else:
            respawn_key = key
        
        # Check reward collection and respawn
        reward, new_collected, new_reward_positions = self._collect_rewards(
            new_pos, state.reward_positions, state.reward_collected, respawn_key
        )
        
        # Update state
        new_state = WorldState(
            agent_pos=new_pos,
            reward_positions=new_reward_positions,
            reward_collected=new_collected,
            total_reward=state.total_reward + reward,
            timestep=state.timestep + 1
        )
        
        # Get observation
        observation = self._get_observation(new_state)
        
        # Check if done (never done now since rewards respawn)
        done = new_state.timestep >= self.config.max_timesteps
        
        return StepResult(
            state=new_state,
            observation=observation,
            reward=reward,
            done=done
        )
    
    def _generate_rewards(self, key: random.PRNGKey, agent_pos: Tuple[int, int]) -> Array:
        """Generate clustered reward positions avoiding duplicates and agent proximity."""
        min_agent_distance = 3  # Minimum distance from agent
        agent_array = jnp.array(agent_pos)
        
        # We'll generate more rewards than needed and filter
        key1, key2 = random.split(key)
        n_candidates = self.config.n_rewards * 3  # Generate extra candidates
        
        # For small numbers of rewards, use simpler approach
        if self.config.n_rewards < 10:
            # Generate random positions
            positions = random.randint(
                key1,
                shape=(n_candidates, 2),
                minval=0,
                maxval=self.config.grid_size
            )
        else:
            # Clustered generation
            n_clusters = min(10, self.config.n_rewards // 3)
            candidates_per_cluster = n_candidates // n_clusters
            
            # Generate cluster centers
            cluster_centers = random.uniform(
                key1, 
                shape=(n_clusters, 2),
                minval=0,
                maxval=self.config.grid_size
            ).astype(jnp.int32)
            
            # Generate candidates around clusters
            all_candidates = []
            keys = random.split(key2, n_clusters)
            
            for i in range(n_clusters):
                offsets = random.normal(keys[i], shape=(candidates_per_cluster, 2)) * 5
                cluster_positions = cluster_centers[i] + offsets.astype(jnp.int32)
                cluster_positions = jnp.mod(cluster_positions, self.config.grid_size)
                all_candidates.append(cluster_positions)
            
            positions = jnp.concatenate(all_candidates, axis=0)
        
        # Filter positions that are too close to agent
        distances_to_agent = self._calculate_distances(agent_array, positions)
        valid_mask = distances_to_agent >= min_agent_distance
        valid_positions = positions[valid_mask]
        
        # Remove duplicates by converting to set of tuples and back
        # This is done in numpy/python since JAX doesn't have good set operations
        unique_positions = []
        seen = set()
        
        for i in range(len(valid_positions)):
            pos_tuple = (int(valid_positions[i, 0]), int(valid_positions[i, 1]))
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(valid_positions[i])
                if len(unique_positions) >= self.config.n_rewards:
                    break
        
        # If we don't have enough unique positions, generate more deterministically
        if len(unique_positions) < self.config.n_rewards:
            # Create a grid of positions and filter
            grid_positions = []
            for x in range(0, self.config.grid_size, 2):  # Step by 2 for efficiency
                for y in range(0, self.config.grid_size, 2):
                    pos = jnp.array([x, y])
                    dist = jnp.linalg.norm(pos - agent_array)
                    if dist >= min_agent_distance:
                        pos_tuple = (x, y)
                        if pos_tuple not in seen:
                            grid_positions.append(pos)
                            seen.add(pos_tuple)
                            if len(unique_positions) + len(grid_positions) >= self.config.n_rewards:
                                break
                if len(unique_positions) + len(grid_positions) >= self.config.n_rewards:
                    break
            
            unique_positions.extend(grid_positions[:self.config.n_rewards - len(unique_positions)])
        
        # Convert back to jax array
        final_positions = jnp.array(unique_positions[:self.config.n_rewards])
        
        return final_positions
    
    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Move agent based on action."""
        x, y = pos
        
        # Action mapping
        dx = jnp.array([0, 1, 0, -1])[action]
        dy = jnp.array([-1, 0, 1, 0])[action]
        
        new_x = x + dx
        new_y = y + dy
        
        if self.config.toroidal:
            # Wrap around edges
            new_x = new_x % self.config.grid_size
            new_y = new_y % self.config.grid_size
        else:
            # Clamp to boundaries
            new_x = jnp.clip(new_x, 0, self.config.grid_size - 1)
            new_y = jnp.clip(new_y, 0, self.config.grid_size - 1)
        
        return (new_x, new_y)
    
    def _collect_rewards(
        self, 
        agent_pos: Tuple[int, int],
        reward_positions: Array,
        reward_collected: Array,
        key: random.PRNGKey
    ) -> Tuple[float, Array, Array]:
        """Check if agent collected any rewards and respawn them at farthest position."""
        # Calculate distances to all rewards
        agent_array = jnp.array(agent_pos)
        distances = self._calculate_distances(agent_array, reward_positions)
        
        # Check which rewards are at agent position
        at_reward = (distances < 0.5) & ~reward_collected
        
        # Calculate reward
        reward = jnp.sum(at_reward.astype(jnp.float32)) * self.config.reward_value
        
        # Add proximity reward for being near uncollected rewards
        near_reward = (distances < 5.0) & ~reward_collected & ~at_reward
        proximity_reward = jnp.sum(near_reward.astype(jnp.float32)) * self.config.proximity_reward
        
        total_reward = reward + proximity_reward
        
        # Update collected status
        new_collected = reward_collected | at_reward
        
        # If any rewards were collected, respawn them at farthest position
        # Use JAX-compatible conditional logic
        def respawn_rewards(positions, collected):
            # Find farthest position for first collected reward
            farthest_pos = self._find_farthest_position(agent_pos, positions, collected)
            
            # Update first collected reward position
            first_collected_idx = jnp.argmax(at_reward)
            new_positions = positions.at[first_collected_idx].set(farthest_pos)
            new_collected_status = collected.at[first_collected_idx].set(False)
            
            return new_positions, new_collected_status
        
        # Respawn if rewards were collected
        any_collected = jnp.any(at_reward)
        
        new_reward_positions, new_collected = jax.lax.cond(
            any_collected,
            respawn_rewards,
            lambda p, c: (p, c),
            reward_positions, new_collected
        )
        
        return total_reward, new_collected, new_reward_positions
    
    def _get_observation(self, state: WorldState) -> Observation:
        """Generate observation for the agent."""
        # Calculate gradient to nearest uncollected reward
        agent_array = jnp.array(state.agent_pos)
        uncollected_mask = ~state.reward_collected
        
        # Calculate distances to all rewards
        distances = self._calculate_distances(agent_array, state.reward_positions)
        
        # Mask out collected rewards by setting their distances to infinity
        masked_distances = jnp.where(uncollected_mask, distances, jnp.inf)
        
        # Find minimum distance
        min_distance = jnp.min(masked_distances)
        
        # Calculate decay constant so gradient reaches ~0.01 at half grid size
        # exp(-distance/decay) = 0.01 when distance = grid_size/2
        # -distance/decay = ln(0.01) ≈ -4.605
        # decay = distance / 4.605 = (grid_size/2) / 4.605
        decay_constant = self.config.grid_size / (2 * 4.605)
        
        # Convert to gradient signal (handle case where all rewards collected)
        gradient = jnp.where(
            jnp.isfinite(min_distance),
            jnp.exp(-min_distance / decay_constant),
            0.0
        )
        
        return Observation(gradient=gradient)
    
    def _calculate_distances(self, pos: Array, positions: Array) -> Array:
        """Calculate toroidal distances from pos to all positions."""
        if self.config.toroidal:
            # Toroidal distance calculation
            diff = positions - pos
            wrapped_diff = jnp.minimum(
                jnp.abs(diff),
                self.config.grid_size - jnp.abs(diff)
            )
            distances = jnp.linalg.norm(wrapped_diff, axis=1)
        else:
            # Regular Euclidean distance
            distances = jnp.linalg.norm(positions - pos, axis=1)
        
        return distances
    
    def _find_farthest_position(self, agent_pos: Tuple[int, int], reward_positions: Array, reward_collected: Array) -> Array:
        """Find the grid position farthest from the agent."""
        agent_array = jnp.array(agent_pos)
        
        # Create a grid of candidate positions
        # Sample evenly across the grid for efficiency
        step_size = max(1, self.config.grid_size // 20)
        x_coords = jnp.arange(0, self.config.grid_size, step_size)
        y_coords = jnp.arange(0, self.config.grid_size, step_size)
        
        # Create meshgrid
        xx, yy = jnp.meshgrid(x_coords, y_coords)
        candidate_positions = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Calculate distances from agent
        distances_from_agent = self._calculate_distances(agent_array, candidate_positions)
        
        # Calculate minimum distance to each uncollected reward using JAX-compatible operations
        # Expand dimensions for broadcasting
        candidate_positions_expanded = candidate_positions[:, None, :]  # Shape: (n_candidates, 1, 2)
        reward_positions_expanded = reward_positions[None, :, :]  # Shape: (1, n_rewards, 2)
        
        # Calculate all pairwise distances
        if self.config.toroidal:
            diff = candidate_positions_expanded - reward_positions_expanded
            wrapped_diff = jnp.minimum(
                jnp.abs(diff),
                self.config.grid_size - jnp.abs(diff)
            )
            pairwise_distances = jnp.linalg.norm(wrapped_diff, axis=2)
        else:
            pairwise_distances = jnp.linalg.norm(
                candidate_positions_expanded - reward_positions_expanded, axis=2
            )
        
        # Mask collected rewards by setting their distances to infinity
        masked_distances = jnp.where(
            reward_collected[None, :],
            jnp.inf,
            pairwise_distances
        )
        
        # Get minimum distance to any uncollected reward for each candidate
        min_distances_to_rewards = jnp.min(masked_distances, axis=1)
        
        # Find position that maximizes distance from agent while maintaining some distance from other rewards
        # Weight agent distance more heavily
        scores = distances_from_agent + 0.2 * jnp.where(
            jnp.isfinite(min_distances_to_rewards),
            min_distances_to_rewards,
            0.0
        )
        
        # Get the position with highest score
        best_idx = jnp.argmax(scores)
        return candidate_positions[best_idx]