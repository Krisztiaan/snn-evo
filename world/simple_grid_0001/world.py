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
    
    def __init__(self, config: WorldConfig = WorldConfig()):
        self.config = config
        
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
    
    def step(self, state: WorldState, action: int) -> StepResult:
        """Take a step in the world."""
        # Update position based on action (0=up, 1=right, 2=down, 3=left)
        new_pos = self._move_agent(state.agent_pos, action)
        
        # Check reward collection
        reward, new_collected = self._collect_rewards(
            new_pos, state.reward_positions, state.reward_collected
        )
        
        # Update state
        new_state = WorldState(
            agent_pos=new_pos,
            reward_positions=state.reward_positions,
            reward_collected=new_collected,
            total_reward=state.total_reward + reward,
            timestep=state.timestep + 1
        )
        
        # Get observation
        observation = self._get_observation(new_state)
        
        # Check if done
        done = (new_state.timestep >= self.config.max_timesteps) | jnp.all(new_collected)
        
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
        reward_collected: Array
    ) -> Tuple[float, Array]:
        """Check if agent collected any rewards."""
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
        
        return total_reward, new_collected
    
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
        
        # Convert to gradient signal (handle case where all rewards collected)
        gradient = jnp.where(
            jnp.isfinite(min_distance),
            jnp.exp(-min_distance / 10.0),
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