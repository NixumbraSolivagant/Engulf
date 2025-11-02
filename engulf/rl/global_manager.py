"""Global RL training manager for coordinating all RL agents."""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np

from .ppo import PPOAgent, PPOConfig
from .experience_buffer import ExperienceBuffer, Experience


class GlobalRLManager:
    """Global manager for RL training across all creatures."""
    
    def __init__(self, state_dim: int = 210, action_dim: int = 12,
                 config: Optional[PPOConfig] = None):
        """Initialize global RL manager.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: PPO configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        
        # Create central agent
        self.agent = PPOAgent(state_dim, action_dim, self.config)
        
        # Global experience buffer
        self.buffer = ExperienceBuffer(capacity=50000)  # Smaller for CPU
        
        # Training state
        self.update_counter = 0
        self.sync_frequency = 50  # Sync every 50 steps
        self.train_frequency = 100  # Train every 100 experiences
        self.total_experiences = 0
        
        # Statistics
        self.training_stats = {
            'total_updates': 0,
            'total_experiences': 0,
            'average_reward': [],
        }
    
    def collect_experience(self, state: np.ndarray, action: np.ndarray,
                          reward: float, next_state: np.ndarray,
                          done: bool, info: Optional[Dict] = None):
        """Collect experience from a creature.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Optional additional info
        """
        experience = Experience(state, action, reward, next_state, done, info)
        self.buffer.add(experience)
        self.total_experiences += 1
        
        # Periodic training
        if len(self.buffer) >= self.config.batch_size and \
           self.total_experiences % self.train_frequency == 0:
            self.train_step()
    
    def train_step(self) -> Optional[Dict]:
        """Perform one training step.
        
        Returns:
            Training statistics or None if not enough data
        """
        if len(self.buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        
        # Update agent
        stats = self.agent.update(batch)
        
        # Update statistics
        self.training_stats['total_updates'] += 1
        self.training_stats['total_experiences'] = len(self.buffer)
        
        # Track average reward
        if batch:
            avg_reward = np.mean([e.reward for e in batch])
            self.training_stats['average_reward'].append(avg_reward)
            # Keep only last 1000
            if len(self.training_stats['average_reward']) > 1000:
                self.training_stats['average_reward'] = \
                    self.training_stats['average_reward'][-1000:]
        
        return stats
    
    def sync_to_agents(self, agents: List) -> int:
        """Sync trained parameters to all agents.
        
        Args:
            agents: List of agent objects with update_network method
            
        Returns:
            Number of agents synced
        """
        params = self.agent.get_params()
        synced = 0
        
        for agent in agents:
            if hasattr(agent, 'update_network'):
                agent.update_network(params)
                synced += 1
        
        return synced
    
    def get_policy_params(self) -> Dict:
        """Get policy network parameters for syncing."""
        return self.agent.get_params()
    
    def save(self, filepath: str):
        """Save manager state."""
        self.agent.save(filepath)
    
    def load(self, filepath: str):
        """Load manager state."""
        self.agent.load(filepath)
    
    def get_statistics(self) -> Dict:
        """Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.training_stats.copy()
        
        # Add recent averages
        if self.training_stats['average_reward']:
            stats['recent_avg_reward'] = np.mean(
                self.training_stats['average_reward'][-100:]
            )
        
        stats['buffer_size'] = len(self.buffer)
        stats['total_experiences_collected'] = self.total_experiences
        
        return stats

