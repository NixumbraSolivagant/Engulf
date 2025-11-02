"""Reward shaping and personalized reward calculation."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class PersonalizedReward:
    """Personalized reward calculator based on genes."""
    
    def __init__(self, genome_dict: Dict[str, float]):
        """Initialize personalized reward.
        
        Args:
            genome_dict: Dictionary of gene values
        """
        self.genome_dict = genome_dict
        
        # Compute reward weights based on genes
        # Increased exploration weight significantly to encourage exploration
        exploration_drive = genome_dict.get('exploration_drive', 0.5)
        curiosity = genome_dict.get('curiosity', 0.5)
        self.reward_weights = {
            'survival': 1.0,
            'resource': 0.5 + 0.5 * curiosity,
            'exploration': 0.8 + 1.2 * exploration_drive,  # Increased from 0.3-1.0 to 0.8-2.0 (much stronger!)
            'movement': 1.5,  # Increased from 1.0 to 1.5 (encourage active movement)
            'distance': 1.2,  # New: reward for traveling distance
            'reproduction': 0.5 + 0.5 * genome_dict.get('fertility', 0.5),
            'safety': 0.3 + 0.7 * genome_dict.get('caution', 0.5),
            'growth': 0.4 + 0.6 * genome_dict.get('metabolism', 0.5),
            'social': 0.2 + 0.8 * genome_dict.get('sociability', 0.5),
        }
    
    def calculate(self, base_rewards: Dict[str, float]) -> float:
        """Calculate personalized reward.
        
        Args:
            base_rewards: Dictionary of base reward components
            
        Returns:
            Personalized reward value
        """
        total_reward = 0.0
        
        for reward_type, base_value in base_rewards.items():
            weight = self.reward_weights.get(reward_type, 1.0)
            total_reward += base_value * weight
        
        return total_reward


class RewardShaping:
    """Reward shaping utilities."""
    
    @staticmethod
    def calculate_base_rewards(info: Dict) -> Dict[str, float]:
        """Calculate base reward components from info.
        
        Args:
            info: Dictionary containing event information
            
        Returns:
            Dictionary of reward components
        """
        rewards = {
            'survival': 0.05,  # Reduced base survival reward (to make other rewards more significant)
            'resource': 0.0,
            'exploration': 0.0,
            'reproduction': 0.0,
            'safety': 0.0,
            'growth': 0.0,
            'social': 0.0,
            'movement': 0.08,  # Increased movement reward (encourage exploration)
            'distance': 0.0,  # Distance traveled reward (calculated dynamically)
        }
        
        # Add distance-based exploration reward (reward for being far from spawn)
        # Reward based on current distance from spawn, not cumulative (to prevent explosion)
        distance = info.get('distance_traveled', 0.0)
        if distance > 100.0:  # Only reward if traveled significant distance
            # Scale reward with distance, but with diminishing returns
            # Uses square root to prevent explosion while still rewarding exploration
            rewards['distance'] = min(1.0 + (distance - 100.0) * 0.002, 3.0)  # 1.0-3.0 range
        
        # Resource reward - increased to make it easier to get positive feedback
        if info.get('gained_resource', False):
            resource_value = info.get('resource_value', 0.0)
            rewards['resource'] = resource_value * 20.0
        # Small bonus for being near resources (encourage resource seeking)
        elif info.get('near_resource', False):
            rewards['resource'] = 0.5  # Small continuous reward for proximity
        
        # Exploration reward - make it much more generous and frequent
        exploration_reward = 0.0
        
        # Priority 1: New terrain exploration (highest reward)
        if info.get('explored_new', False):
            exploration_reward = 15.0  # Increased from 5.0 to 15.0 (3x!)
        # Priority 2: Movement-based exploration (if not new terrain)
        elif info.get('moved_significantly', False):
            # Scale exploration reward with movement distance
            movement_dist = info.get('movement_distance', 5.0)
            # Base reward + bonus for longer distances
            exploration_reward = 0.8 + 0.3 * min(movement_dist / 50.0, 2.0)  # 0.8-1.4 reward range
        
        # Additional reward for speed (faster = more exploration) - always applies if moving fast
        if info.get('high_speed', False):
            exploration_reward += 0.5  # Bonus for fast movement
        
        rewards['exploration'] = exploration_reward
        
        # Reproduction reward
        if info.get('bred_successfully', False):
            offspring_quality = info.get('offspring_quality', 0.5)
            rewards['reproduction'] = 100.0 * (1.0 + offspring_quality)
        
        # Safety reward
        if info.get('avoided_hazard', False):
            rewards['safety'] = 10.0
        elif info.get('took_hazard', False):
            hazard_level = info.get('hazard_level', 0.0)
            rewards['safety'] = -20.0 * hazard_level
        
        # Growth reward
        if info.get('grew', False):
            growth_amount = info.get('growth_amount', 0.0)
            rewards['growth'] = growth_amount * 30.0
        
        # Social reward
        if info.get('social_interaction', False):
            rewards['social'] = 3.0
        
        # Death penalty
        if info.get('died', False):
            if info.get('natural_death', False):
                rewards['survival'] = 0.0  # No penalty for natural death
            else:
                rewards['survival'] = -100.0  # Large penalty for unnatural death
        
        return rewards
    
    @staticmethod
    def compute_episodic_reward(episode_stats: Dict) -> float:
        """Compute episodic reward.
        
        Args:
            episode_stats: Dictionary of episode statistics
            
        Returns:
            Episodic reward
        """
        reward = 0.0
        
        # Lifespan reward
        lifespan_ratio = episode_stats.get('actual_lifespan', 0.0) / max(1.0, episode_stats.get('genetic_lifespan', 1.0))
        reward += lifespan_ratio * 30.0
        
        # Reproduction reward
        offspring_count = episode_stats.get('offspring_count', 0)
        offspring_survival = episode_stats.get('offspring_survival_rate', 0.0)
        reward += offspring_count * offspring_survival * 20.0
        
        # Resource accumulation
        total_resources = episode_stats.get('total_resources_consumed', 0.0)
        reward += total_resources * 0.5
        
        return reward

