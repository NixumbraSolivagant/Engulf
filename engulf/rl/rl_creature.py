"""RL-enabled Creature class."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List
import random

from ..creatures import Creature, Genome
from .state_encoder import StateEncoder
from .action_executor import ActionExecutor
from .reward_shaping import PersonalizedReward, RewardShaping
from .global_manager import GlobalRLManager
from .ppo import PPOAgent


class RLCreature(Creature):
    """Creature with RL decision-making capability.
    
    All behavior is controlled by RL policy, with genes providing
    initial biases and physical constraints.
    """
    
    def __init__(self, space, position: Tuple[float, float],
                 genome: Optional[Genome] = None,
                 global_manager: Optional[GlobalRLManager] = None,
                 use_rl: bool = True):
        """Initialize RL creature.
        
        Args:
            space: Pymunk physics space
            position: Initial position
            genome: Genome (None for random)
            global_manager: Global RL manager (None = local agent)
            use_rl: Whether to use RL (if False, falls back to gene-driven)
        """
        # Initialize base creature
        super().__init__(space, position, genome)
        
        self.use_rl = use_rl
        self.global_manager = global_manager
        
        if use_rl:
            # State encoder
            self.state_encoder = StateEncoder()
            
            # Action executor
            self.action_executor = ActionExecutor()
            
            # Personalized reward
            self.reward_shaper = PersonalizedReward(self.genome.to_dict())
            
            # Local policy (synced from global)
            if global_manager:
                # Use global manager's agent
                self.local_agent = None  # Will use global manager
            else:
                # Create local agent
                state_dim = 210  # Will be updated after first encoding
                action_dim = self.action_executor.get_action_space_size()
                self.local_agent = PPOAgent(state_dim, action_dim)
            
            # Apply gene bias to policy
            if self.local_agent:
                self.local_agent.policy.apply_gene_bias(self.genome.to_dict())
            
            # RL state tracking
            self.last_state = None
            self.last_action = None
            self.last_action_log_prob = None
            self.last_value = None
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.last_position = None  # Track position for movement detection
            self.episode_info = {
                'resources_gained': 0,
                'hazards_encountered': 0,
                'hazards_avoided': 0,
                'offspring_count': 0,
                'growth_amount': 0.0,
                'gained_resource': False,  # Track resource gains
                'resource_value': 0.0,  # Track resource value
                'explored_new': False,  # Track exploration
                'near_resource': False,  # Track resource proximity
                'moved_significantly': False,  # Track movement
            }
            self._explored_terrains = set()  # Track explored terrain types
            
            # For state encoding
            self._nearby_creatures_cache = []
        
        # Override update method to use RL
        self._original_update = super().update
    
    def update(self, dt: float, scene=None, nearby_creatures: List = None):
        """Update creature with RL decision-making.
        
        Args:
            dt: Time delta
            scene: TopDownScene for state encoding
            nearby_creatures: List of nearby creatures
        """
        if not self.use_rl or self.dead or scene is None:
            # Fall back to original behavior
            self._original_update(dt)
            return
        
        # Update episode stats
        self.age += dt
        self.episode_steps += 1
        
        # Check lifespan
        if self.age >= self.genome.lifespan_secs:
            self.dead = True
            # Finalize episode (will be called in _cull_dead)
            return
        
        # Encode state
        state = self.state_encoder.encode(self, scene, nearby_creatures or [])
        
        # Select action
        if self.global_manager:
            # Use global manager's agent
            agent = self.global_manager.agent
        else:
            agent = self.local_agent
        
        action, log_prob = agent.select_action(state, deterministic=False)
        value = agent.get_value(state)
        
        # Execute action
        action_info = self.action_executor.execute(self, action, dt)
        
        # Store for experience collection
        if self.last_state is not None:
            # Compute reward (delayed by one step)
            reward = self._compute_reward(self.last_state, self.last_action, state, scene)
            
            # Collect experience
            if self.global_manager:
                self.global_manager.collect_experience(
                    self.last_state,
                    self.last_action,
                    reward,
                    state,
                    False,  # Not done yet
                    self.episode_info.copy()
                )
            
            self.episode_reward += reward
        
        # Update state tracking
        self.last_state = state
        self.last_action = action
        self.last_action_log_prob = log_prob
        self.last_value = value
        
        # Apply environment modifiers (from terrain)
        # This is still needed for physics effects
        if hasattr(self, '_env_cooldown_bonus_per_sec'):
            if self._env_cooldown_bonus_per_sec != 0.0:
                self._breed_cd -= dt * self._env_cooldown_bonus_per_sec
        
        # Update breeding cooldown
        self._breed_cd -= dt
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray,
                       next_state: np.ndarray, scene) -> float:
        """Compute reward for transition.
        
        Args:
            state: Previous state
            action: Action taken
            next_state: New state
            scene: Scene for context
            
        Returns:
            Reward value
        """
        # Compute base rewards from episode info
        base_rewards = RewardShaping.calculate_base_rewards(self.episode_info)
        
        # Add step-based rewards
        base_rewards['survival'] = 0.05  # Reduced base survival (match reward_shaping.py)
        
        # Movement reward - reward active movement
        if self.last_position is not None:
            current_pos = (self.body.position.x, self.body.position.y)
            movement_dist = ((current_pos[0] - self.last_position[0])**2 + 
                           (current_pos[1] - self.last_position[1])**2)**0.5
            if movement_dist > 5.0:  # Moved significantly (> 5 pixels)
                self.episode_info['moved_significantly'] = True
            self.last_position = current_pos
        else:
            self.last_position = (self.body.position.x, self.body.position.y)
        
        # Check for resource consumption
        if hasattr(self, '_last_body_radius'):
            if self.genome.body_radius > self._last_body_radius:
                base_rewards['growth'] = (self.genome.body_radius - self._last_body_radius) * 30.0
                self.episode_info['growth_amount'] += base_rewards['growth']
        self._last_body_radius = self.genome.body_radius
        
        # Apply personalized reward
        reward = self.reward_shaper.calculate(base_rewards)
        
        # Reset episode info for next step (will be updated by events)
        self.episode_info = {
            'resources_gained': 0,
            'hazards_encountered': 0,
            'hazards_avoided': 0,
            'offspring_count': 0,
            'growth_amount': self.episode_info['growth_amount'],
            'moved_significantly': False,  # Reset movement flag
            'near_resource': False,  # Reset resource proximity flag
        }
        
        return reward
    
    def _finalize_episode(self, scene, natural_death: bool = False):
        """Finalize episode and compute final reward.
        
        Args:
            scene: Scene for context
            natural_death: Whether death was natural
        """
        if self.last_state is None:
            return
        
        # Compute final rewards
        episode_stats = {
            'actual_lifespan': self.age,
            'genetic_lifespan': self.genome.lifespan_secs,
            'offspring_count': self.episode_info.get('offspring_count', 0),
            'offspring_survival_rate': 0.5,  # Would track in full implementation
            'total_resources_consumed': self.episode_info.get('resources_gained', 0),
        }
        
        # Episodic reward
        episodic_reward = RewardShaping.compute_episodic_reward(episode_stats)
        
        # Add death penalty if unnatural
        if not natural_death:
            episodic_reward -= 50.0
        
        # Final experience
        if self.global_manager:
            self.global_manager.collect_experience(
                self.last_state,
                self.last_action,
                episodic_reward,
                self.last_state,  # Terminal state (same as last)
                True,  # Done
                episode_stats
            )
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_info = {
            'resources_gained': 0,
            'hazards_encountered': 0,
            'hazards_avoided': 0,
            'offspring_count': 0,
            'growth_amount': 0.0,
        }
        self.last_state = None
        self.last_action = None
    
    def update_network(self, params: Dict):
        """Update local network from global parameters.
        
        Args:
            params: Network parameters from global manager
        """
        if self.local_agent:
            self.local_agent.set_params(params)
            # Re-apply gene bias
            self.local_agent.policy.apply_gene_bias(self.genome.to_dict())
    
    def consume_resource(self):
        """Override to track resource consumption."""
        super().consume_resource()
        self.episode_info['resources_gained'] += 1
    
    def suffer_hazard(self, base_kill_probability: float = 0.5, terrain_type: str = "default"):
        """Override to track hazard encounters."""
        old_dead = self.dead
        super().suffer_hazard(base_kill_probability, terrain_type)
        
        if not old_dead:
            if self.dead:
                self.episode_info['hazards_encountered'] += 1
            else:
                self.episode_info['hazards_avoided'] += 1

