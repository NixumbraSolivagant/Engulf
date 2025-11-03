"""RL-enabled Creature class."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List
import random

from ..creatures import Creature, Genome
from .state_encoder import StateEncoder
from .action_executor import ActionExecutor
from .reward_shaping import PersonalizedReward, RewardShaping
from .ppo import PPOAgent
from .evolution_supervisor import EvolutionSupervisor, FitnessMetrics
from .experience_buffer import ExperienceBuffer, Experience


class RLCreature(Creature):
    """Creature with RL decision-making capability.
    
    All behavior is controlled by RL policy, with genes providing
    initial biases and physical constraints.
    """
    
    def __init__(self, space, position: Tuple[float, float],
                 genome: Optional[Genome] = None,
                 global_manager: Optional = None,  # Kept for compatibility, but not used
                 use_rl: bool = True):
        """Initialize RL creature.
        
        Args:
            space: Pymunk physics space
            position: Initial position
            genome: Genome (None for random)
            global_manager: Deprecated - kept for compatibility only
            use_rl: Whether to use RL (if False, falls back to gene-driven)
        """
        # Initialize base creature
        super().__init__(space, position, genome)
        
        self.use_rl = use_rl
        
        if use_rl:
            # State encoder
            self.state_encoder = StateEncoder()
            
            # Action executor
            self.action_executor = ActionExecutor()
            
            # Personalized reward
            self.reward_shaper = PersonalizedReward(self.genome.to_dict())
            
            # Create local agent (each creature has its own)
            state_dim = 210  # Will be updated after first encoding
            action_dim = self.action_executor.get_action_space_size()
            self.local_agent = PPOAgent(state_dim, action_dim)
            
            # Apply gene bias to policy
            self.local_agent.policy.apply_gene_bias(self.genome.to_dict())
            
            # Individual experience buffer (each creature learns independently)
            buffer_capacity = 5000  # Smaller buffer per creature
            self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)
            self.total_experiences = 0
            self.train_counter = 0
            self.train_frequency = 50  # Train every 50 experiences
            
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
                'movement_distance': 0.0,  # Track movement distance for exploration reward
                'high_speed': False,  # Track high-speed movement
                'distance_traveled': 0.0,  # Cumulative distance from spawn
            }
            self._explored_terrains = set()  # Track explored terrain types
            self._spawn_position = position  # Track spawn position for distance calculation
            self._cumulative_distance = 0.0  # Track total distance traveled
            self._last_distance_reward = 0.0  # Track last distance reward value
            
            # Evolution & fitness tracking
            self._fitness_metrics = FitnessMetrics()
            self._total_lifetime_reward = 0.0  # Cumulative reward over lifetime
            self._birth_time = 0.0  # Will be set when created
            self._unique_id = id(self)  # Unique identifier for fitness tracking
            
            # Individual learning rate (based on gene)
            gene_lr = self.genome.to_dict().get('learning_rate', 0.5)
            # Scale learning rate: 0.0 gene = 0.5x base, 1.0 gene = 1.5x base
            self._individual_lr_multiplier = 0.5 + gene_lr  # Range: 0.5-1.5
            
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
        
        # Select action using local agent (each creature learns independently)
        action, log_prob = self.local_agent.select_action(state, deterministic=False)
        value = self.local_agent.get_value(state)
        
        # Execute action
        action_info = self.action_executor.execute(self, action, dt)
        
        # Track movement and speed before computing reward
        if self.last_position is not None:
            current_pos = (self.body.position.x, self.body.position.y)
            movement_dist = ((current_pos[0] - self.last_position[0])**2 + 
                           (current_pos[1] - self.last_position[1])**2)**0.5
            
            if movement_dist > 5.0:  # Moved significantly (> 5 pixels)
                self.episode_info['moved_significantly'] = True
                self.episode_info['movement_distance'] = movement_dist
                
                # Track cumulative distance traveled
                self._cumulative_distance += movement_dist
                
                # Reward high-speed movement (encourages active exploration)
                speed = movement_dist / (dt if dt > 0 else 0.016)  # pixels per second
                if speed > 100.0:  # Fast movement (> 100 px/s)
                    self.episode_info['high_speed'] = True
                
                # Distance from spawn (for distance-based reward)
                if hasattr(self, '_spawn_position'):
                    dist_from_spawn = ((current_pos[0] - self._spawn_position[0])**2 +
                                     (current_pos[1] - self._spawn_position[1])**2)**0.5
                    self.episode_info['distance_traveled'] = dist_from_spawn
                else:
                    self.episode_info['distance_traveled'] = self._cumulative_distance
            
            self.last_position = current_pos
        else:
            self.last_position = (self.body.position.x, self.body.position.y)
        
        # Store for experience collection
        if self.last_state is not None:
            # Compute reward (delayed by one step)
            reward = self._compute_reward(self.last_state, self.last_action, state, scene)
            
            # Collect experience in local buffer (independent learning)
            experience = Experience(
                self.last_state,
                self.last_action,
                reward,
                state,
                False,  # Not done yet
                self.episode_info.copy()
            )
            self.experience_buffer.add(experience)
            self.total_experiences += 1
            
            # Local training (each creature trains independently)
            if len(self.experience_buffer) >= self.local_agent.config.batch_size and \
               self.total_experiences % self.train_frequency == 0:
                self._train_local_agent()
            
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
        
        # Movement tracking is now done in update() method before calling _compute_reward
        # So episode_info should already have movement_distance, high_speed, etc. set
        
        # Check for resource consumption
        if hasattr(self, '_last_body_radius'):
            if self.genome.body_radius > self._last_body_radius:
                base_rewards['growth'] = (self.genome.body_radius - self._last_body_radius) * 30.0
                self.episode_info['growth_amount'] += base_rewards['growth']
        self._last_body_radius = self.genome.body_radius
        
        # Add lifespan reward (encourages long-term survival)
        # Small bonus per step that increases with age
        lifespan_bonus = 0.01 * (1.0 + self.age / max(1.0, self.genome.lifespan_secs))
        base_rewards['survival'] += lifespan_bonus
        
        # Apply personalized reward
        reward = self.reward_shaper.calculate(base_rewards)
        
        # Update lifetime tracking
        self._total_lifetime_reward += reward
        self._fitness_metrics.total_reward = self._total_lifetime_reward
        self._fitness_metrics.lifespan = self.age
        self._fitness_metrics.exploration_score = len(self._explored_terrains)
        self._fitness_metrics.resource_score = self.episode_info.get('resources_gained', 0)
        self._fitness_metrics.offspring_count = self.episode_info.get('offspring_count', 0)
        
        # Reset episode info for next step (will be updated by events)
        # Keep cumulative stats, reset per-step flags
        self.episode_info = {
            'resources_gained': self.episode_info.get('resources_gained', 0),
            'hazards_encountered': 0,
            'hazards_avoided': 0,
            'offspring_count': self.episode_info.get('offspring_count', 0),
            'growth_amount': self.episode_info.get('growth_amount', 0.0),
            'moved_significantly': False,  # Reset movement flag
            'near_resource': False,  # Reset resource proximity flag
            'movement_distance': 0.0,  # Reset movement distance
            'high_speed': False,  # Reset speed flag
            'distance_traveled': self.episode_info.get('distance_traveled', 0.0),  # Keep cumulative
            'gained_resource': False,  # Reset resource gain flag
            'resource_value': 0.0,  # Reset resource value
            'explored_new': False,  # Reset exploration flag
        }
        
        return reward
    
    def get_fitness_metrics(self) -> FitnessMetrics:
        """Get current fitness metrics.
        
        Returns:
            FitnessMetrics object
        """
        # Update metrics before returning
        self._fitness_metrics.lifespan = self.age
        self._fitness_metrics.total_reward = self._total_lifetime_reward
        self._fitness_metrics.exploration_score = len(self._explored_terrains)
        self._fitness_metrics.resource_score = self.episode_info.get('resources_gained', 0)
        self._fitness_metrics.offspring_count = self.episode_info.get('offspring_count', 0)
        # Survival bonus = lifespan / expected_lifespan
        if self.genome.lifespan_secs > 0:
            self._fitness_metrics.survival_bonus = self.age / self.genome.lifespan_secs
        return self._fitness_metrics
    
    @property
    def id(self) -> int:
        """Get unique identifier for fitness tracking."""
        return self._unique_id
    
    @property
    def individual_learning_rate_multiplier(self) -> float:
        """Get individual learning rate multiplier (based on gene)."""
        return self._individual_lr_multiplier
    
    def _train_local_agent(self):
        """Train local agent using own experience buffer.
        
        Each creature trains independently from its own experiences.
        """
        if len(self.experience_buffer) < self.local_agent.config.batch_size:
            return
        
        # Sample batch from own buffer
        batch = self.experience_buffer.sample(self.local_agent.config.batch_size)
        
        # Apply individual learning rate multiplier (based on gene)
        original_lr = self.local_agent.config.learning_rate
        adjusted_lr = original_lr * self._individual_lr_multiplier
        
        # Temporarily adjust learning rate
        self.local_agent.config.learning_rate = adjusted_lr
        
        # Train agent
        try:
            stats = self.local_agent.update(batch)
            self.train_counter += 1
        finally:
            # Restore original learning rate
            self.local_agent.config.learning_rate = original_lr

        # Cache latest training stats for scene aggregation (optional keys)
        self._last_train_stats = stats or {}

    def get_recent_training_stats(self) -> Dict:
        """Return recent training stats for visualization aggregation.
        
        Keys may include: policy_loss, value_loss, entropy, clip_fraction.
        Returns empty dict if none yet.
        """
        return getattr(self, "_last_train_stats", {})
    
    def _finalize_episode(self, scene, natural_death: bool = False):
        """Finalize episode and compute final reward.
        
        Args:
            scene: Scene for context
            natural_death: Whether death was natural
        """
        if self.last_state is None:
            return
        
        # Update final fitness metrics
        final_metrics = self.get_fitness_metrics()
        
        # Add final experience (terminal state)
        if self.last_state is not None and self.last_action is not None:
            episodic_reward = self.episode_reward
            # Add final terminal experience
            terminal_experience = Experience(
                self.last_state,
                self.last_action,
                episodic_reward,
                self.last_state,  # Terminal state (same as last)
                True,  # Done
                self.episode_info.copy()
            )
            self.experience_buffer.add(terminal_experience)
            self.total_experiences += 1
            
            # Final training step
            if len(self.experience_buffer) >= self.local_agent.config.batch_size:
                self._train_local_agent()
        
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
        
        # Note: Final terminal experience was already added in _finalize_episode above
        # No need to add it again here
        
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

