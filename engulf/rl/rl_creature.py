"""RL-enabled Creature class."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List
import math
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
            # Tunables (kept small and localized for clarity)
            self._MIN_SIGNIFICANT_MOVE = 5.0
            self._HIGH_SPEED_THRESHOLD = 100.0  # px/s
            self._MAX_EPISODE_STEPS = 4000  # soft cap to avoid runaway episodes
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
            self.train_frequency = 50  # default; may override from scene.cfg at runtime
            self._train_freq_overridden = False
            
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
            # Intrinsic exploration (novelty) tracking
            self._visit_counts: Dict[Tuple[int, int], int] = {}
            self._visit_last_time: Dict[Tuple[int, int], float] = {}
            self._cell_size: float = 60.0
            # Resource accumulation (for diminishing returns)
            self._resource_total: float = 0.0
            # Hazard streak tracking
            self._consecutive_hazard_steps: int = 0
            # Learning: reward leaving danger after hazard
            self._took_hazard_last_step: bool = False
            # Resource diversity tracking
            self._last_resource_type: Optional[str] = None
            # Long safe-chain tracking
            self._safe_steps: int = 0
            
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
        
        # Episode step cap (soft): end episode if too long
        if self.episode_steps >= self._MAX_EPISODE_STEPS:
            self.dead = True
            return
        
        # Encode state
        state = self.state_encoder.encode(self, scene, nearby_creatures or [])
        
        # Select action using local agent (each creature learns independently)
        action, log_prob = self.local_agent.select_action(state, deterministic=False)
        value = self.local_agent.get_value(state)
        
        # Execute action
        action_info = self.action_executor.execute(self, action, dt)
        
        # Track movement and speed before computing reward
        # One-time override of train frequency and cell size from config if available
        if scene is not None and hasattr(scene, 'cfg') and self.use_rl and not self._train_freq_overridden:
            try:
                self.train_frequency = int(getattr(scene.cfg, 'rl_train_frequency', self.train_frequency))
            except Exception:
                pass
            try:
                self._cell_size = float(getattr(scene.cfg, 'rl_cell_size', self._cell_size))
            except Exception:
                pass
            self._train_freq_overridden = True

        self._update_movement_flags(dt)
        # Update intrinsic novelty signal based on coarse grid visitation with time decay
        if self.use_rl:
            cx = int(self.body.position.x // max(1.0, self._cell_size))
            cy = int(self.body.position.y // max(1.0, self._cell_size))
            key = (cx, cy)
            visits = self._visit_counts.get(key, 0)
            last_t = self._visit_last_time.get(key, None)
            time_since = (self.age - last_t) if (last_t is not None) else 1e9
            # novelty in (0,1], higher for first-time cells, recovers over time
            tau = getattr(scene.cfg, 'rl_novelty_time_const', 500.0) if scene is not None and hasattr(scene, 'cfg') else 500.0
            novelty = 1.0 / (1.0 + visits * math.exp(-time_since / max(1.0, tau)))
            coeff = getattr(scene.cfg, 'rl_novelty_coeff', 0.6) if scene is not None and hasattr(scene, 'cfg') else 0.6
            self.episode_info['novelty'] = float(novelty * coeff)
            self._visit_counts[key] = visits + 1
            self._visit_last_time[key] = self.age
            self.episode_info['visits_in_cell'] = visits + 1

            # Directional value: check forward cell (based on velocity direction)
            vel = self.body.velocity
            speed = (vel.x * vel.x + vel.y * vel.y) ** 0.5
            self.episode_info['forward_unvisited'] = False
            if speed > 5.0:
                ux, uy = vel.x / speed, vel.y / speed
                fx = self.body.position.x + ux * self._cell_size
                fy = self.body.position.y + uy * self._cell_size
                fkey = (int(fx // max(1.0, self._cell_size)), int(fy // max(1.0, self._cell_size)))
                if self._visit_counts.get(fkey, 0) == 0:
                    self.episode_info['forward_unvisited'] = True

            # Resource diversity: detect resource type switch (if near/gaining resource)
            try:
                if scene is not None:
                    terrain_type = scene._find_terrain_type_at(self.body.position.x, self.body.position.y)
                    self.episode_info['terrain_type'] = terrain_type
                    if self.episode_info.get('resource_value', 0.0) > 0.0 and terrain_type:
                        if self._last_resource_type is None:
                            self._last_resource_type = terrain_type
                        elif terrain_type != self._last_resource_type:
                            self.episode_info['resource_switched'] = True
                            self._last_resource_type = terrain_type
                        else:
                            self.episode_info['resource_switched'] = False
                    else:
                        self.episode_info['resource_switched'] = False
            except Exception:
                pass
        
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

    def _update_movement_flags(self, dt: float) -> None:
        """Update per-step movement-related episode flags.
        
        This keeps `update()` concise and centralizes thresholds.
        """
        current_pos = (self.body.position.x, self.body.position.y)
        if self.last_position is None:
            self.last_position = current_pos
            return
        
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        movement_dist = (dx * dx + dy * dy) ** 0.5
        
        if movement_dist > self._MIN_SIGNIFICANT_MOVE:
            self.episode_info['moved_significantly'] = True
            self.episode_info['movement_distance'] = movement_dist
            self._cumulative_distance += movement_dist
            
            # Speed in px/s (fallback dt to nominal frame delta)
            denom_dt = dt if dt and dt > 0 else 0.016
            speed = movement_dist / denom_dt
            if speed > self._HIGH_SPEED_THRESHOLD:
                self.episode_info['high_speed'] = True
            
            # Remove distance-from-spawn signal (disabled)
            self.episode_info['distance_traveled'] = 0.0
        
        self.last_position = current_pos
    
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
        
        # Add lifespan reward with exponential decay (avoid "idle survival")
        lifespan_bonus = 0.01 * (1.0 + self.age / max(1.0, self.genome.lifespan_secs))
        survival_with_bonus = base_rewards['survival'] + lifespan_bonus
        decay_base = getattr(scene.cfg, 'rl_survival_decay_factor', 0.6) if scene is not None and hasattr(scene, 'cfg') else 0.6
        decay = math.exp(- self.age / (decay_base * max(1.0, self.genome.lifespan_secs)))
        base_rewards['survival'] = survival_with_bonus * decay
        
        # Apply personalized reward
        # Update cumulative resource and expose to reward shaping for diminishing returns
        if self.episode_info.get('gained_resource', False):
            self._resource_total += float(self.episode_info.get('resource_value', 0.0))
        self.episode_info['resource_total'] = self._resource_total
        
        reward = self.reward_shaper.calculate(base_rewards)

        # Immediate hazard avoidance penalty (with nonlinear scaling for high hazards)
        hazard_level = float(self.episode_info.get('hazard_level', 0.0)) if isinstance(self.episode_info.get('hazard_level', 0.0), (int, float)) else 0.0
        if hazard_level > 0.0:
            # Continuous penalty scaled by hazard intensity
            scale = getattr(scene.cfg, 'rl_hazard_penalty_scale', 0.8) if scene is not None and hasattr(scene, 'cfg') else 0.8
            cap = getattr(scene.cfg, 'rl_hazard_penalty_cap', 3.0) if scene is not None and hasattr(scene, 'cfg') else 3.0
            # Nonlinear penalty: hazard^1.5 for stronger aversion to high hazards
            penalty = scale * min(cap, hazard_level ** 1.5)
            reward -= penalty
        if self.episode_info.get('took_hazard', False):
            # Discrete penalty when hazard event happened this step
            event_pen = getattr(scene.cfg, 'rl_hazard_event_penalty', 1.0) if scene is not None and hasattr(scene, 'cfg') else 1.0
            reward -= event_pen
            self._consecutive_hazard_steps += 1
            self._took_hazard_last_step = True
        else:
            self._consecutive_hazard_steps = 0
            # Learning bonus: left danger after hazard
            if self._took_hazard_last_step and hazard_level <= 0.02:
                leave_bonus = getattr(scene.cfg, 'rl_leave_danger_bonus', 1.0) if scene is not None and hasattr(scene, 'cfg') else 1.0
                reward += leave_bonus
                self._took_hazard_last_step = False
        streak_th = getattr(scene.cfg, 'rl_hazard_streak_threshold', 3) if scene is not None and hasattr(scene, 'cfg') else 3
        streak_pen = getattr(scene.cfg, 'rl_hazard_streak_penalty', 2.0) if scene is not None and hasattr(scene, 'cfg') else 2.0
        if self._consecutive_hazard_steps >= streak_th:
            reward -= streak_pen

        # Intrinsic curiosity bonus (novelty-driven)
        novelty = float(self.episode_info.get('novelty', 0.0))
        if novelty > 0.0:
            # Reward more for first-time cells, decays with visits
            reward += float(self.episode_info.get('novelty', 0.0))

        # Exploration refinements
        visits_in_cell = int(self.episode_info.get('visits_in_cell', 1))
        if visits_in_cell >= 2:
            rep_pen = getattr(scene.cfg, 'rl_repeat_visit_penalty', 0.2) if scene is not None and hasattr(scene, 'cfg') else 0.2
            reward -= rep_pen * visits_in_cell
        if self.episode_info.get('forward_unvisited', False):
            fwd_bonus = getattr(scene.cfg, 'rl_forward_unvisited_bonus', 1.0) if scene is not None and hasattr(scene, 'cfg') else 1.0
            reward += fwd_bonus
        if self.episode_info.get('resource_switched', False):
            switch_bonus = getattr(scene.cfg, 'rl_resource_switch_bonus', 3.0) if scene is not None and hasattr(scene, 'cfg') else 3.0
            reward += switch_bonus

        # Long safe-chain bonus
        if hazard_level <= 0.02:
            self._safe_steps += 1
        else:
            self._safe_steps = 0
        safe_th = getattr(scene.cfg, 'rl_safe_chain_steps', 200) if scene is not None and hasattr(scene, 'cfg') else 200
        if self._safe_steps >= safe_th:
            caution = float(self.genome.to_dict().get('caution', 0.5))
            safe_scale = getattr(scene.cfg, 'rl_safe_chain_bonus_scale', 3.0) if scene is not None and hasattr(scene, 'cfg') else 3.0
            reward += safe_scale * caution

        # Optional module: Territoriality (own spawn area bonus, others' spawn area penalty)
        try:
            territory_radius = 120.0
            if scene is not None and hasattr(scene, 'cfg'):
                territory_radius = float(scene.cfg.rl_territory_radius)
            sx, sy = self._spawn_position if hasattr(self, '_spawn_position') else (self.body.position.x, self.body.position.y)
            dx = self.body.position.x - sx
            dy = self.body.position.y - sy
            if (dx * dx + dy * dy) <= (territory_radius * territory_radius):
                bonus = getattr(scene.cfg, 'rl_territory_bonus', 1.5) if scene is not None and hasattr(scene, 'cfg') else 1.5
                reward += bonus
            # Penalty if inside another individual's spawn territory
            # Limit checks for efficiency
            checks = 0
            for other in (scene.creatures if scene is not None else []):
                if checks >= 15:
                    break
                if other is self or getattr(other, 'dead', False):
                    continue
                if not hasattr(other, '_spawn_position'):
                    continue
                osx, osy = other._spawn_position
                odx = self.body.position.x - osx
                ody = self.body.position.y - osy
                if (odx * odx + ody * ody) <= (territory_radius * territory_radius):
                    pen = getattr(scene.cfg, 'rl_intrude_penalty', 1.0) if scene is not None and hasattr(scene, 'cfg') else 1.0
                    reward -= pen
                    checks += 1
        except Exception:
            pass

        # Optional module: Energy metabolism (direct addition, not gene-weighted)
        # energy_cost proportional to movement distance; resource_intake from resource_value
        movement_dist = float(self.episode_info.get('movement_distance', 0.0))
        resource_intake = float(self.episode_info.get('resource_value', 0.0))
        e_cost = getattr(scene.cfg, 'rl_energy_cost_per_px', 0.02) if scene is not None and hasattr(scene, 'cfg') else 0.02
        e_gain = getattr(scene.cfg, 'rl_energy_intake_coeff', 0.5) if scene is not None and hasattr(scene, 'cfg') else 0.5
        energy_cost = e_cost * movement_dist
        reward += (-energy_cost + e_gain * resource_intake)

        # Optional module: Diversity (global, from scene)
        if scene is not None and hasattr(scene, '_current_diversity'):
            diversity = float(getattr(scene, '_current_diversity', 0.0))
            center = getattr(scene.cfg, 'rl_diversity_center', 0.6) if scene is not None and hasattr(scene, 'cfg') else 0.6
            scale = getattr(scene.cfg, 'rl_diversity_scale', 2.0) if scene is not None and hasattr(scene, 'cfg') else 2.0
            reward += scale * max(0.0, diversity - center)

        # Global reward scaling and clipping for stability
        clip_val = 2.0
        scale_val = 0.1
        if scene is not None and hasattr(scene, 'cfg'):
            try:
                clip_val = float(getattr(scene.cfg, 'rl_reward_clip', clip_val))
                scale_val = float(getattr(scene.cfg, 'rl_reward_scale', scale_val))
            except Exception:
                pass
        reward = max(-clip_val, min(clip_val, reward * scale_val))
        
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

