"""Action executor for converting RL actions to creature behavior."""

from __future__ import annotations

import numpy as np
import math
from typing import Dict, Tuple
import pymunk


class ActionExecutor:
    """Converts RL action vectors to creature behaviors."""
    
    def __init__(self):
        """Initialize action executor."""
        # Action space definition (12 continuous actions)
        self.action_space = {
            'direction_x': (-1.0, 1.0),
            'direction_y': (-1.0, 1.0),
            'speed_ratio': (0.0, 1.0),
            'angular_velocity': (-1.0, 1.0),
            'exploration_weight': (0.0, 1.0),
            'resource_weight': (0.0, 1.0),
            'safety_weight': (0.0, 1.0),
            'social_weight': (0.0, 1.0),
            'patience': (0.0, 1.0),
            'risk_taking': (0.0, 1.0),
            'competitiveness': (0.0, 1.0),
            'adaptation_rate': (0.0, 1.0),
        }
        self.action_dim = len(self.action_space)
    
    def execute(self, creature, action: np.ndarray, dt: float) -> Dict:
        """Execute action for creature.
        
        Args:
            creature: Creature object
            action: Action array (12 dim)
            dt: Time delta
            
        Returns:
            Dictionary of executed action info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Extract action components
        direction_x = action[0]
        direction_y = action[1]
        speed_ratio = (action[2] + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        angular_velocity = action[3]
        
        # Extract behavior weights
        exploration_weight = (action[4] + 1.0) / 2.0
        resource_weight = (action[5] + 1.0) / 2.0
        safety_weight = (action[6] + 1.0) / 2.0
        social_weight = (action[7] + 1.0) / 2.0
        patience = (action[8] + 1.0) / 2.0
        risk_taking = (action[9] + 1.0) / 2.0
        competitiveness = (action[10] + 1.0) / 2.0
        adaptation_rate = (action[11] + 1.0) / 2.0
        
        # Normalize direction vector
        direction_norm = math.sqrt(direction_x**2 + direction_y**2)
        if direction_norm < 0.01:
            # Default direction (forward)
            direction_x, direction_y = 1.0, 0.0
            direction_norm = 1.0
        
        direction_vec = pymunk.Vec2d(direction_x / direction_norm, direction_y / direction_norm)
        
        # Compute desired speed (constrained by gene max_speed)
        desired_speed = speed_ratio * creature.genome.max_speed
        
        # Apply behavior weights to compute final movement
        # This modifies the desired direction based on behavior priorities
        # In full implementation, would blend with environment signals
        
        # For now, directly apply the direction
        creature._desired_dir = direction_vec
        
        # Apply angular velocity (rotation)
        if abs(angular_velocity) > 0.01:
            angle_change = angular_velocity * creature.genome.angular_speed * dt
            creature._desired_dir = creature._desired_dir.rotated(angle_change)
        
        # Compute desired velocity
        desired_velocity = creature._desired_dir * desired_speed
        
        # Apply steering force
        current_velocity = creature.body.velocity
        steer = desired_velocity - current_velocity
        
        # Compute drive force (based on aggression and competitiveness)
        genome_dict = creature.genome.to_dict()
        aggression = creature.genome.aggression
        base_drive = 90.0 + 240.0 * aggression
        
        # Boost drive based on competitiveness
        drive_multiplier = 1.0 + 0.3 * competitiveness
        drive = base_drive * drive_multiplier
        
        # Apply impulse
        impulse = steer * (creature.body.mass * dt * (drive / max(1.0, desired_speed)))
        creature.body.apply_impulse_at_local_point(impulse)
        
        # Soft cap speed
        v = creature.body.velocity
        speed = v.length
        if speed > desired_speed:
            creature.body.velocity = v * (desired_speed / speed)
        
        # Store behavior weights for reward calculation
        creature._rl_behavior_weights = {
            'exploration': exploration_weight,
            'resource': resource_weight,
            'safety': safety_weight,
            'social': social_weight,
            'patience': patience,
            'risk_taking': risk_taking,
            'competitiveness': competitiveness,
            'adaptation_rate': adaptation_rate,
        }
        
        return {
            'direction': direction_vec,
            'speed': desired_speed,
            'behavior_weights': creature._rl_behavior_weights,
        }
    
    def get_action_space_size(self) -> int:
        """Get action space dimension."""
        return self.action_dim


