"""State encoder for converting environment state to RL state vector."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple
import math

from .utils import normalize, normalize_to_range


class StateEncoder:
    """Encodes creature and environment state into RL state vector."""
    
    def __init__(self, window_width: int = 1280, window_height: int = 720):
        """Initialize state encoder.
        
        Args:
            window_width: Window width for position normalization
            window_height: Window height for position normalization
        """
        self.window_width = window_width
        self.window_height = window_height
        
        # State dimensions (will be computed)
        self.state_dim = None
    
    def encode(self, creature, scene, nearby_creatures: List = None) -> np.ndarray:
        """Encode full state for creature.
        
        Args:
            creature: Creature object
            scene: TopDownScene object
            nearby_creatures: List of nearby creatures
            
        Returns:
            State vector array
        """
        state_parts = []
        
        # 1. Self state (25 dim)
        state_parts.append(self._encode_self(creature))
        
        # 2. Gene features (30 dim)
        state_parts.append(self._encode_genes(creature))
        
        # 3. Local terrain (60 dim - simplified)
        state_parts.append(self._encode_local_terrain(creature, scene))
        
        # 4. Nearby creatures (40 dim)
        state_parts.append(self._encode_nearby_creatures(creature, nearby_creatures or []))
        
        # 5. Global environment (15 dim)
        state_parts.append(self._encode_global_env(creature, scene))
        
        # 6. History (25 dim - simplified)
        state_parts.append(self._encode_history(creature))
        
        # 7. Computed features (15 dim)
        state_parts.append(self._encode_computed_features(creature, scene, nearby_creatures or []))
        
        # Concatenate all parts
        state = np.concatenate(state_parts, axis=0).astype(np.float32)
        self.state_dim = len(state)
        
        return state
    
    def _encode_self(self, creature) -> np.ndarray:
        """Encode self state (25 dim)."""
        body = creature.body
        px, py = body.position.x, body.position.y
        vx, vy = body.velocity.x, body.velocity.y
        
        features = [
            # Position (normalized)
            px / self.window_width,
            py / self.window_height,
            # Velocity
            vx / 300.0,  # Normalize by max expected speed
            vy / 300.0,
            # Speed ratio
            math.sqrt(vx**2 + vy**2) / max(1.0, creature.genome.max_speed),
            # Body radius (normalized)
            creature.genome.body_radius / 30.0,
            # Age ratio
            creature.age / max(1.0, creature.genome.lifespan_secs),
            # Breed cooldown ratio
            max(0.0, creature._breed_cd) / 10.0,
            # Health (simplified - no damage tracking yet)
            1.0 - creature.age / max(1.0, creature.genome.lifespan_secs * 2.0),
            # Species ID (one-hot, simplified to single value)
            float(creature.genome.species_id) / 10.0,
            # Species color features
            *self._hue_to_features(creature.genome.hue),
        ]
        
        # Fill to 25 dim
        while len(features) < 25:
            features.append(0.0)
        
        return np.array(features[:25], dtype=np.float32)
    
    def _encode_genes(self, creature) -> np.ndarray:
        """Encode gene features (30 dim)."""
        genome_dict = creature.genome.to_dict()
        
        # Personality genes (18)
        personality_genes = [
            genome_dict.get('curiosity', 0.5),
            genome_dict.get('aggression', 0.5),
            genome_dict.get('caution', 0.5),
            genome_dict.get('sociability', 0.5),
            genome_dict.get('territoriality', 0.5),
            genome_dict.get('strategic_flexibility', 0.5),
            genome_dict.get('predatory_instinct', 0.5),
            genome_dict.get('exploration_drive', 0.5),
            genome_dict.get('risk_tolerance', 0.5),
            genome_dict.get('hoarding_instinct', 0.5),
            genome_dict.get('migration_tendency', 0.5),
            genome_dict.get('competitiveness', 0.5),
            genome_dict.get('breeding_urge', 0.5),
            genome_dict.get('group_defense', 0.5),
            genome_dict.get('resource_competition_strategy', 0.5),
            genome_dict.get('group_size_preference', 0.5),
            genome_dict.get('fear_threshold', 0.5),
            genome_dict.get('stress_resistance', 0.5),
        ]
        
        # Ability genes (4)
        ability_genes = [
            normalize(genome_dict.get('max_speed', 165.0), 80.0, 240.0),
            normalize(genome_dict.get('angular_speed', 2.85), 0.8, 5.0),
            genome_dict.get('agility', 0.5),
            genome_dict.get('stamina', 0.5),
        ]
        
        # Adaptation genes (8)
        adaptation_genes = [
            genome_dict.get('environmental_adaptability', 0.5),
            genome_dict.get('cold_resistance', 0.5),
            genome_dict.get('heat_resistance', 0.5),
            genome_dict.get('aquatic_affinity', 0.5),
            genome_dict.get('nocturnality', 0.5),
            genome_dict.get('niche_breadth', 0.5),
            genome_dict.get('terrain_specialization', 0.5),
            genome_dict.get('toxin_resistance', 0.5),
        ]
        
        all_genes = personality_genes + ability_genes + adaptation_genes
        return np.array(all_genes[:30], dtype=np.float32)
    
    def _encode_local_terrain(self, creature, scene) -> np.ndarray:
        """Encode local terrain (60 dim - simplified)."""
        body = creature.body
        px, py = body.position.x, body.position.y
        
        # Sample terrain in 8 directions around creature
        terrain_features = []
        radius = 100.0  # Sampling radius
        
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            sample_x = px + radius * math.cos(angle)
            sample_y = py + radius * math.sin(angle)
            
            # Get terrain at sample point
            # Find terrain type by checking all terrain shapes
            terrain_type = None
            for ttype, shapes in scene.terrain_shapes.items():
                for shape in shapes:
                    if scene._point_in_terrain_shape(sample_x, sample_y, shape):
                        terrain_type = ttype
                        break
                if terrain_type:
                    break
            
            # Get terrain properties
            # Find the actual shape to get state
            terrain_shape = None
            if terrain_type:
                for shape in scene.terrain_shapes.get(terrain_type, []):
                    if scene._point_in_terrain_shape(sample_x, sample_y, shape):
                        terrain_shape = shape
                        break
            
            state = scene.terrain_dynamic_state.get(terrain_shape, {}) if terrain_shape else {}
            resource_val = state.get('current_resource', 0.0) if state else 0.0
            hazard_val = state.get('current_hazard', 0.0) if state else 0.0
            
            # Terrain type encoding (one-hot would be 14 dim, simplified)
            terrain_features.extend([
                resource_val,
                hazard_val,
                float(terrain_type is not None),
                normalize_to_range(sample_x, 0, self.window_width) if terrain_type else 0.5,
                normalize_to_range(sample_y, 0, self.window_height) if terrain_type else 0.5,
            ])
        
        # Current terrain
        current_terrain = None
        for ttype, shapes in scene.terrain_shapes.items():
            for shape in shapes:
                if scene._point_in_terrain_shape(px, py, shape):
                    current_terrain = shape  # Store shape for state lookup
                    break
            if current_terrain:
                break
        
        current_state = scene.terrain_dynamic_state.get(current_terrain, {}) if current_terrain else {}
        terrain_features.extend([
            current_state.get('current_resource', 0.0),
            current_state.get('current_hazard', 0.0),
            float(current_terrain is not None),
        ])
        
        # Pad to 60 dim
        terrain_features = terrain_features[:60]
        while len(terrain_features) < 60:
            terrain_features.append(0.0)
        
        return np.array(terrain_features[:60], dtype=np.float32)
    
    def _encode_nearby_creatures(self, creature, nearby_creatures: List) -> np.ndarray:
        """Encode nearby creatures (40 dim)."""
        body = creature.body
        px, py = body.position.x, body.position.y
        
        # Get 5 nearest creatures
        distances = []
        for other in nearby_creatures[:20]:  # Check up to 20
            if other.dead or other is creature:
                continue
            other_px, other_py = other.body.position.x, other.body.position.y
            dist = math.sqrt((px - other_px)**2 + (py - other_py)**2)
            distances.append((dist, other))
        
        distances.sort(key=lambda x: x[0])
        nearest = distances[:5]
        
        features = []
        for dist, other in nearest:
            other_px, other_py = other.body.position.x, other.body.position.y
            other_vx, other_vy = other.body.velocity.x, other.body.velocity.y
            
            features.extend([
                (other_px - px) / 500.0,  # Relative x
                (other_py - py) / 500.0,  # Relative y
                dist / 500.0,  # Distance
                other.genome.body_radius / creature.genome.body_radius,  # Size ratio
                float(other.genome.species_id == creature.genome.species_id),  # Same species
                math.sqrt(other_vx**2 + other_vy**2) / max(1.0, other.genome.max_speed),  # Speed ratio
                other.age / max(1.0, other.genome.lifespan_secs),  # Age ratio
                float(other.ready_to_breed()),  # Can breed
            ])
        
        # Pad to 40 dim (5 creatures * 8 dim = 40)
        while len(features) < 40:
            features.append(0.0)
        
        return np.array(features[:40], dtype=np.float32)
    
    def _encode_global_env(self, creature, scene) -> np.ndarray:
        """Encode global environment (15 dim)."""
        # Season state
        season = scene.current_season
        season_one_hot = [
            1.0 if season == 'growth' else 0.0,
            1.0 if season == 'danger' else 0.0,
            1.0 if season == 'normal' else 0.0,
        ]
        
        # Time of day (for meadow day/night cycle)
        time_of_day = 0.5  # Simplified
        
        # Population density (nearby creatures)
        nearby_count = sum(1 for c in scene.creatures 
                          if not c.dead and 
                          math.sqrt((c.body.position.x - creature.body.position.x)**2 + 
                                   (c.body.position.y - creature.body.position.y)**2) < 200)
        density = min(1.0, nearby_count / 10.0)
        
        # Resource/hazard availability (simplified)
        resource_availability = 0.5
        hazard_prevalence = 0.3
        
        # Terrain diversity (simplified)
        terrain_diversity = 0.5
        
        features = season_one_hot + [
            time_of_day,
            density,
            resource_availability,
            hazard_prevalence,
            terrain_diversity,
        ]
        
        # Pad to 15 dim
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15], dtype=np.float32)
    
    def _encode_history(self, creature) -> np.ndarray:
        """Encode history (25 dim - simplified)."""
        # Simplified: just use current state features
        # In full implementation, would track actual history
        
        features = [
            creature.age / 100.0,  # Normalized age
            float(creature.ready_to_breed()),  # Breeding readiness
            0.0,  # Recent reward sum (would track)
            0.0,  # Recent trajectory direction
            0.0,  # Resource trend
            0.0,  # Hazard trend
        ]
        
        # Pad to 25 dim
        while len(features) < 25:
            features.append(0.0)
        
        return np.array(features[:25], dtype=np.float32)
    
    def _encode_computed_features(self, creature, scene, nearby_creatures: List) -> np.ndarray:
        """Encode computed features (15 dim).
        
        Args:
            creature: Creature object
            scene: TopDownScene object
            nearby_creatures: List of nearby creatures
            
        Returns:
            Feature array
        """
        body = creature.body
        px, py = body.position.x, body.position.y
        
        # Distance to nearest resource-rich terrain
        best_resource_dist = 500.0  # Default large distance
        best_resource_val = 0.0
        
        # Distance to safest area
        safest_dist = 500.0
        
        # Distance to nearest mate
        nearest_mate_dist = 500.0
        if creature.ready_to_breed():
            for other in nearby_creatures:
                if other.dead or not other.ready_to_breed():
                    continue
                if other.genome.species_id != creature.genome.species_id:
                    continue
                dist = math.sqrt((other.body.position.x - px)**2 + 
                               (other.body.position.y - py)**2)
                if dist < nearest_mate_dist:
                    nearest_mate_dist = dist
        
        # Distance to nearest potential predator (larger creature)
        nearest_predator_dist = 500.0
        
        # Exploration value (simplified)
        exploration_value = 0.5
        
        # Safety score
        safety_score = 0.7  # Simplified
        
        # Resource attraction
        resource_attraction = 0.5
        
        # Hazard repulsion
        hazard_repulsion = 0.5
        
        # Group cohesion
        group_cohesion = 0.3
        
        # Breeding opportunity
        breeding_opportunity = 1.0 / max(1.0, nearest_mate_dist / 100.0) if nearest_mate_dist < 500 else 0.0
        
        features = [
            best_resource_dist / 500.0,
            safest_dist / 500.0,
            nearest_mate_dist / 500.0,
            nearest_predator_dist / 500.0,
            exploration_value,
            safety_score,
            resource_attraction,
            hazard_repulsion,
            group_cohesion,
            breeding_opportunity,
            0.5,  # energy_efficiency
            0.5,  # strategic_value
            0.3,  # competition_level
            0.5,  # evolutionary_pressure
            0.5,  # adaptation_potential
        ]
        
        return np.array(features[:15], dtype=np.float32)
    
    def _hue_to_features(self, hue: float) -> List[float]:
        """Convert hue to RGB-like features."""
        # Convert hue to normalized RGB
        h = (hue % 360.0) / 60.0
        i = int(h)
        f = h - i
        p = 0.0
        q = 1.0 - f
        t = f
        
        if i == 0:
            r, g, b = 1.0, t, p
        elif i == 1:
            r, g, b = q, 1.0, p
        elif i == 2:
            r, g, b = p, 1.0, t
        elif i == 3:
            r, g, b = p, q, 1.0
        elif i == 4:
            r, g, b = t, p, 1.0
        else:
            r, g, b = 1.0, p, q
        
        return [r, g, b]

