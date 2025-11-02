"""Creature definitions with genetics, behavior, and physics."""

from __future__ import annotations

import math
import random
from typing import Tuple, List, Dict

import pymunk
from .genetics import (
    generate_random_genome,
    generate_base_species_genome,
    breed_genomes,
    GeneEffects,
    get_gene_def,
    is_same_species,
    are_different_species,
)
from .settings import WINDOW_WIDTH, WINDOW_HEIGHT


class Genome:
    """Genome containing all genetic traits of a creature (64 genes).
    
    Uses a dictionary-based system for flexible gene management.
    Provides property accessors for backward compatibility.
    """
    
    def __init__(self, gene_dict: Dict[str, float] | None = None):
        """Initialize genome from gene dictionary.
        
        Args:
            gene_dict: Dictionary mapping gene names to values.
                      If None, generates a random genome.
        """
        if gene_dict is None:
            gene_dict = generate_random_genome()
        self._genes: Dict[str, float] = gene_dict.copy()
    
    def __getitem__(self, key: str) -> float:
        """Get gene value by name."""
        return self._genes.get(key, 0.0)
    
    def __setitem__(self, key: str, value: float) -> None:
        """Set gene value by name."""
        self._genes[key] = value
    
    def get(self, key: str, default: float = 0.0) -> float:
        """Get gene value with default."""
        return self._genes.get(key, default)
    
    def to_dict(self) -> Dict[str, float]:
        """Get full gene dictionary."""
        return self._genes.copy()
    
    # Backward compatibility: property accessors for commonly used genes
    @property
    def species_id(self) -> int:
        return int(self._genes.get("species_id", 0))
    
    @species_id.setter
    def species_id(self, value: int) -> None:
        self._genes["species_id"] = float(value)
    
    @property
    def body_radius(self) -> float:
        return self._genes.get("body_radius", 15.0)
    
    @body_radius.setter
    def body_radius(self, value: float) -> None:
        self._genes["body_radius"] = value
    
    @property
    def hue(self) -> float:
        return self._genes.get("hue", 0.0)
    
    @hue.setter
    def hue(self, value: float) -> None:
        self._genes["hue"] = value
    
    @property
    def curiosity(self) -> float:
        return self._genes.get("curiosity", 0.5)
    
    @property
    def aggression(self) -> float:
        return self._genes.get("aggression", 0.4)
    
    @property
    def caution(self) -> float:
        return self._genes.get("caution", 0.5)
    
    @property
    def max_speed(self) -> float:
        return self._genes.get("max_speed", 165.0)
    
    @max_speed.setter
    def max_speed(self, value: float) -> None:
        self._genes["max_speed"] = value
    
    @property
    def angular_speed(self) -> float:
        return self._genes.get("angular_speed", 2.85)
    
    @property
    def lifespan_secs(self) -> float:
        return self._genes.get("lifespan_secs", 42.5)
    
    @property
    def metabolism(self) -> float:
        return self._genes.get("metabolism", 0.5)
    
    @property
    def speed_adaptability(self) -> float:
        return self._genes.get("speed_adaptability", 0.5)
    
    @property
    def fertility(self) -> float:
        return self._genes.get("fertility", 0.5)
    
    @property
    def resilience(self) -> float:
        return self._genes.get("resilience", 0.5)
    
    @property
    def beauty(self) -> float:
        return self._genes.get("beauty", 0.5)
    
    @staticmethod
    def random(num_species: int = 3) -> "Genome":
        """Generate a random genome.

        Args:
            num_species: Number of species to choose from

        Returns:
            Randomly generated Genome
        """
        gene_dict = generate_random_genome(num_species)
        return Genome(gene_dict)

    @staticmethod
    def breed(a: "Genome", b: "Genome") -> "Genome":
        """Breed two genomes to create offspring with mutations.

        Args:
            a: First parent genome
            b: Second parent genome (must have same species_id)

        Returns:
            New genome with mixed traits and mutations
        """
        parent_a_dict = a.to_dict()
        parent_b_dict = b.to_dict()
        offspring_dict = breed_genomes(parent_a_dict, parent_b_dict)
        return Genome(offspring_dict)


def hue_to_rgb(h: float) -> Tuple[int, int, int]:
    """Convert hue to RGB color.

    Args:
        h: Hue in degrees (0..360)

    Returns:
        RGB tuple (r, g, b) with saturation=0.6, value=0.95
    """
    s, v = 0.6, 0.95
    h = (h % 360.0) / 60.0
    i = int(h)
    f = h - i
    p = int(v * (1 - s) * 255)
    q = int(v * (1 - s * f) * 255)
    t = int(v * (1 - s * (1 - f)) * 255)
    v_ = int(v * 255)
    if i == 0:
        r, g, b = v_, t, p
    elif i == 1:
        r, g, b = q, v_, p
    elif i == 2:
        r, g, b = p, v_, t
    elif i == 3:
        r, g, b = p, q, v_
    elif i == 4:
        r, g, b = t, p, v_
    else:
        r, g, b = v_, p, q
    return r, g, b


class Creature:
    """A creature in the simulation with physics body, genome, and behavior."""

    def __init__(self, space: pymunk.Space, position: Tuple[float, float], genome: Genome | None = None):
        """Initialize a creature.

        Args:
            space: Pymunk physics space
            position: Initial position (x, y)
            genome: Genome to use (None for random)
        """
        self.genome = genome or Genome.random()
        mass = max(0.6, self.genome.body_radius * 0.15)
        moment = pymunk.moment_for_circle(mass, 0, self.genome.body_radius)
        body = pymunk.Body(mass, moment)
        body.position = position
        body.angular_velocity = 0
        shape = pymunk.Circle(body, self.genome.body_radius)
        shape.friction = 0.6
        shape.elasticity = 0.2
        # Color only used by debug draw; add alpha
        r, g, b = hue_to_rgb(self.genome.hue)
        shape.color = (r, g, b, 255)

        space.add(body, shape)
        self.body = body
        self.shape = shape

        # Internal drive timers
        self._dir_timer = 0.0
        self._desired_dir = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))
        self._affinity_vec = pymunk.Vec2d(0, 0)
        
        # Exploration state - for long-range movement
        self._exploration_target = None  # Optional far-away target position
        self._exploration_timer = 0.0  # Timer for changing exploration target
        self._last_position = pymunk.Vec2d(position[0], position[1])  # Track movement to detect stagnation
        self._stagnation_timer = 0.0  # Track if creature is stuck in one area

        # Breeding cooldown
        self._breed_cd = random.uniform(2.0, 5.0)

        # Life
        self.age = 0.0
        self.dead = False

        # Environment modifiers
        self._env_speed_mult = 1.0
        self._env_cooldown_bonus_per_sec = 0.0
        self._env_hazard_prob_per_sec = 0.0
        self._env_hazard_accum = 0.0
        
        # Gene effects system
        self._gene_effects = GeneEffects()
        
        # Terrain memory (for memory gene)
        self._terrain_memory: Dict[str, float] = {}  # terrain_type -> experience score

    def set_env_mods(self, speed_mult: float, cooldown_bonus_per_sec: float, hazard_prob_per_sec: float) -> None:
        """Set environment modifiers from terrain effects.

        Args:
            speed_mult: Speed multiplier (applied to max_speed)
            cooldown_bonus_per_sec: Breeding cooldown reduction per second
            hazard_prob_per_sec: Hazard probability per second (ambient)
        """
        self._env_speed_mult = speed_mult
        self._env_cooldown_bonus_per_sec = cooldown_bonus_per_sec
        self._env_hazard_prob_per_sec = hazard_prob_per_sec

    def set_affinity(self, vec: Tuple[float, float]) -> None:
        """Set steering bias vector from scene (resources, hazards, mates).

        Args:
            vec: Steering bias vector (x, y)
        """
        self._affinity_vec = pymunk.Vec2d(*vec)

    def ready_to_breed(self) -> bool:
        """Check if creature is ready to breed.

        Returns:
            True if creature is alive and breeding cooldown has expired
        """
        return (not self.dead) and self._breed_cd <= 0.0

    def reset_breed_cd(self) -> None:
        """Reset breeding cooldown to random value."""
        self._breed_cd = random.uniform(3.0, 6.0)

    def consume_resource(self) -> None:
        """Consume a resource: grow, gain speed, reduce breeding cooldown."""
        if self.dead:
            return
        genome_dict = self.genome.to_dict()
        
        # Growth magnitude influenced by metabolism and body_plasticity
        base_growth = random.uniform(0.8, 1.8)
        metabolism = self.genome.metabolism
        body_plasticity = genome_dict.get("body_plasticity", 0.5)
        growth_scale = 0.6 + 0.8 * metabolism + 0.3 * body_plasticity
        delta_r = base_growth * growth_scale
        old_r = self.genome.body_radius
        new_r = min(28.0, old_r + delta_r)
        if new_r > old_r:
            self.genome.body_radius = new_r
            # Update mass and moment roughly proportional to size
            self.body.mass = max(0.6, new_r * 0.15)
            self.body.moment = pymunk.moment_for_circle(self.body.mass, 0, new_r)
            self.shape.unsafe_set_radius(new_r)
        
        # Speed boost influenced by speed_adaptability and resource_conversion_efficiency
        # Optimized for better resource rewards
        base_boost = 15.0  # Increased from 10.0
        speed_adapt = self.genome.speed_adaptability
        conversion_eff = genome_dict.get("resource_conversion_efficiency", 0.5)
        boost = base_boost * (0.6 + 0.9 * speed_adapt + 0.3 * conversion_eff)  # Increased multipliers
        self.genome.max_speed = min(300.0, self.genome.max_speed + boost)  # Increased cap from 260 to 300
        
        # Cooldown reduction influenced by fertility (more effective)
        base_reduce = 5.5  # Increased from 3.5
        reduce = base_reduce * (0.6 + 0.9 * self.genome.fertility)  # Increased multipliers
        self._breed_cd = max(0.0, self._breed_cd - reduce)

    def suffer_hazard(self, base_kill_probability: float = 0.5, terrain_type: str = "default") -> None:
        """Apply hazard effect: death chance or penalties.

        Args:
            base_kill_probability: Base probability of death (modified by resilience)
            terrain_type: Type of terrain causing hazard (for resistance calculation)
        """
        if self.dead:
            return
        genome_dict = self.genome.to_dict()
        
        # Get environmental resistance based on terrain type
        env_resistance = self._gene_effects.get_environmental_resistance(genome_dict, terrain_type)
        
        # Effective kill probability reduced by resilience and environmental resistance
        resilience = self.genome.resilience
        immune_system = genome_dict.get("immune_system", 0.5)
        effective_p = base_kill_probability * (1.0 - 0.7 * resilience) * (1.0 - 0.3 * immune_system) * (1.0 - 0.2 * env_resistance)
        effective_p = max(0.0, min(1.0, effective_p))
        
        if random.random() < effective_p:
            self.dead = True
        else:
            # Non-lethal effect scaled with lack of resilience and recovery_rate
            recovery_rate = genome_dict.get("recovery_rate", 0.5)
            penalty_scale = (0.4 + 0.6 * (1.0 - resilience)) * (1.0 - 0.3 * recovery_rate)
            self.genome.max_speed = max(60.0, self.genome.max_speed - 6.0 * penalty_scale)
            self._breed_cd += 1.5 * penalty_scale

    def devour(self, other: "Creature") -> None:
        """Devour another creature: gain size, speed, and breeding benefits.

        Args:
            other: Creature to devour
        """
        if self.dead or other.dead:
            return
        # Gain size from other (fractional)
        gain = max(0.6, other.genome.body_radius * 0.35)
        new_r = min(32.0, self.genome.body_radius + gain)
        if new_r > self.genome.body_radius:
            self.genome.body_radius = new_r
            self.body.mass = max(0.6, new_r * 0.15)
            self.body.moment = pymunk.moment_for_circle(self.body.mass, 0, new_r)
            self.shape.unsafe_set_radius(new_r)
        # Slight speed bump and cooldown reduction
        self.genome.max_speed = min(260.0, self.genome.max_speed + 6.0)
        self._breed_cd = max(0.0, self._breed_cd - 4.0)

    def update(self, dt: float) -> None:
        """Update creature state: aging, environment effects, movement.

        Args:
            dt: Time delta since last update
        """
        if self.dead:
            return
        # Environment continuous effects
        if self._env_cooldown_bonus_per_sec != 0.0:
            self._breed_cd -= dt * self._env_cooldown_bonus_per_sec
        if self._env_hazard_prob_per_sec > 0.0:
            self._env_hazard_accum += dt * self._env_hazard_prob_per_sec
            while self._env_hazard_accum >= 1.0:
                self._env_hazard_accum -= 1.0
                self.suffer_hazard(0.15)  # small ambient hazard attempt

        self._breed_cd -= dt
        self.age += dt
        if self.age >= self.genome.lifespan_secs:
            self.dead = True
            return

        # Track movement for stagnation detection
        current_pos = pymunk.Vec2d(self.body.position.x, self.body.position.y)
        movement_distance = (current_pos - self._last_position).length
        self._last_position = current_pos
        
        # Update stagnation timer - if creature moves very little, increase timer
        if movement_distance < 30.0:  # Less than 30 pixels movement
            self._stagnation_timer += dt
        else:
            self._stagnation_timer = max(0.0, self._stagnation_timer - dt * 2.0)  # Reset faster
        
        # Get genome dict once for all gene lookups (performance optimization)
        genome_dict = self.genome.to_dict()
        curiosity = self.genome.curiosity
        exploration_drive = genome_dict.get("exploration_drive", 0.5)
        exploration_factor = 0.7 * curiosity + 0.3 * exploration_drive
        self._dir_timer -= dt
        self._exploration_timer -= dt
        
        # Long-range exploration: set a distant target if stagnant or timer expires
        if (self._stagnation_timer > 2.0 or self._exploration_timer <= 0.0 or 
            self._exploration_target is None or 
            (current_pos - self._exploration_target).length < 50.0):
            # Set new exploration target - far away from current position
            exploration_distance = 200.0 + 300.0 * exploration_drive  # 200-500 pixels away
            angle = random.uniform(0, 6.283)
            target_x = current_pos.x + exploration_distance * math.cos(angle)
            target_y = current_pos.y + exploration_distance * math.sin(angle)
            # Clamp to window bounds with margin
            target_x = max(50.0, min(WINDOW_WIDTH - 50.0, target_x))
            target_y = max(50.0, min(WINDOW_HEIGHT - 50.0, target_y))
            self._exploration_target = pymunk.Vec2d(target_x, target_y)
            # Set exploration timer based on exploration_drive
            self._exploration_timer = 3.0 + 5.0 * (1.0 - exploration_drive)  # 3-8 seconds
            self._stagnation_timer = 0.0  # Reset stagnation
        
        # Direction changes: less frequent but more significant
        base_interval = 1.5 - 0.8 * exploration_factor  # Longer intervals for more sustained movement
        if self._dir_timer <= 0:
            self._dir_timer = random.uniform(base_interval * 0.7, base_interval * 1.5)
            
            # Decide between exploration target vs random direction (higher priority for exploration)
            if self._exploration_target and random.random() < 0.75 + 0.2 * exploration_drive:
                # Move towards exploration target (long-range movement) - HIGH PRIORITY
                to_target = (self._exploration_target - current_pos)
                to_target_len = to_target.length
                if to_target_len > 0.0001:
                    to_target = to_target.normalized()
                else:
                    # Too close to target, use random direction instead
                    to_target = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))
                # Stronger blend towards exploration target (more decisive)
                blend = 0.6 + 0.3 * exploration_drive  # Increased from 0.4-0.8 to 0.6-0.9
                self._desired_dir = (self._desired_dir * (1.0 - blend) + to_target * blend)
                # Normalize after blend (always needed after vector addition)
                desired_dir_len = self._desired_dir.length
                if desired_dir_len > 0.0001:
                    self._desired_dir = self._desired_dir.normalized()
                else:
                    # Fallback: use random direction if blend resulted in zero vector
                    self._desired_dir = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))
            else:
                # Random direction change (but less frequent jitter)
                if random.random() < 0.4 + 0.4 * exploration_factor:  # More likely to change direction
                    # Significant direction change
                    angle_change = random.uniform(-1.5, 1.5)  # Larger angle change
                    self._desired_dir = self._desired_dir.rotated(angle_change)
                    # Ensure still normalized (rotated() preserves length, but be safe)
                    if self._desired_dir.length > 0.0001:
                        self._desired_dir = self._desired_dir.normalized()
                    else:
                        self._desired_dir = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))
                else:
                    # Complete random direction
                    self._desired_dir = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))

        # Balanced blend: exploration priority with affinity as modifier (not dominant)
        if self._affinity_vec.length > 0.0001:
            target_dir = self._affinity_vec.normalized()
            strategic_flex = genome_dict.get("strategic_flexibility", 0.5)
            
            # Reduced blend strength - affinity should modify but not dominate exploration
            # Base blend much lower to allow exploration to take priority
            affinity_strength = min(1.0, self._affinity_vec.length / 2.0)  # Normalize affinity strength
            base_blend = 0.25 + 0.20 * affinity_strength * (self.genome.curiosity * 0.5 + self.genome.caution * 0.5) * (1.0 + 0.2 * strategic_flex)
            
            # Only strongly blend when affinity is VERY strong (immediate danger/resources)
            if self._affinity_vec.length > 1.2:  # Much higher threshold
                base_blend *= 1.4
            elif self._affinity_vec.length < 0.3:  # Weak affinity - prioritize exploration
                base_blend *= 0.5  # Reduce blend when affinity is weak
            
            # Cap blend to allow exploration to always have significant influence
            base_blend = min(0.55, base_blend)  # Max 55% instead of 98%
            
            # Blend with current desired_dir (which includes exploration target)
            self._desired_dir = (self._desired_dir * (1.0 - base_blend) + target_dir * base_blend)
            # Normalize after blend (always needed after vector addition)
            desired_dir_len = self._desired_dir.length
            if desired_dir_len > 0.0001:
                self._desired_dir = self._desired_dir.normalized()
            # If length is too small, keep current direction (don't corrupt it)

        # Steer towards desired_dir with environment speed multiplier
        # Apply gene-based speed effects
        effective_speed_mult = self._gene_effects.get_effective_speed(genome_dict, self._env_speed_mult)
        effective_max_speed = self.genome.max_speed * max(0.3, min(2.0, effective_speed_mult))
        # Ensure desired_dir is normalized before use (safety check)
        desired_dir_len = self._desired_dir.length
        if desired_dir_len > 0.0001:
            desired_velocity = self._desired_dir.normalized() * effective_max_speed
        else:
            # Fallback: use current velocity direction if desired_dir is invalid
            current_vel_len = self.body.velocity.length
            if current_vel_len > 0.0001:
                desired_velocity = self.body.velocity.normalized() * effective_max_speed
            else:
                desired_velocity = pymunk.Vec2d(1, 0) * effective_max_speed
        steer = desired_velocity - self.body.velocity
        
        # Enhanced drive: much stronger base drive for higher activity
        # Consider stamina for sustained drive
        stamina = genome_dict.get("stamina", 0.5)
        agility = genome_dict.get("agility", 0.5)
        base_drive = (250.0 + 200.0 * self.genome.aggression + 150.0 * curiosity + 120.0 * agility) * (0.95 + 0.35 * stamina)  # Increased base drive
        
        # Boost drive when exploring (long-range movement)
        if self._exploration_target:
            dist_to_target = (self._exploration_target - current_pos).length
            if dist_to_target > 50.0:  # Still far from target
                exploration_boost = 1.0 + 0.5 * exploration_drive + 0.3 * (1.0 - min(1.0, dist_to_target / 500.0))  # Stronger when far
                base_drive *= exploration_boost
        
        # Boost drive when stagnant (force movement)
        if self._stagnation_timer > 1.5:
            base_drive *= 1.8  # Strong boost to break out of stagnation
        
        # Boost drive when strongly attracted/repelled
        if self._affinity_vec.length > 0.6:
            base_drive *= 1.6  # Increased from 1.4 to 1.6 for stronger response
        # Additional boost for exploration
        if exploration_drive > 0.6:
            base_drive *= 1.3  # Increased from 1.2
        drive = base_drive
        impulse = steer * (self.body.mass * dt * (drive / max(1.0, effective_max_speed)))
        self.body.apply_impulse_at_local_point(impulse)

        # Soft cap speed
        v = self.body.velocity
        speed = v.length
        if speed > effective_max_speed:
            self.body.velocity = v * (effective_max_speed / speed)


def spawn_creatures(space: pymunk.Space, count: int, area: Tuple[int, int, int, int], 
                   num_species: int = 3, high_speed: bool = True) -> List[Creature]:
    """Spawn multiple creatures with same-species clustering.
    
    Creates creatures from `num_species` distinct base species with high initial speeds.
    Each species spawns in its own clustered area to start together.

    Args:
        space: Pymunk physics space
        count: Number of creatures to spawn
        area: Tuple of (x0, y0, x1, y1) bounding box
        num_species: Number of distinct initial species (default: 3)
        high_speed: Whether to give creatures high initial speed (default: True)

    Returns:
        List of created creatures
    """
    x0, y0, x1, y1 = area
    area_width = x1 - x0
    area_height = y1 - y0
    creatures: List[Creature] = []
    
    # Create base genomes for each species
    base_genomes = []
    for species_idx in range(num_species):
        base_genome = generate_base_species_genome(species_idx, high_speed=high_speed)
        base_genomes.append(base_genome)
    
    # Distribute creatures evenly across species
    creatures_per_species = count // num_species
    remainder = count % num_species
    
    # Calculate cluster radius - make species groups close but not overlapping
    # Use about 15-20% of area size for each cluster
    cluster_radius = min(area_width, area_height) * 0.15  # 15% of smaller dimension
    
    for species_idx in range(num_species):
        # Determine how many creatures for this species
        num_this_species = creatures_per_species
        if species_idx < remainder:
            num_this_species += 1
        
        # Choose a cluster center for this species
        # Distribute species clusters across the area
        if num_species == 3:
            # For 3 species, place them in left, center, and right regions
            if species_idx == 0:
                # Left region
                center_x = x0 + area_width * 0.25
                center_y = y0 + area_height * 0.5
            elif species_idx == 1:
                # Center region
                center_x = x0 + area_width * 0.5
                center_y = y0 + area_height * 0.5
            else:  # species_idx == 2
                # Right region
                center_x = x0 + area_width * 0.75
                center_y = y0 + area_height * 0.5
        else:
            # For other numbers, distribute evenly
            cluster_spacing = area_width / (num_species + 1)
            center_x = x0 + cluster_spacing * (species_idx + 1)
            center_y = y0 + area_height * 0.5
        
        # Add some random offset to cluster center for variety
        center_x += random.uniform(-cluster_radius * 0.3, cluster_radius * 0.3)
        center_y += random.uniform(-cluster_radius * 0.3, cluster_radius * 0.3)
        
        # Clamp center to area bounds
        center_x = max(x0 + cluster_radius, min(x1 - cluster_radius, center_x))
        center_y = max(y0 + cluster_radius, min(y1 - cluster_radius, center_y))
        
        base_genome = base_genomes[species_idx]
        
        # Spawn creatures in a cluster around the center
        for _ in range(num_this_species):
            # Generate position in a circle around cluster center
            # Use a more compact distribution (smaller radius for tighter clusters)
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, cluster_radius * 0.8)  # 80% of cluster radius for tighter packing
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Clamp to area bounds
            x = max(x0, min(x1, x))
            y = max(y0, min(y1, y))
            
            # Add small random variations to base genome
            varied_genome = base_genome.copy()
            for gene_name in varied_genome.keys():
                if gene_name == "species_id":
                    continue
                # Add small random variation (±5%)
                variation = random.uniform(-0.05, 0.05)
                gene_def = get_gene_def(gene_name)
                if gene_def:
                    min_val, max_val = gene_def.default_range
                    new_val = varied_genome[gene_name] * (1.0 + variation)
                    varied_genome[gene_name] = max(min_val, min(max_val, new_val))
            
            genome = Genome(varied_genome)
            creature = Creature(space, (x, y), genome)
            creatures.append(creature)
    
    return creatures


def spawn_child(space: pymunk.Space, a: Creature, b: Creature) -> Creature:
    """Spawn a child creature from two parents.

    Args:
        space: Pymunk physics space
        a: First parent creature
        b: Second parent creature (must be same species)

    Returns:
        New child creature with smaller initial size
    """
    g = Genome.breed(a.genome, b.genome)
    # Smaller initial body radius for child, will behave like smaller individual
    g.body_radius = max(8.0, g.body_radius * random.uniform(0.6, 0.85))
    mid = ((a.body.position.x + b.body.position.x) * 0.5, (a.body.position.y + b.body.position.y) * 0.5)
    child = Creature(space, mid, g)
    # Give a tiny separating impulse to avoid immediate repeated collisions
    child.body.apply_impulse_at_local_point((random.uniform(-30, 30), random.uniform(-30, 30)))
    return child
