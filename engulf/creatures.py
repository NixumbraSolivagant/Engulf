"""Creature definitions with genetics, behavior, and physics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple, List

import pymunk


@dataclass
class Genome:
    """Genome containing all genetic traits of a creature.
    
    Attributes:
        species_id: Species identifier (0..N-1)
        body_radius: Physical body radius (12..22)
        hue: Color hue in degrees (0..360)
        curiosity: Exploration tendency (0..1)
        aggression: Aggressive behavior (0..1)
        caution: Caution level (0..1)
        max_speed: Maximum speed in px/s (110..220)
        angular_speed: Angular velocity in rad/s (1.2..4.5)
        lifespan_secs: Lifespan in seconds (25..60)
        metabolism: Growth rate modifier (0..1)
        speed_adaptability: Speed boost modifier (0..1)
        fertility: Breeding cooldown modifier (0..1)
        resilience: Hazard resistance (0..1)
        beauty: Breeding success modifier (0..1)
    """
    # Species id (same id -> same species)
    species_id: int  # 0..N-1

    # Appearance
    body_radius: float  # 10..24
    hue: float          # 0..360 (used to colorize)

    # Personality (0..1)
    curiosity: float    # explore tendency
    aggression: float   # seek collisions / speed up
    caution: float      # avoid walls / slow near walls

    # Abilities
    max_speed: float    # px/s
    angular_speed: float  # rad/s

    # Life
    lifespan_secs: float  # seconds

    # Modifiers (0..1) indirectly controlling resource/hazard effects
    metabolism: float           # higher -> more growth per resource
    speed_adaptability: float   # higher -> larger speed boost per resource
    fertility: float            # higher -> more cooldown reduction per resource
    resilience: float           # higher -> lower hazard death probability
    beauty: float               # higher -> higher breeding success chance

    @staticmethod
    def random(num_species: int = 3) -> "Genome":
        """Generate a random genome.

        Args:
            num_species: Number of species to choose from

        Returns:
            Randomly generated Genome
        """
        hue_band = 360.0 / num_species
        sid = random.randrange(num_species)
        base_hue = sid * hue_band + hue_band * 0.5
        return Genome(
            species_id=sid,
            body_radius=random.uniform(12.0, 22.0),
            hue=(base_hue + random.uniform(-hue_band * 0.25, hue_band * 0.25)) % 360.0,
            curiosity=random.uniform(0.2, 0.9),
            aggression=random.uniform(0.0, 0.8),
            caution=random.uniform(0.1, 0.9),
            max_speed=random.uniform(110.0, 220.0),
            angular_speed=random.uniform(1.2, 4.5),
            lifespan_secs=random.uniform(25.0, 60.0),
            metabolism=random.uniform(0.0, 1.0),
            speed_adaptability=random.uniform(0.0, 1.0),
            fertility=random.uniform(0.0, 1.0),
            resilience=random.uniform(0.0, 1.0),
            beauty=random.uniform(0.0, 1.0),
        )

    @staticmethod
    def breed(a: "Genome", b: "Genome") -> "Genome":
        """Breed two genomes to create offspring with mutations.

        Args:
            a: First parent genome
            b: Second parent genome (must have same species_id)

        Returns:
            New genome with mixed traits and mutations
        """
        def mix(x: float, y: float, mut_scale: float, clamp_min: float | None = None, clamp_max: float | None = None):
            v = 0.5 * (x + y) + random.uniform(-1.0, 1.0) * mut_scale
            if clamp_min is not None:
                v = max(clamp_min, v)
            if clamp_max is not None:
                v = min(clamp_max, v)
            return v

        hue = (0.5 * (a.hue + b.hue) + random.uniform(-8.0, 8.0)) % 360.0
        body_radius = mix(a.body_radius, b.body_radius, mut_scale=1.0, clamp_min=8.0, clamp_max=24.0)
        curiosity = mix(a.curiosity, b.curiosity, mut_scale=0.05, clamp_min=0.0, clamp_max=1.0)
        aggression = mix(a.aggression, b.aggression, mut_scale=0.05, clamp_min=0.0, clamp_max=1.0)
        caution = mix(a.caution, b.caution, mut_scale=0.05, clamp_min=0.0, clamp_max=1.0)
        max_speed = mix(a.max_speed, b.max_speed, mut_scale=6.0, clamp_min=80.0, clamp_max=240.0)
        angular_speed = mix(a.angular_speed, b.angular_speed, mut_scale=0.2, clamp_min=0.8, clamp_max=5.0)
        lifespan_secs = mix(a.lifespan_secs, b.lifespan_secs, mut_scale=2.5, clamp_min=15.0, clamp_max=90.0)
        metabolism = mix(a.metabolism, b.metabolism, mut_scale=0.08, clamp_min=0.0, clamp_max=1.0)
        speed_adaptability = mix(a.speed_adaptability, b.speed_adaptability, mut_scale=0.08, clamp_min=0.0, clamp_max=1.0)
        fertility = mix(a.fertility, b.fertility, mut_scale=0.08, clamp_min=0.0, clamp_max=1.0)
        resilience = mix(a.resilience, b.resilience, mut_scale=0.08, clamp_min=0.0, clamp_max=1.0)
        beauty = mix(a.beauty, b.beauty, mut_scale=0.08, clamp_min=0.0, clamp_max=1.0)
        return Genome(
            species_id=a.species_id,
            body_radius=body_radius,
            hue=hue,
            curiosity=curiosity,
            aggression=aggression,
            caution=caution,
            max_speed=max_speed,
            angular_speed=angular_speed,
            lifespan_secs=lifespan_secs,
            metabolism=metabolism,
            speed_adaptability=speed_adaptability,
            fertility=fertility,
            resilience=resilience,
            beauty=beauty,
        )


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
        # Growth magnitude influenced by metabolism
        base_growth = random.uniform(0.8, 1.8)
        growth_scale = 0.6 + 0.8 * self.genome.metabolism
        delta_r = base_growth * growth_scale
        old_r = self.genome.body_radius
        new_r = min(28.0, old_r + delta_r)
        if new_r > old_r:
            self.genome.body_radius = new_r
            # Update mass and moment roughly proportional to size
            self.body.mass = max(0.6, new_r * 0.15)
            self.body.moment = pymunk.moment_for_circle(self.body.mass, 0, new_r)
            self.shape.unsafe_set_radius(new_r)
        # Speed boost influenced by speed_adaptability
        base_boost = 10.0
        boost = base_boost * (0.5 + 0.8 * self.genome.speed_adaptability)
        self.genome.max_speed = min(260.0, self.genome.max_speed + boost)
        # Cooldown reduction influenced by fertility
        base_reduce = 3.5
        reduce = base_reduce * (0.5 + 0.8 * self.genome.fertility)
        self._breed_cd = max(0.0, self._breed_cd - reduce)

    def suffer_hazard(self, base_kill_probability: float = 0.5) -> None:
        """Apply hazard effect: death chance or penalties.

        Args:
            base_kill_probability: Base probability of death (modified by resilience)
        """
        if self.dead:
            return
        # Effective kill probability reduced by resilience
        effective_p = max(0.0, min(1.0, base_kill_probability * (1.0 - 0.7 * self.genome.resilience)))
        if random.random() < effective_p:
            self.dead = True
        else:
            # Non-lethal effect scaled with lack of resilience
            penalty_scale = 0.4 + 0.6 * (1.0 - self.genome.resilience)
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

        # Choose a new desired direction occasionally, more often with curiosity
        self._dir_timer -= dt
        base_interval = 1.8 - 1.6 * self.genome.curiosity
        if self._dir_timer <= 0:
            self._dir_timer = random.uniform(base_interval * 0.5, base_interval * 1.2)
            jitter = random.uniform(-1.0, 1.0) * self.genome.angular_speed
            self._desired_dir = self._desired_dir.rotated(jitter * dt)
            if random.random() < 0.35 * self.genome.curiosity:
                self._desired_dir = pymunk.Vec2d(1, 0).rotated(random.uniform(0, 6.283))

        # Enhanced blend in attraction/avoidance bias - stronger response
        if self._affinity_vec.length > 0.0001:
            target_dir = self._affinity_vec.normalized()
            # Enhanced blend: stronger response to affinity signals
            base_blend = 0.35 + 0.55 * (self.genome.curiosity * 0.6 + self.genome.caution * 0.4)
            # Boost blend when affinity is strong
            if self._affinity_vec.length > 0.7:
                base_blend *= 1.2  # 20% boost for strong signals
            if base_blend > 0.95:  # Increased max from 0.9 to 0.95
                base_blend = 0.95
            self._desired_dir = (self._desired_dir * (1.0 - base_blend) + target_dir * base_blend).normalized()

        # Steer towards desired_dir with environment speed multiplier
        effective_max_speed = self.genome.max_speed * max(0.3, min(2.0, self._env_speed_mult))
        desired_velocity = self._desired_dir.normalized() * effective_max_speed
        steer = desired_velocity - self.body.velocity
        # Enhanced drive: aggression increases drive, and strong affinity also boosts
        base_drive = 90.0 + 240.0 * self.genome.aggression
        # Boost drive when strongly attracted/repelled
        if self._affinity_vec.length > 0.6:
            base_drive *= 1.4  # 40% boost when affinity is strong
        drive = base_drive
        impulse = steer * (self.body.mass * dt * (drive / max(1.0, effective_max_speed)))
        self.body.apply_impulse_at_local_point(impulse)

        # Soft cap speed
        v = self.body.velocity
        speed = v.length
        if speed > effective_max_speed:
            self.body.velocity = v * (effective_max_speed / speed)


def spawn_creatures(space: pymunk.Space, count: int, area: Tuple[int, int, int, int]) -> List[Creature]:
    """Spawn multiple creatures randomly within the specified area.

    Args:
        space: Pymunk physics space
        count: Number of creatures to spawn
        area: Tuple of (x0, y0, x1, y1) bounding box

    Returns:
        List of created creatures
    """
    x0, y0, x1, y1 = area
    creatures: List[Creature] = []
    for _ in range(count):
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        creatures.append(Creature(space, (x, y)))
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
