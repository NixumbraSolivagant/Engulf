"""Top-down physics simulation scene with creatures and dynamic terrain with resource/hazard properties."""

from __future__ import annotations

import math
import random
from typing import Tuple, Set, List, Dict

import pygame
import pymunk
import pymunk.pygame_util

from .settings import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    BACKGROUND_COLOR,
    GRAVITY,
    SPACE_ITERATIONS,
    COLLTYPE_WATER_SENSOR,
    COLLTYPE_TERRAIN_MUD,
    COLLTYPE_TERRAIN_BOOST,
    COLLTYPE_TERRAIN_TOXIC,
    COLLTYPE_TERRAIN_MEADOW,
    COLLTYPE_TERRAIN_CRYSTAL,
    COLLTYPE_TERRAIN_FOREST,
    COLLTYPE_TERRAIN_GEYSER,
    COLLTYPE_TERRAIN_LAVA,
    COLLTYPE_TERRAIN_SPIKE,
    COLLTYPE_TERRAIN_STORM,
)
from .config import Config
from . import terrain
from .i18n import get_strings
from .fonts import load_font
from .creatures import spawn_creatures, Creature, spawn_child


# Constants for creature behavior
BREED_WINDOW_SECONDS = 2.0
BREED_PROXIMITY_BONUS = 10.0
MAX_BIRTHS_PER_FRAME = 5
POPULATION_CAP = 64
INITIAL_CREATURE_COUNT = 40

# Affinity computation constants - Enhanced for stronger tropism
RESOURCE_ATTRACT_BASE = 4.0  # Increased from 2.0
RESOURCE_ATTRACT_CURIOSITY_SCALE = 2.5  # Increased from 1.5
RESOURCE_DISTANCE_FALLOFF = 60.0  # Increased from 40.0 (wider attraction range)
HAZARD_REPEL_BASE = 4.5  # Increased from 2.2
HAZARD_REPEL_CAUTION_SCALE = 2.8  # Increased from 1.8
HAZARD_DISTANCE_FALLOFF = 50.0  # Increased from 25.0 (wider repulsion range)
MATE_DISTANCE_FALLOFF = 40.0
MATE_BEAUTY_FERTILITY_SCALE = 1.2


class TopDownScene:
    """Top-down physics simulation scene with creatures and dynamic terrain."""
    def __init__(self, screen: pygame.Surface, cfg: Config):
        """Initialize the top-down scene with physics space, creatures, and terrain.

        Args:
            screen: Pygame surface for rendering
            cfg: Configuration object with simulation parameters
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.cfg = cfg
        self.strings = get_strings(self.cfg.language)
        self.font = load_font(18)

        # Initialize physics space
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.iterations = SPACE_ITERATIONS

        pygame.display.set_caption(self.strings.title)

        # Terrain management - each type has independent timer
        self.terrain_shapes: Dict[str, Set[pymunk.Shape]] = {
            'ice': set(),
            'sand': set(),
            'bounce': set(),
            'water': set(),
            'mud': set(),
            'boost': set(),
            'toxic': set(),
            'meadow': set(),
            'crystal': set(),
            'forest': set(),
            'geyser': set(),
            'lava': set(),
            'spike': set(),
            'storm': set(),
        }
        # Track terrain types for resource/hazard generation
        self.terrain_info: Dict[pymunk.Shape, str] = {}  # shape -> terrain_type
        self.terrain_timers: Dict[str, float] = {
            'ice': random.uniform(15.0, 30.0),
            'sand': random.uniform(18.0, 35.0),
            'bounce': random.uniform(20.0, 40.0),
            'water': random.uniform(22.0, 42.0),
            'mud': random.uniform(16.0, 32.0),
            'boost': random.uniform(19.0, 38.0),
            'toxic': random.uniform(17.0, 34.0),
            'meadow': random.uniform(21.0, 41.0),
            'crystal': random.uniform(20.0, 38.0),
            'forest': random.uniform(22.0, 40.0),
            'geyser': random.uniform(19.0, 36.0),
            'lava': random.uniform(18.0, 35.0),
            'spike': random.uniform(17.0, 33.0),
            'storm': random.uniform(21.0, 39.0),
        }
        
        # Dynamic system state (must be initialized before _build_all_terrains)
        self.terrain_dynamic_state: Dict[pymunk.Shape, Dict] = {}  # shape -> state dict
        self.evolution_timer = 0.0
        self.season_timer = 0.0
        self.current_season = 'normal'  # 'growth', 'danger', 'normal'
        
        self._build_all_terrains()

        # Creatures
        self.creatures: List[Creature] = spawn_creatures(
            self.space,
            count=INITIAL_CREATURE_COUNT,
            area=(120, 120, WINDOW_WIDTH - 120, WINDOW_HEIGHT - 120),
        )
        self.pop_cap = POPULATION_CAP

        # Map bodies to creatures for quick lookup
        self.body_to_creature: Dict[pymunk.Body, Creature] = {c.body: c for c in self.creatures}

        # Resources and hazards are now terrain properties, not separate objects
        # No need for resource/hazard lists or respawn queues

        # Track bodies in water
        self.bodies_in_water: Set[pymunk.Body] = set()

        # Terrain memberships (for special effects)
        self.mud_bodies: Set[pymunk.Body] = set()
        self.boost_bodies: Set[pymunk.Body] = set()
        self.toxic_bodies: Set[pymunk.Body] = set()
        self.meadow_bodies: Set[pymunk.Body] = set()
        self.lava_bodies: Set[pymunk.Body] = set()
        self.spike_bodies: Set[pymunk.Body] = set()
        self.storm_bodies: Set[pymunk.Body] = set()

        self._setup_handlers()

        # Draw modes
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.use_debug_draw = False

    def _build_all_terrains(self):
        """Build all terrain types initially using unified grid system."""
        terrain.build_walls(self.space, self.cfg)
        
        # Build unified grid - all terrain types in one pass
        all_shapes = terrain.build_tiled_terrain(self.space, self.cfg, 'all', 12, 8)
        
        # Group shapes by terrain type and initialize dynamic state
        for shape in all_shapes:
            terrain_type = terrain._get_terrain_type_from_shape(shape)
            self.terrain_shapes[terrain_type].add(shape)
            self.terrain_info[shape] = terrain_type
            # Initialize dynamic state for each terrain
            self._init_terrain_dynamic_state(shape, terrain_type)

    def _clear_terrain_type(self, terrain_type: str):
        """Clear shapes for a specific terrain type."""
        for s in list(self.terrain_shapes[terrain_type]):
            try:
                self.space.remove(s, s.body)
            except Exception:
                try:
                    self.space.remove(s)
                except Exception:
                    pass
            self.terrain_info.pop(s, None)
        self.terrain_shapes[terrain_type].clear()
        # Clear memberships when specific terrain types are removed
        if terrain_type == 'water':
            self.bodies_in_water.clear()
        elif terrain_type == 'mud':
            self.mud_bodies.clear()
        elif terrain_type == 'boost':
            self.boost_bodies.clear()
        elif terrain_type == 'toxic':
            self.toxic_bodies.clear()
        elif terrain_type == 'meadow':
            self.meadow_bodies.clear()

    def _rebuild_terrain_type(self, terrain_type: str):
        """Rebuild a specific terrain type by regenerating entire grid.
        
        For seamless coverage, we regenerate the entire grid when any type changes,
        but keep the same grid structure.
        """
        # Clear all terrains
        self._clear_all_terrains()
        
        # Clear old dynamic states
        self.terrain_dynamic_state.clear()
        
        # Rebuild unified grid with new random distribution
        all_shapes = terrain.build_tiled_terrain(self.space, self.cfg, 'all', 12, 8)
        
        # Group shapes by terrain type and initialize dynamic state
        for shape in all_shapes:
            ttype = terrain._get_terrain_type_from_shape(shape)
            self.terrain_shapes[ttype].add(shape)
            self.terrain_info[shape] = ttype
            # Initialize dynamic state for each terrain
            self._init_terrain_dynamic_state(shape, ttype)
        
        # Reset timer for all terrain types to stay synchronized
        for ttype in self.terrain_timers:
            self.terrain_timers[ttype] = random.uniform(15.0, 40.0)
        
        # Resources and hazards are now terrain properties, no regeneration needed

    def _clear_all_terrains(self):
        """Clear all terrain shapes."""
        for terrain_type in list(self.terrain_shapes.keys()):
            self._clear_terrain_type(terrain_type)

    def _clear_all_resources_and_hazards(self):
        """Clear all resources and hazards - no longer needed as they're terrain properties."""
        pass

    def _maybe_rebuild_terrains(self, dt: float):
        """Check each terrain type independently and rebuild if timer expires.

        Args:
            dt: Time delta since last frame
        """
        for terrain_type in self.terrain_timers:
            self.terrain_timers[terrain_type] -= dt
            if self.terrain_timers[terrain_type] <= 0.0:
                self._rebuild_terrain_type(terrain_type)

    # Resources and hazards are now terrain properties, not separate objects

    def _setup_handlers(self):
        h_water = self.space.add_collision_handler(0, COLLTYPE_WATER_SENSOR)
        h_water.begin = self._on_enter_water
        h_water.separate = self._on_exit_water

        # Resources and hazards are now terrain properties, no collision handlers needed

        def _enter(set_ref: Set[pymunk.Body], arbiter: pymunk.Arbiter):
            other_shape = arbiter.shapes[0] if arbiter.shapes[1].collision_type >= COLLTYPE_TERRAIN_MUD else arbiter.shapes[1]
            if other_shape.body.body_type == pymunk.Body.DYNAMIC:
                set_ref.add(other_shape.body)
            return True

        def _exit(set_ref: Set[pymunk.Body], arbiter: pymunk.Arbiter):
            other_shape = arbiter.shapes[0] if arbiter.shapes[1].collision_type >= COLLTYPE_TERRAIN_MUD else arbiter.shapes[1]
            set_ref.discard(other_shape.body)
            return None

        for ctype, s in [
            (COLLTYPE_TERRAIN_MUD, self.mud_bodies),
            (COLLTYPE_TERRAIN_BOOST, self.boost_bodies),
            (COLLTYPE_TERRAIN_TOXIC, self.toxic_bodies),
            (COLLTYPE_TERRAIN_MEADOW, self.meadow_bodies),
            (COLLTYPE_TERRAIN_LAVA, self.lava_bodies),
            (COLLTYPE_TERRAIN_STORM, self.storm_bodies),
        ]:
            h = self.space.add_collision_handler(0, ctype)
            h.begin = (lambda arb, sp, d, sref=s: _enter(sref, arb))
            h.separate = (lambda arb, sp, d, sref=s: _exit(sref, arb))

    def _on_enter_water(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data):
        other_shape = arbiter.shapes[0] if arbiter.shapes[1].collision_type == COLLTYPE_WATER_SENSOR else arbiter.shapes[1]
        if other_shape.body.body_type == pymunk.Body.DYNAMIC:
            self.bodies_in_water.add(other_shape.body)
        return True

    def _on_exit_water(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data):
        other_shape = arbiter.shapes[0] if arbiter.shapes[1].collision_type == COLLTYPE_WATER_SENSOR else arbiter.shapes[1]
        self.bodies_in_water.discard(other_shape.body)
        return None

    # Resource and hazard collision handlers removed - effects are now terrain-based

    def _apply_water_damping(self):
        for body in list(self.bodies_in_water):
            body.velocity = body.velocity * self.cfg.water_damping_linear
            body.angular_velocity *= self.cfg.water_damping_angular

    def _handle_events(self) -> str | None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.key == pygame.K_g:
                    self.use_debug_draw = not self.use_debug_draw
                if event.key in (pygame.K_F1, pygame.K_TAB):
                    return "settings"
                if event.key == pygame.K_l:
                    self.cfg.language = "en" if self.cfg.language == "zh" else "zh"
                    return "rebuild"
                if event.key == pygame.K_r:
                    return "rebuild"
        return None

    def _compute_affinity(self, c: Creature) -> Tuple[float, float]:
        """Compute steering bias vector for creature based on terrain properties and mates.

        Args:
            c: Creature to compute affinity for

        Returns:
            Tuple of (bias_x, bias_y) normalized steering direction
        """
        px, py = c.body.position.x, c.body.position.y
        
        # Find attractive terrains (high resource value, low hazard)
        # Consider multiple good terrains for better decision
        resource_terrains = []  # List of (dx, dy, dist2, score)
        best_resource_terrain = None
        best_resource_score = -1e12
        best_resource_dist = 1e12
        
        # Find repelling terrains (high hazard, low resource value)
        # Consider multiple dangerous terrains for better avoidance
        hazard_terrains = []  # List of (dx, dy, dist2, score)
        worst_hazard_terrain = None
        worst_hazard_score = 1e12
        worst_hazard_dist = 1e12
        
        # Sample nearby terrains to compute attraction/repulsion
        # Optimize: only check terrains within reasonable distance
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                # Get terrain center
                cx = shape.body.position.x
                cy = shape.body.position.y
                dx = cx - px
                dy = cy - py
                dist2 = dx * dx + dy * dy
                
                # Early exit if too far (optimization)
                if dist2 > 100000:  # Skip if distance > 316 pixels
                    continue
                
                # Get resource/hazard values from dynamic state if available
                state = self.terrain_dynamic_state.get(shape, {})
                if state:
                    resource_val = state.get('current_resource', 0.0)
                    hazard_val = state.get('current_hazard', 0.0)
                elif hasattr(shape, 'resource_value') and hasattr(shape, 'hazard_level'):
                    resource_val = shape.resource_value
                    hazard_val = shape.hazard_level
                else:
                    continue  # Skip terrains without resource/hazard data
                
                # Enhanced scoring: stronger weighting for resources and hazards
                # Resource score: heavily weighted by resource value and curiosity
                resource_score = resource_val * (2.0 + 3.0 * c.genome.curiosity) - hazard_val * (0.5 + 0.5 * c.genome.caution)
                # Hazard score: heavily weighted by hazard level and caution
                hazard_score = hazard_val * (2.5 + 3.5 * c.genome.caution) - resource_val * (0.3 + 0.3 * c.genome.curiosity)
                
                # Collect multiple attractive terrains
                if resource_score > 0 and dist2 < 25000:  # Increased range
                    resource_terrains.append((dx, dy, dist2, resource_score))
                    if resource_score > best_resource_score:
                        best_resource_score = resource_score
                        best_resource_terrain = (dx, dy, dist2)
                        best_resource_dist = dist2
                
                # Collect multiple dangerous terrains
                if hazard_score > 0 and dist2 < 90000:  # Increased range
                    hazard_terrains.append((dx, dy, dist2, hazard_score))
                    if hazard_score > worst_hazard_score:
                        worst_hazard_score = hazard_score
                        worst_hazard_terrain = (dx, dy, dist2)
                        worst_hazard_dist = dist2
        
        # Mating instinct: find nearest ready same-species mate
        mate_dx = 0.0
        mate_dy = 0.0
        mate_d2 = 1e12
        if c._breed_cd <= BREED_WINDOW_SECONDS:
            for other in self.creatures:
                if other is c or other.dead:
                    continue
                if other.genome.species_id != c.genome.species_id:
                    continue
                if other._breed_cd > BREED_WINDOW_SECONDS:
                    continue
                ox, oy = other.body.position.x, other.body.position.y
                dx = ox - px
                dy = oy - py
                d2 = dx * dx + dy * dy
                if d2 < mate_d2:
                    mate_d2 = d2
                    mate_dx, mate_dy = dx, dy
        
        # Convert to bias vector with distance falloff and gene scaling
        bias_x = 0.0
        bias_y = 0.0

        # Enhanced terrain attraction - consider multiple attractive terrains
        # Use top 3 terrains for better navigation
        resource_terrains.sort(key=lambda x: x[3], reverse=True)
        for i, (dx, dy, dist2, score) in enumerate(resource_terrains[:3]):  # Top 3
            if dist2 > 0.1:
                sqrt_dist = math.sqrt(max(0.01, dist2))
                # Reduced distance falloff for stronger long-range attraction
                inv = 1.0 / (sqrt_dist * 0.7 + RESOURCE_DISTANCE_FALLOFF * 0.6)
                # Weight decreases for further terrains
                weight = 1.0 / (1.0 + i * 0.5)
                attract = RESOURCE_ATTRACT_BASE * (
                    1.2 + RESOURCE_ATTRACT_CURIOSITY_SCALE * c.genome.curiosity
                ) * inv * max(0.2, score) * 1.5 * weight  # Amplified by 1.5x
                bias_x += dx * attract
                bias_y += dy * attract

        # Enhanced terrain repulsion - consider multiple dangerous terrains
        # Use top 3 most dangerous terrains for better avoidance
        hazard_terrains.sort(key=lambda x: x[3], reverse=True)
        for i, (dx, dy, dist2, score) in enumerate(hazard_terrains[:3]):  # Top 3
            if dist2 > 0.1:
                sqrt_dist = math.sqrt(max(0.01, dist2))
                # Reduced distance falloff for stronger long-range repulsion
                inv = 1.0 / (sqrt_dist * 0.6 + HAZARD_DISTANCE_FALLOFF * 0.5)
                # Weight decreases for further terrains
                weight = 1.0 / (1.0 + i * 0.4)
                repel = HAZARD_REPEL_BASE * (
                    1.3 + HAZARD_REPEL_CAUTION_SCALE * c.genome.caution
                ) * inv * max(0.2, score) * 1.8 * weight  # Amplified by 1.8x
                bias_x -= dx * repel
                bias_y -= dy * repel

        # Mating attraction (beauty and fertility influence)
        if mate_d2 < 1e12 and mate_d2 > 0.1:
            sqrt_mate_d2 = math.sqrt(max(0.01, mate_d2))
            inv = 1.0 / (sqrt_mate_d2 + MATE_DISTANCE_FALLOFF)
            mate_weight = 1.0 + MATE_BEAUTY_FERTILITY_SCALE * (
                0.5 * (c.genome.beauty + c.genome.fertility)
            )
            bias_x += mate_dx * inv * mate_weight
            bias_y += mate_dy * inv * mate_weight
        
        # Clamp magnitude
        mag = math.sqrt(bias_x * bias_x + bias_y * bias_y)
        if mag > 1.0:
            bias_x /= mag
            bias_y /= mag
        return bias_x, bias_y

    def _apply_terrain_env(self, c: Creature, dt: float):
        """Apply terrain effects to creature (speed, breeding, hazards, resources).

        Args:
            c: Creature to apply terrain effects to
            dt: Time delta since last frame
        """
        body = c.body
        speed_mult = 1.0
        cooldown_bonus = 0.0
        ambient_hazard = 0.0
        resource_value = 0.0
        hazard_level = 0.0

        # Find which terrain the creature is on and apply its properties
        px, py = body.position.x, body.position.y
        current_terrain_shapes = []
        
        # Check all terrain types
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                if isinstance(shape, pymunk.Circle):
                    dx = px - shape.body.position.x
                    dy = py - shape.body.position.y
                    if dx * dx + dy * dy <= shape.radius * shape.radius:
                        current_terrain_shapes.append(shape)
                elif isinstance(shape, pymunk.Poly):
                    # Point-in-polygon check
                    verts = shape.get_vertices()
                    if len(verts) >= 3:
                        # Transform point to local coordinates
                        cosa = math.cos(-shape.body.angle)
                        sina = math.sin(-shape.body.angle)
                        bx, by = shape.body.position.x, shape.body.position.y
                        local_x = (px - bx) * cosa - (py - by) * sina
                        local_y = (px - bx) * sina + (py - by) * cosa
                        
                        inside = False
                        j = len(verts) - 1
                        for i in range(len(verts)):
                            vi_x, vi_y = verts[i].x, verts[i].y
                            vj_x, vj_y = verts[j].x, verts[j].y
                            if ((vi_y > local_y) != (vj_y > local_y)) and (local_x < (vj_x - vi_x) * (local_y - vi_y) / (vj_y - vi_y) + vi_x):
                                inside = not inside
                            j = i
                        if inside:
                            current_terrain_shapes.append(shape)
        
        # Check if on spike terrain (spike is not sensor, need to check manually)
        on_spike = False
        for shape in current_terrain_shapes:
            if hasattr(shape, 'terrain_type') and shape.terrain_type == 'spike':
                on_spike = True
                break
        if on_spike:
            self.spike_bodies.add(body)
        else:
            self.spike_bodies.discard(body)
        
        # Accumulate effects from all overlapping terrains
        # Use dynamic state if available, otherwise fall back to shape attributes
        for shape in current_terrain_shapes:
            state = self.terrain_dynamic_state.get(shape, {})
            if state:
                # Use current dynamic values
                resource_value += state.get('current_resource', 0.0)
                hazard_level += state.get('current_hazard', 0.0)
            elif hasattr(shape, 'resource_value') and hasattr(shape, 'hazard_level'):
                # Fallback to static values
                resource_value += shape.resource_value
                hazard_level += shape.hazard_level
        
        # Apply terrain-specific modifiers
        if body in self.mud_bodies:
            speed_mult *= 0.6
        if body in self.boost_bodies:
            speed_mult *= 1.25
        if body in self.toxic_bodies:
            # Toxic already handled via hazard_level from terrain properties
            pass
        if body in self.meadow_bodies:
            cooldown_bonus += 0.8  # Breeding cooldown reduction per second
        if body in self.lava_bodies:
            speed_mult *= 0.7  # Lava is viscous
            # Accumulating danger for lava - handled in _update_terrain_dynamics
            # The accumulated_danger is already added to hazard_level in the dynamic update
        if body in self.spike_bodies:
            # Spike: speed penalty - faster movement = higher danger
            speed = math.sqrt(body.velocity.x ** 2 + body.velocity.y ** 2)
            if speed > 50:  # Fast movement
                hazard_level += 0.15 * (speed / 100.0)  # Scale with speed
        if body in self.storm_bodies:
            # Storm: random push effect
            if random.random() < 0.1:  # 10% chance per frame
                angle = random.uniform(0, 2 * math.pi)
                push_force = 50.0
                body.apply_impulse_at_local_point(
                    (math.cos(angle) * push_force, math.sin(angle) * push_force),
                    (0, 0)
                )

        # Apply resource effects continuously
        if resource_value > 0:
            # Resource provides growth, speed, and breeding benefits
            # Scale by creature's metabolism, speed_adaptability, and fertility genes
            growth_amount = resource_value * dt * c.genome.metabolism
            speed_boost = resource_value * dt * c.genome.speed_adaptability * 0.3
            breed_reduction = resource_value * dt * c.genome.fertility * 0.5
            
            # Apply growth (increase size and mass)
            if growth_amount > 0:
                old_radius = c.genome.body_radius
                new_radius = min(30.0, old_radius + growth_amount * 0.5)
                if new_radius > old_radius:
                    # Update physics body
                    c.genome.body_radius = new_radius
                    new_mass = max(0.6, c.genome.body_radius * 0.15)
                    # Remove old shape and create new one with updated radius
                    old_shape = c.shape
                    try:
                        self.space.remove(old_shape)
                    except Exception:
                        pass
                    # Create new shape with updated radius
                    c.shape = pymunk.Circle(c.body, c.genome.body_radius)
                    c.shape.friction = old_shape.friction if hasattr(old_shape, 'friction') else 0.6
                    c.shape.elasticity = old_shape.elasticity if hasattr(old_shape, 'elasticity') else 0.2
                    r, g, b = old_shape.color[:3] if hasattr(old_shape, "color") and isinstance(old_shape.color, (tuple, list)) and len(old_shape.color) >= 3 else (255, 255, 255)
                    c.shape.color = (r, g, b, 255)
                    # Update physics properties
                    c.body.mass = new_mass
                    c.body.moment = pymunk.moment_for_circle(new_mass, 0, c.genome.body_radius)
                    self.space.add(c.shape)
                    # Update body mapping if needed
                    if c.body in self.body_to_creature:
                        self.body_to_creature[c.body] = c
            
            # Apply speed boost
            if speed_boost > 0:
                speed_mult *= (1.0 + speed_boost)
            
            # Apply breeding cooldown reduction
            cooldown_bonus += breed_reduction

        # Apply hazard effects continuously
        if hazard_level > 0:
            ambient_hazard = hazard_level
            
            # Chance to suffer hazard each frame
            if random.random() < hazard_level * dt:
                # Apply hazard effect (scaled by resilience)
                kill_prob = 0.5 * (1.0 - c.genome.resilience * 0.7)
                if random.random() < kill_prob:
                    c.dead = True
                else:
                    # Penalties
                    c.genome.max_speed *= 0.95
                    c._breed_cd += 1.0

        c.set_env_mods(speed_mult, cooldown_bonus, ambient_hazard)

    def _init_terrain_dynamic_state(self, shape: pymunk.Shape, terrain_type: str):
        """Initialize dynamic state for a terrain shape.
        
        Args:
            shape: Terrain shape
            terrain_type: Type of terrain
        """
        base_resource, base_hazard = terrain.TERRAIN_PROPERTIES.get(terrain_type, (0.0, 0.0))
        
        state = {
            'base_resource': base_resource,
            'base_hazard': base_hazard,
            'current_resource': base_resource,
            'current_hazard': base_hazard,
            'wave_timer': random.uniform(0.0, 10.0),  # For periodic waves
            'wave_period': random.uniform(5.0, 15.0),
            'event_timer': random.uniform(0.0, 5.0),  # For special events
            'event_active': False,
            'accumulated_danger': 0.0,  # For accumulating hazards
            'creature_count': 0,
            'last_creature_count': 0,
        }
        
        # Terrain-specific initialization
        if terrain_type == 'crystal':
            state['wave_period'] = random.uniform(5.0, 8.0)
            state['burst_timer'] = random.uniform(0.0, 10.0)
        elif terrain_type == 'geyser':
            state['burst_timer'] = random.uniform(0.0, 6.0)
            state['burst_duration'] = 0.0
        elif terrain_type == 'lava':
            state['accumulated_danger'] = 0.0
            state['lava_surge_timer'] = random.uniform(0.0, 10.0)
        elif terrain_type == 'storm':
            state['lightning_timer'] = random.uniform(0.0, 4.0)
            state['lightning_duration'] = 0.0
        elif terrain_type == 'meadow':
            state['day_night_timer'] = random.uniform(0.0, 20.0)
            state['is_day'] = True
        
        self.terrain_dynamic_state[shape] = state

    def _update_terrain_dynamics(self, dt: float):
        """Update all terrain dynamic systems (waves, events, evolution, feedback).
        
        Args:
            dt: Time delta since last frame
        """
        # Update season timer
        self.season_timer += dt
        if self.season_timer >= 90.0:  # 90 seconds per season cycle
            self.season_timer = 0.0
            # Cycle: growth -> normal -> danger -> normal
            if self.current_season == 'normal':
                self.current_season = 'growth' if random.random() < 0.5 else 'danger'
            else:
                self.current_season = 'normal'
        
        # Update evolution timer
        self.evolution_timer += dt
        if self.evolution_timer >= 60.0:  # Check every 60 seconds
            self.evolution_timer = 0.0
            self._check_terrain_evolution()
        
        # Update each terrain's dynamic state
        for shape, state in self.terrain_dynamic_state.items():
            if not hasattr(shape, 'terrain_type'):
                continue
            
            terrain_type = shape.terrain_type
            state['wave_timer'] += dt
            
            # Layer 2: Periodic waves (all terrains)
            if state['wave_timer'] >= state['wave_period']:
                state['wave_timer'] = 0.0
                state['wave_period'] = random.uniform(5.0, 15.0)
            
            # Calculate wave factor with safety check
            wave_period = max(0.1, state['wave_period'])  # Avoid division by zero
            wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
            state['current_resource'] = state['base_resource'] * wave_factor
            state['current_hazard'] = state['base_hazard'] * wave_factor
            
            # Layer 3: Special events (terrain-specific)
            if terrain_type == 'crystal':
                state.setdefault('burst_timer', 0.0)
                state['burst_timer'] += dt
                if state['burst_timer'] >= 10.0:
                    if random.random() < 0.2:  # 20% chance
                        state['event_active'] = True
                        state['event_duration'] = 2.0
                        state['current_resource'] = state['base_resource'] * 2.0  # Double resource
                    state['burst_timer'] = 0.0
                if state.get('event_active', False):
                    state['event_duration'] -= dt
                    if state['event_duration'] <= 0:
                        state['event_active'] = False
                        # Reset to wave-based value
                        wave_period = max(0.1, state['wave_period'])
                        wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
                        state['current_resource'] = state['base_resource'] * wave_factor
            
            elif terrain_type == 'geyser':
                state.setdefault('burst_timer', 0.0)
                state.setdefault('burst_interval', random.uniform(4.0, 6.0))
                state['burst_timer'] += dt
                if state['burst_timer'] >= state['burst_interval']:
                    state['burst_timer'] = 0.0
                    state['burst_interval'] = random.uniform(4.0, 6.0)  # Set next interval
                    state['event_active'] = True
                    state['event_duration'] = 1.0
                    state['current_resource'] = state['base_resource'] * 2.0
                    state['current_hazard'] = state['base_hazard'] * 2.0
                if state.get('event_active', False):
                    state['event_duration'] -= dt
                    if state['event_duration'] <= 0:
                        state['event_active'] = False
                        # Reset to wave-based values
                        wave_period = max(0.1, state['wave_period'])
                        wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
                        state['current_resource'] = state['base_resource'] * wave_factor
                        state['current_hazard'] = state['base_hazard'] * wave_factor
            
            elif terrain_type == 'lava':
                state.setdefault('lava_surge_timer', 0.0)
                state.setdefault('lava_surge_interval', random.uniform(8.0, 12.0))
                state.setdefault('accumulated_danger', 0.0)
                
                # Accumulate danger when creatures are present
                if state.get('creature_count', 0) > 0:
                    state['accumulated_danger'] = min(0.2, state['accumulated_danger'] + dt * 0.05)
                else:
                    # Decay accumulated danger when no creatures
                    state['accumulated_danger'] = max(0.0, state['accumulated_danger'] - dt * 0.02)
                
                state['lava_surge_timer'] += dt
                if state['lava_surge_timer'] >= state['lava_surge_interval']:
                    state['lava_surge_timer'] = 0.0
                    state['lava_surge_interval'] = random.uniform(8.0, 12.0)  # Set next interval
                    state['current_hazard'] = state['base_hazard'] * 1.5 + state['accumulated_danger']
                    state['event_duration'] = 2.0
                elif state.get('event_duration', 0) > 0:
                    state['event_duration'] -= dt
                    if state['event_duration'] <= 0:
                        # Reset to wave-based value plus accumulated danger
                        wave_period = max(0.1, state['wave_period'])
                        wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
                        state['current_hazard'] = state['base_hazard'] * wave_factor + state['accumulated_danger']
                else:
                    # Normal state: wave + accumulated danger
                    wave_period = max(0.1, state['wave_period'])
                    wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
                    state['current_hazard'] = state['base_hazard'] * wave_factor + state['accumulated_danger']
            
            elif terrain_type == 'storm':
                state.setdefault('lightning_timer', 0.0)
                state.setdefault('lightning_interval', random.uniform(2.0, 4.0))
                state['lightning_timer'] += dt
                if state['lightning_timer'] >= state['lightning_interval']:
                    state['lightning_timer'] = 0.0
                    state['lightning_interval'] = random.uniform(2.0, 4.0)  # Set next interval
                    if random.random() < 0.35:  # 35% chance
                        state['current_hazard'] = state['base_hazard'] * 1.75
                        state['event_duration'] = 0.5
                if state.get('event_duration', 0) > 0:
                    state['event_duration'] -= dt
                    if state['event_duration'] <= 0:
                        # Reset to wave-based value
                        wave_period = max(0.1, state['wave_period'])
                        wave_factor = 0.85 + 0.3 * math.sin(state['wave_timer'] / wave_period * 2 * math.pi)
                        state['current_hazard'] = state['base_hazard'] * wave_factor
            
            elif terrain_type == 'meadow':
                state.setdefault('day_night_timer', 0.0)
                state.setdefault('is_day', True)
                state['day_night_timer'] += dt
                if state['day_night_timer'] >= 20.0:
                    state['day_night_timer'] = 0.0
                    state['is_day'] = not state['is_day']
                # Day: normal, Night: -33% resource
                if state['is_day']:
                    state['current_resource'] = state['base_resource'] * wave_factor
                else:
                    state['current_resource'] = state['base_resource'] * wave_factor * 0.67
            
            # Layer 4: Season effects (apply after special events to avoid conflicts)
            if not state.get('event_active', False) and state.get('event_duration', 0) <= 0:
                if self.current_season == 'growth' and state['base_resource'] > 0:
                    state['current_resource'] *= 1.5
                elif self.current_season == 'danger' and state['base_hazard'] > 0:
                    state['current_hazard'] *= 1.3
            
            # Layer 5: Creature feedback
            # Optimize: only count creatures every 0.5 seconds to reduce CPU load
            state.setdefault('creature_count_update_timer', 0.0)
            state['creature_count_update_timer'] += dt
            if state['creature_count_update_timer'] >= 0.5:  # Update every 0.5 seconds
                state['creature_count_update_timer'] = 0.0
                creature_count = 0
                for creature in self.creatures:
                    if creature.dead:
                        continue
                    px, py = creature.body.position.x, creature.body.position.y
                    if self._point_in_terrain_shape(px, py, shape):
                        creature_count += 1
                state['creature_count'] = creature_count
            else:
                creature_count = state.get('creature_count', 0)
            
            # Resource consumption
            if creature_count > 0 and state['base_resource'] > 0:
                consumption = min(0.05 * creature_count * dt, state['current_resource'] * 0.5)
                state['current_resource'] = max(state['base_resource'] * 0.5, state['current_resource'] - consumption)
            elif creature_count == 0 and state['current_resource'] < state['base_resource']:
                # Recover when no creatures
                state['current_resource'] = min(state['base_resource'], state['current_resource'] + 0.02 * dt)
            
            # Pollution accumulation (for hazard terrains)
            if creature_count >= 10 and state['base_hazard'] > 0:
                state['current_hazard'] = min(state['base_hazard'] * 1.5, state['current_hazard'] + 0.05 * dt)
            
            # Update shape properties
            shape.resource_value = state['current_resource']
            shape.hazard_level = state['current_hazard']
            
            state['last_creature_count'] = creature_count

    def _point_in_terrain_shape(self, x: float, y: float, shape: pymunk.Shape) -> bool:
        """Check if a point is inside a terrain shape.
        
        Args:
            x: X coordinate
            y: Y coordinate
            shape: Terrain shape
            
        Returns:
            True if point is inside shape
        """
        if isinstance(shape, pymunk.Circle):
            dx = x - shape.body.position.x
            dy = y - shape.body.position.y
            return dx * dx + dy * dy <= shape.radius * shape.radius
        elif isinstance(shape, pymunk.Poly):
            verts = shape.get_vertices()
            if len(verts) < 3:
                return False
            # Transform to local coordinates
            cosa = math.cos(-shape.body.angle)
            sina = math.sin(-shape.body.angle)
            bx, by = shape.body.position.x, shape.body.position.y
            local_x = (x - bx) * cosa - (y - by) * sina
            local_y = (x - bx) * sina + (y - by) * cosa
            
            inside = False
            j = len(verts) - 1
            for i in range(len(verts)):
                vi_x, vi_y = verts[i].x, verts[i].y
                vj_x, vj_y = verts[j].x, verts[j].y
                if ((vi_y > local_y) != (vj_y > local_y)) and (local_x < (vj_x - vi_x) * (local_y - vi_y) / (vj_y - vi_y) + vi_x):
                    inside = not inside
                j = i
            return inside
        return False

    def _check_terrain_evolution(self):
        """Check and perform terrain evolution (low probability type changes)."""
        for terrain_type, shapes in list(self.terrain_shapes.items()):
            for shape in list(shapes):
                if random.random() > 0.05:  # 5% chance per terrain per cycle
                    continue
                
                # Evolution rules
                if terrain_type == 'sand':
                    # Check nearby terrains
                    nearby_resources = 0
                    cx, cy = shape.body.position.x, shape.body.position.y
                    for other_type, other_shapes in self.terrain_shapes.items():
                        if other_type in ['meadow', 'forest', 'crystal']:
                            for other_shape in other_shapes:
                                ox, oy = other_shape.body.position.x, other_shape.body.position.y
                                dist2 = (cx - ox) ** 2 + (cy - oy) ** 2
                                if dist2 < 5000:  # Within ~70 pixels
                                    nearby_resources += 1
                    if nearby_resources >= 2:
                        # Evolve to meadow
                        self._evolve_terrain(shape, 'sand', 'meadow')
                
                elif terrain_type == 'toxic':
                    # Long recovery to forest
                    state = self.terrain_dynamic_state.get(shape, {})
                    if state.get('recovery_timer', 0) >= 60.0:
                        self._evolve_terrain(shape, 'toxic', 'forest')
                    else:
                        state['recovery_timer'] = state.get('recovery_timer', 0) + 60.0

    def _evolve_terrain(self, shape: pymunk.Shape, old_type: str, new_type: str):
        """Evolve a terrain from one type to another.
        
        Args:
            shape: Terrain shape to evolve
            old_type: Current terrain type
            new_type: New terrain type
        """
        # Remove from old type
        self.terrain_shapes[old_type].discard(shape)
        
        # Update terrain properties
        base_resource, base_hazard = terrain.TERRAIN_PROPERTIES.get(new_type, (0.0, 0.0))
        shape.terrain_type = new_type
        shape.resource_value = base_resource
        shape.hazard_level = base_hazard
        
        # Add to new type
        self.terrain_shapes[new_type].add(shape)
        
        # Reinitialize dynamic state
        self._init_terrain_dynamic_state(shape, new_type)

    def _update_creatures(self, dt: float):
        """Update all creatures with terrain effects, affinity, and physics.

        Args:
            dt: Time delta since last frame
        """
        for c in self.creatures:
            # Update env mods from terrain memberships and apply resource/hazard effects
            self._apply_terrain_env(c, dt)
            c.set_affinity(self._compute_affinity(c))
            c.update(dt)

    def _cull_dead(self) -> None:
        """Remove dead creatures from simulation and physics space."""
        alive: List[Creature] = []
        new_map: Dict[pymunk.Body, Creature] = {}
        for c in self.creatures:
            if c.dead:
                try:
                    self.space.remove(c.shape, c.body)
                except Exception:
                    pass
            else:
                alive.append(c)
                new_map[c.body] = c
        self.creatures = alive
        self.body_to_creature = new_map

    def _breed_encounters(self) -> None:
        """Check for breeding opportunities between same-species creatures."""
        if len(self.creatures) >= self.pop_cap:
            return
        births = 0
        n = len(self.creatures)
        for i in range(n):
            if births >= MAX_BIRTHS_PER_FRAME or len(self.creatures) >= self.pop_cap:
                break
            a = self.creatures[i]
            for j in range(i + 1, n):
                if births >= MAX_BIRTHS_PER_FRAME or len(self.creatures) >= self.pop_cap:
                    break
                b = self.creatures[j]
                if a.genome.species_id != b.genome.species_id:
                    continue
                # Relaxed readiness: one ready and the other near-ready also allowed
                a_ready = a._breed_cd <= 0.0
                b_ready = b._breed_cd <= 0.0
                a_near = a._breed_cd <= 1.0
                b_near = b._breed_cd <= 1.0
                # Both must be ready or one ready and the other near-ready
                if not ((a_ready and (b_ready or b_near)) or (b_ready and (a_ready or a_near))):
                    continue
                pa = a.body.position
                pb = b.body.position
                dist2 = (pa.x - pb.x) * (pa.x - pb.x) + (pa.y - pb.y) * (pa.y - pb.y)
                threshold = a.genome.body_radius + b.genome.body_radius + BREED_PROXIMITY_BONUS
                if dist2 <= threshold * threshold:
                    # Breeding success influenced by beauty (average of parents)
                    beauty_avg = 0.5 * (a.genome.beauty + b.genome.beauty)
                    base_p = 0.70
                    bonus = 0.5 * (beauty_avg - 0.5)  # Range: [-0.25, +0.25]
                    success_p = max(0.2, min(0.95, base_p + bonus))
                    if random.random() < success_p:
                        child = spawn_child(self.space, a, b)
                        self.creatures.append(child)
                        self.body_to_creature[child.body] = child
                        # Shorter cooldowns after breeding
                        a._breed_cd = random.uniform(2.0, 4.0)
                        b._breed_cd = random.uniform(2.0, 4.0)
                        births += 1

    def _predation_encounters(self) -> None:
        """Check for predation opportunities between different species."""
        n = len(self.creatures)
        for i in range(n):
            a = self.creatures[i]
            if a.dead:
                continue
            for j in range(i + 1, n):
                b = self.creatures[j]
                if b.dead:
                    continue
                # Prevent same-species predation
                if a.genome.species_id == b.genome.species_id:
                    continue
                pa = a.body.position
                pb = b.body.position
                dx = pa.x - pb.x
                dy = pa.y - pb.y
                dist2 = dx * dx + dy * dy
                threshold = a.genome.body_radius + b.genome.body_radius
                if dist2 > threshold * threshold:
                    continue
                # Determine predator and prey by size
                if a.genome.body_radius > b.genome.body_radius * 1.25:
                    predator, prey = a, b
                elif b.genome.body_radius > a.genome.body_radius * 1.25:
                    predator, prey = b, a
                else:
                    continue
                # Probability based on aggression vs prey caution
                p = 0.35 + 0.45 * predator.genome.aggression - 0.30 * prey.genome.caution
                if random.random() < max(0.05, min(0.95, p)):
                    prey.dead = True
                    predator.devour(prey)

    def _find_terrain_type_at(self, x: float, y: float) -> str:
        """Find terrain type at given position.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Terrain type string, or 'ice' as default
        """
        # Check which terrain shape contains this point
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                if isinstance(shape, pymunk.Circle):
                    dx = shape.body.position.x - x
                    dy = shape.body.position.y - y
                    if dx * dx + dy * dy <= shape.radius * shape.radius:
                        return terrain_type
                elif isinstance(shape, pymunk.Poly):
                    # Simple point-in-polygon check with rotation
                    verts = shape.get_vertices()
                    if len(verts) >= 3:
                        # Transform point to local coordinates
                        cosa = math.cos(-shape.body.angle)
                        sina = math.sin(-shape.body.angle)
                        bx, by = shape.body.position.x, shape.body.position.y
                        local_x = (x - bx) * cosa - (y - by) * sina
                        local_y = (x - bx) * sina + (y - by) * cosa
                        
                        # Check if point is inside polygon (in local coords)
                        inside = False
                        j = len(verts) - 1
                        for i in range(len(verts)):
                            vi_x, vi_y = verts[i].x, verts[i].y
                            vj_x, vj_y = verts[j].x, verts[j].y
                            if ((vi_y > local_y) != (vj_y > local_y)) and (local_x < (vj_x - vi_x) * (local_y - vi_y) / (vj_y - vi_y) + vi_x):
                                inside = not inside
                            j = i
                        if inside:
                            return terrain_type
        return 'ice'  # Default

    # Resource respawn logic removed - resources are now terrain properties

    def _draw_custom(self):
        # Create transparent overlay layer for terrain visualization
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # Draw terrains with transparency
        # Draw all terrain shapes (both sensor and non-sensor), but skip creature shapes
        creature_bodies = {c.body for c in self.creatures}
        for shape in self.space.shapes:
            # Skip creature shapes (they are drawn separately)
            if shape.body in creature_bodies:
                continue
            
            # Only draw shapes that have a color attribute (terrain shapes)
            if not hasattr(shape, "color"):
                continue
            
            color = shape.color
            if not isinstance(color, (tuple, list)) or len(color) < 4:
                continue
            
            if isinstance(shape, pymunk.Poly):
                verts = shape.get_vertices()
                pts = []
                cosa = math.cos(shape.body.angle)
                sina = math.sin(shape.body.angle)
                bx, by = shape.body.position.x, shape.body.position.y
                for v in verts:
                    x = v.x * cosa - v.y * sina + bx
                    y = v.x * sina + v.y * cosa + by
                    pts.append((x, y))
                pygame.draw.polygon(overlay, color, pts, 0)
                # Draw outline with higher alpha
                outline_color = (color[0], color[1], color[2], min(255, color[3] + 40))
                pygame.draw.polygon(overlay, outline_color, pts, 2)
            
            elif isinstance(shape, pymunk.Circle):
                bx, by = shape.body.position.x, shape.body.position.y
                pygame.draw.circle(overlay, color, (int(bx), int(by)), int(shape.radius))
                # Draw outline with higher alpha
                outline_color = (color[0], color[1], color[2], min(255, color[3] + 40))
                pygame.draw.circle(overlay, outline_color, (int(bx), int(by)), int(shape.radius), 2)
        
        # Apply overlay to screen
        self.screen.blit(overlay, (0, 0))

        # Draw creatures as filled circles on top of terrain
        for c in self.creatures:
            bx, by = c.body.position.x, c.body.position.y
            r, g, b, a = c.shape.color if hasattr(c.shape, "color") else (255, 255, 255, 255)
            pygame.draw.circle(self.screen, (r, g, b), (int(bx), int(by)), int(c.genome.body_radius))

    def _draw_hud(self):
        s = self.strings
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            s.scene_label,
            f"  {s.hint_settings}",
            f"  {s.hint_toggle_draw}",
            f"  {s.hint_quit}",
            f"  G: Toggle debug/custom draw",
        ]
        x, y = 10, 10
        for line in lines:
            surf = self.font.render(line, True, (235, 235, 235))
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 2

    def run(self) -> str:
        while True:
            route = self._handle_events()
            if route:
                return route

            dt = 1.0 / FPS

            self._apply_water_damping()
            self._update_terrain_dynamics(dt)  # Update dynamic terrain systems
            self._update_creatures(dt)
            self._breed_encounters()
            self._predation_encounters()
            self._cull_dead()
            self._maybe_rebuild_terrains(dt)

            self.space.step(dt)

            self.screen.fill(BACKGROUND_COLOR)
            if self.use_debug_draw:
                self.space.debug_draw(self.draw_options)
            else:
                self._draw_custom()
            self._draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

