"""Top-down physics simulation scene with creatures and dynamic terrain with resource/hazard properties."""

from __future__ import annotations

import math
import random
from typing import Tuple, Set, List, Dict
import os

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
from .creatures import spawn_creatures, Creature, spawn_child, Genome
from .genetics import is_same_species, are_different_species, check_species_divergence, calculate_genetic_distance

# RL imports (optional - only if USE_RL is enabled)
try:
    from .rl import GlobalRLManager, RLCreature, RLTrainingVisualizer
    from .rl.evolution_supervisor import EvolutionSupervisor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLCreature = None
    GlobalRLManager = None
    RLTrainingVisualizer = None
    EvolutionSupervisor = None


# Constants for creature behavior
BREED_WINDOW_SECONDS = 5.0  # Increased from 2.0 to allow more breeding opportunities
BREED_PROXIMITY_BONUS = 35.0  # Increased to make proximity detection easier
MAX_BIRTHS_PER_FRAME = 10  # Increased from 5 to allow more births per frame
POPULATION_CAP = 5102
INITIAL_CREATURE_COUNT = 120

# Affinity computation constants - Enhanced for stronger tropism and higher activity
RESOURCE_ATTRACT_BASE = 6.0  # Increased from 4.0 for more active movement
RESOURCE_ATTRACT_CURIOSITY_SCALE = 3.0  # Increased from 2.5
RESOURCE_DISTANCE_FALLOFF = 70.0  # Increased from 60.0 (wider attraction range)
HAZARD_REPEL_BASE = 6.5  # Increased from 4.5 for more active avoidance
HAZARD_REPEL_CAUTION_SCALE = 3.5  # Increased from 2.8
HAZARD_DISTANCE_FALLOFF = 60.0  # Increased from 50.0 (wider repulsion range)
MATE_DISTANCE_FALLOFF = 50.0  # Increased from 40.0
MATE_BEAUTY_FERTILITY_SCALE = 1.5  # Increased from 1.2


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
        self._current_diversity = 0.0
        
        self._build_all_terrains()
        
        # Initialize cache timers
        self._species_cache_clear_timer = 0.0
        
        # Species tracking (for unique colors)
        self._existing_species: Dict[int, float] = {}  # species_id -> hue mapping
        self._next_species_id = 3  # Start after initial 3 species
        # Initialize existing species colors (from initial 3 species: 0=red, 1=green, 2=blue)
        self._existing_species[0] = 0.0    # Red
        self._existing_species[1] = 120.0  # Green
        self._existing_species[2] = 240.0   # Blue

        # RL System (if enabled and available)
        self.use_rl = cfg.use_rl and RL_AVAILABLE
        self.rl_visualizer = None
        self.evolution_supervisor = None
        # RL reporting intervals (named constants instead of magic numbers)
        self._rl_report_interval = 500
        self._evolution_record_interval = 1000
        self._evolution_verbose_interval = 5000
        if self.use_rl:
            # Each creature now learns independently (no global manager)
            if RLTrainingVisualizer:
                self.rl_visualizer = RLTrainingVisualizer(max_history=10000)
            if EvolutionSupervisor:
                self.evolution_supervisor = EvolutionSupervisor(
                    selection_rate=0.2,
                    mutation_rate=0.1,
                    diversity_preservation=0.1
                )
        
        # Map bodies to creatures for quick lookup (initialize before spawning)
        self.body_to_creature: Dict[pymunk.Body, Creature] = {}
        
        # Creatures
        if self.use_rl:
            # Use RL creatures
            self.creatures: List[Creature] = self._spawn_rl_creatures(
                count=INITIAL_CREATURE_COUNT,
                area=(120, 120, WINDOW_WIDTH - 120, WINDOW_HEIGHT - 120),
            )
            # Optionally load initial weights from a directory (round-robin)
            try:
                load_dir = getattr(self.cfg, 'rl_load_weights_dir', '')
                if isinstance(load_dir, str) and load_dir:
                    abspath = os.path.abspath(load_dir)
                    if os.path.isdir(abspath):
                        files = [os.path.join(abspath, f) for f in os.listdir(abspath) if f.endswith('.json')]
                        files.sort()
                        if files:
                            assigned = 0
                            for idx, c in enumerate(self.creatures):
                                if hasattr(c, 'use_rl') and c.use_rl and hasattr(c, 'local_agent'):
                                    fpath = files[idx % len(files)]
                                    try:
                                        c.local_agent.load(fpath)
                                        assigned += 1
                                    except Exception:
                                        pass
                            print(f"[RL] Loaded weights for {assigned} agents from {abspath}")
                    else:
                        print(f"[RL] Weight dir not found: {abspath}")
            except Exception as e:
                print(f"[RL] Error loading weights: {e}")
        else:
            # Use regular gene-driven creatures (3 species, high speed)
            self.creatures: List[Creature] = spawn_creatures(
                self.space,
                count=INITIAL_CREATURE_COUNT,
                area=(120, 120, WINDOW_WIDTH - 120, WINDOW_HEIGHT - 120),
                num_species=3,
                high_speed=True,
            )
            # Update body mapping for regular creatures
            self.body_to_creature.update({c.body: c for c in self.creatures})
        
        self.pop_cap = POPULATION_CAP
        
        # RL training state
        if self.use_rl:
            self._rl_train_counter = 0
        
        # Performance optimization: Cache GeneEffects (shared instance)
        from .genetics import GeneEffects
        self._gene_effects_cache = GeneEffects()
        
        # Performance optimization: Cache terrain centers and bounding boxes
        self._terrain_centers_cache: Dict[pymunk.Shape, Tuple[float, float]] = {}
        self._terrain_bbox_cache: Dict[pymunk.Shape, Tuple[float, float, float]] = {}  # (min_dist_sq, max_dist_sq)
        self._cache_terrain_centers()
        
        # Performance optimization: Affinity computation cache (update less frequently)
        self._affinity_cache: Dict[Creature, Tuple[float, float]] = {}
        self._affinity_cache_timer = 0.0
        self._affinity_cache_interval = 0.15  # Update affinity every 0.15 seconds instead of every frame
        
        # Performance optimization: Species comparison cache
        self._species_cache: Dict[Tuple[int, int], bool] = {}  # (id1, id2) -> is_same_species
        
        # Statistics tracking for post-run analysis
        self.stats = {
            'time_steps': [],
            'population': [],
            'births_total': 0,
            'deaths_total': 0,
            'predations_total': 0,
            'resource_consumed': 0.0,
            'avg_speed': [],
            'avg_age': [],
            'species_count': [],
            'terrain_resources': {t: [] for t in self.terrain_shapes.keys()},
            'terrain_hazards': {t: [] for t in self.terrain_shapes.keys()},
        }
        self.stats_timer = 0.0
        self.stats_update_interval = 0.5  # Update stats every 0.5 seconds
        
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
        
        Optimized with caching and early exits.

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
        
        # Pre-compute gene effects (use cached instance)
        genome_dict = c.genome.to_dict()
        resource_attract_mult = self._gene_effects_cache.get_resource_attraction_strength(genome_dict)
        hazard_repel_mult = self._gene_effects_cache.get_hazard_repulsion_strength(genome_dict)
        
        # Sample nearby terrains to compute attraction/repulsion
        # Optimize: use cached centers and early exit (reduced range to avoid over-computation)
        max_check_dist_sq = 80000  # Reduced from 100000 to 80000 (~282 pixels) to reduce locking
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                # Use cached center if available, otherwise compute
                if shape in self._terrain_bbox_cache:
                    cx, cy, radius = self._terrain_bbox_cache[shape]
                    # Quick bounding box check before precise calculation
                    dx = cx - px
                    dy = cy - py
                    dist2 = dx * dx + dy * dy
                    # Early exit: check if outside bounding sphere
                    if dist2 > (radius + max_check_dist_sq ** 0.5) ** 2:
                        continue
                else:
                    # Fallback: compute center
                    cx = shape.body.position.x
                    cy = shape.body.position.y
                    dx = cx - px
                    dy = cy - py
                    dist2 = dx * dx + dy * dy
                
                # Early exit if too far
                if dist2 > max_check_dist_sq:
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
                
                # Early exit if both resource and hazard are negligible
                if resource_val < 0.01 and hazard_val < 0.01:
                    continue
                
                # Resource score: heavily weighted by resource value and curiosity/olfactory
                resource_score = resource_val * (2.0 + 3.0 * c.genome.curiosity) * resource_attract_mult - hazard_val * (0.5 + 0.5 * c.genome.caution)
                # Hazard score: heavily weighted by hazard level and caution
                hazard_score = hazard_val * (2.5 + 3.5 * c.genome.caution) * hazard_repel_mult - resource_val * (0.3 + 0.3 * c.genome.curiosity)
                
                # Collect multiple attractive terrains (reduced range to avoid over-attraction)
                if resource_score > 0 and dist2 < 25000:  # Reduced from 40000 to 25000 to reduce locking
                    resource_terrains.append((dx, dy, dist2, resource_score))
                    if resource_score > best_resource_score:
                        best_resource_score = resource_score
                        best_resource_terrain = (dx, dy, dist2)
                        best_resource_dist = dist2
                
                # Collect multiple dangerous terrains (reduced range slightly)
                if hazard_score > 0 and dist2 < 70000:  # Reduced from 90000 to 70000
                    hazard_terrains.append((dx, dy, dist2, hazard_score))
                    if hazard_score > worst_hazard_score:
                        worst_hazard_score = hazard_score
                        worst_hazard_terrain = (dx, dy, dist2)
                        worst_hazard_dist = dist2
        
        # Mating instinct: find nearest ready same-species mate (optimized)
        mate_dx = 0.0
        mate_dy = 0.0
        mate_d2 = 1e12
        if c._breed_cd <= BREED_WINDOW_SECONDS:
            c_species_id = id(c.genome)
            for other in self.creatures:
                if other is c or other.dead:
                    continue
                if other._breed_cd > BREED_WINDOW_SECONDS:
                    continue
                
                # Quick species check using cache
                cache_key = (c_species_id, id(other.genome))
                if cache_key not in self._species_cache:
                    self._species_cache[cache_key] = is_same_species(other.genome.to_dict(), c.genome.to_dict())
                
                if not self._species_cache[cache_key]:
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
                ) * inv * max(0.2, score) * 1.0 * weight  # Reduced from 1.5x to 1.0x to reduce locking
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
                ) * inv * max(0.2, score) * 1.2 * weight  # Reduced from 1.8x to 1.2x to reduce locking
                bias_x -= dx * repel
                bias_y -= dy * repel

        # Mating attraction (beauty and fertility influence, plus breeding_urge)
        from .genetics import GeneEffects
        gene_effects = GeneEffects()
        genome_dict = c.genome.to_dict()
        breeding_urge = genome_dict.get("breeding_urge", 0.5)
        if mate_d2 < 1e12 and mate_d2 > 0.1:
            sqrt_mate_d2 = math.sqrt(max(0.01, mate_d2))
            inv = 1.0 / (sqrt_mate_d2 + MATE_DISTANCE_FALLOFF)
            mate_weight = 1.0 + MATE_BEAUTY_FERTILITY_SCALE * (
                    (0.3 * c.genome.beauty + 0.3 * c.genome.fertility + 0.4 * breeding_urge)
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
        
        # Check all terrain types (optimized with bounding box pre-check)
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                # Use cached bounding box for quick rejection
                if shape in self._terrain_bbox_cache:
                    cx, cy, radius = self._terrain_bbox_cache[shape]
                    dx = px - cx
                    dy = py - cy
                    dist_sq = dx * dx + dy * dy
                    # Quick rejection: outside bounding sphere
                    if dist_sq > radius * radius * 1.1:  # 10% margin
                        continue
                
                # Precise check
                if isinstance(shape, pymunk.Circle):
                    if shape not in self._terrain_bbox_cache:
                        dx = px - shape.body.position.x
                        dy = py - shape.body.position.y
                    if dx * dx + dy * dy <= shape.radius * shape.radius:
                        current_terrain_shapes.append(shape)
                elif isinstance(shape, pymunk.Poly):
                    # Point-in-polygon check (only if inside bounding box)
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

        # Apply resource effects continuously (optimized for better gameplay)
        # Track resource consumption for RL creatures
        if resource_value > 0:
            # Resource provides growth, speed, and breeding benefits
            # Scale by creature's metabolism, speed_adaptability, and fertility genes
            # Increased multipliers for more noticeable effects
            metabolism = c.genome.metabolism
            speed_adapt = c.genome.speed_adaptability
            fertility = c.genome.fertility
            
            # Growth: more noticeable (increased multiplier)
            growth_amount = resource_value * dt * metabolism * 1.5  # Increased from 1.0 to 1.5
            
            # Speed boost: more impactful (increased multiplier)
            speed_boost = resource_value * dt * speed_adapt * 0.5  # Increased from 0.3 to 0.5
            
            # Breeding cooldown reduction: more effective (increased multiplier)
            breed_reduction = resource_value * dt * fertility * 1.2  # Increased for faster cooldown reduction
            
            # Apply growth (increase size and mass)
            if growth_amount > 0:
                old_radius = c.genome.body_radius
                new_radius = min(30.0, old_radius + growth_amount * 0.6)  # Increased from 0.5 to 0.6
                if new_radius > old_radius:
                    # Update physics body
                    c.genome.body_radius = new_radius
                    
                    # Track resource consumption for RL creatures
                    if self.use_rl and hasattr(c, 'episode_info') and isinstance(c, RLCreature):
                        c.episode_info['gained_resource'] = True
                        c.episode_info['resource_value'] = resource_value * dt
                        c.episode_info['resources_gained'] += resource_value * dt
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
            
            # Apply speed boost (more noticeable)
            if speed_boost > 0:
                speed_mult *= (1.0 + speed_boost * 1.2)  # Additional 20% boost multiplier
            
            # Apply breeding cooldown reduction (more effective)
            cooldown_bonus += breed_reduction * 1.6  # Stronger effectiveness

        # Apply hazard effects continuously
        if hazard_level > 0:
            ambient_hazard = hazard_level
            
            # Chance to suffer hazard each frame
            if random.random() < hazard_level * dt:
                # Get terrain type for RL tracking
                terrain_type = "default"
                for terrain_type_name, shapes in self.terrain_shapes.items():
                    for shape in shapes:
                        if hasattr(shape, 'terrain_type') and self._point_in_terrain_shape(
                            c.body.position.x, c.body.position.y, shape
                        ):
                            terrain_type = shape.terrain_type
                            break
                
                # Apply hazard effect (scaled by resilience)
                # RL creatures track hazards in episode_info
                c.suffer_hazard(0.5, terrain_type=terrain_type)

            # For RL creatures: mark hazard encounter to strengthen learning signal only (no physics push/damp)
            if self.use_rl and isinstance(c, RLCreature) and hasattr(c, 'episode_info'):
                c.episode_info['took_hazard'] = True
                c.episode_info['hazard_level'] = float(hazard_level)

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
                
                # Accumulate danger when creatures are present (slower rate)
                if state.get('creature_count', 0) > 0:
                    # Reduced accumulation rate: 0.02 instead of 0.05
                    state['accumulated_danger'] = min(0.15, state['accumulated_danger'] + dt * 0.02)  # Max cap reduced from 0.2 to 0.15
                else:
                    # Faster decay accumulated danger when no creatures
                    state['accumulated_danger'] = max(0.0, state['accumulated_danger'] - dt * 0.05)  # Increased decay from 0.02 to 0.05
                
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
            # Optimize: only count creatures every 1.0 seconds to further reduce CPU load
            state.setdefault('creature_count_update_timer', 0.0)
            state['creature_count_update_timer'] += dt
            if state['creature_count_update_timer'] >= 1.0:  # Update every 1.0 seconds (increased from 0.5)
                state['creature_count_update_timer'] = 0.0
                creature_count = 0
                # Quick bounding box pre-filter
                if shape in self._terrain_bbox_cache:
                    cx, cy, radius = self._terrain_bbox_cache[shape]
                    for creature in self.creatures:
                        if creature.dead:
                            continue
                        px, py = creature.body.position.x, creature.body.position.y
                        dx = px - cx
                        dy = py - cy
                        dist_sq = dx * dx + dy * dy
                        # Quick rejection
                        if dist_sq > radius * radius * 1.2:
                            continue
                        # Precise check
                        if self._point_in_terrain_shape(px, py, shape):
                            creature_count += 1
                else:
                    # Fallback if no cache
                    for creature in self.creatures:
                        if creature.dead:
                            continue
                        px, py = creature.body.position.x, creature.body.position.y
                        if self._point_in_terrain_shape(px, py, shape):
                            creature_count += 1
                state['creature_count'] = creature_count
            else:
                creature_count = state.get('creature_count', 0)
            
            # Resource consumption (optimized - less aggressive consumption)
            if creature_count > 0 and state['base_resource'] > 0:
                # Further reduced consumption rate for better resource availability
                consumption = min(0.02 * creature_count * dt, state['current_resource'] * 0.25)  # Reduced from 0.03 to 0.02, and from 0.3 to 0.25
                state['current_resource'] = max(state['base_resource'] * 0.65, state['current_resource'] - consumption)  # Slightly higher floor (0.65 vs 0.6)
                # Track resource consumption
                self.stats['resource_consumed'] += consumption
            elif creature_count == 0 and state['current_resource'] < state['base_resource']:
                # Faster recovery when no creatures for better resource flow
                recovery_rate = 0.20 if state['base_resource'] > 0.4 else 0.25  # Faster recovery (increased from 0.10-0.15 to 0.20-0.25)
                state['current_resource'] = min(state['base_resource'], state['current_resource'] + recovery_rate * dt)
            
            # Pollution accumulation (for hazard terrains) - reduced rate and capped
            if creature_count >= 10 and state['base_hazard'] > 0:
                # Much slower accumulation: 0.02 instead of 0.05, max cap at 1.3x instead of 1.5x
                state['current_hazard'] = min(state['base_hazard'] * 1.3, state['current_hazard'] + 0.02 * dt)
            # Decay pollution when fewer creatures (allow recovery)
            elif creature_count < 5 and state['current_hazard'] > state['base_hazard']:
                decay_rate = 0.03  # Allow hazard to decay back to base
                state['current_hazard'] = max(state['base_hazard'], state['current_hazard'] - decay_rate * dt)
            
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
        
        # Update cache
        self._cache_terrain_centers()

    def _get_nearby_creatures(self, creature: Creature, radius: float = 300.0) -> List[Creature]:
        """Get nearby creatures within radius.
        
        Args:
            creature: Reference creature
            radius: Search radius in pixels
            
        Returns:
            List of nearby creatures
        """
        nearby = []
        px, py = creature.body.position.x, creature.body.position.y
        
        for other in self.creatures:
            if other.dead or other is creature:
                continue
            ox, oy = other.body.position.x, other.body.position.y
            dist_sq = (px - ox) ** 2 + (py - oy) ** 2
            if dist_sq < radius * radius:
                nearby.append(other)
        
        return nearby
    
    def _update_creatures(self, dt: float):
        """Update all creatures with terrain effects, affinity, and physics.
        
        Optimized with affinity caching and early exits.

        Args:
            dt: Time delta since last frame
        """
        # Update affinity cache timer
        self._affinity_cache_timer += dt
        should_update_affinity = self._affinity_cache_timer >= self._affinity_cache_interval
        
        if should_update_affinity:
            self._affinity_cache_timer = 0.0
            self._affinity_cache.clear()  # Clear cache for recalculation
        
        for c in self.creatures:
            if c.dead:
                continue
            
            # RL creatures need special update
            if self.use_rl and isinstance(c, RLCreature) and c.use_rl:
                # Track terrain exploration for RL creatures
                terrain_type_at_pos = self._find_terrain_type_at(c.body.position.x, c.body.position.y)
                if terrain_type_at_pos and hasattr(c, '_explored_terrains'):
                    if terrain_type_at_pos not in c._explored_terrains:
                        c.episode_info['explored_new'] = True
                        c._explored_terrains.add(terrain_type_at_pos)
                
                # For RL creatures, compute affinity still needed for some internal tracking
                # but action is controlled by RL
                nearby_creatures = self._get_nearby_creatures(c, radius=300.0)
                
                # Check for resources before applying effects (for near_resource tracking)
                if hasattr(c, 'episode_info'):
                    # Get terrain at position
                    terrain_shapes_at_pos = []
                    px, py = c.body.position.x, c.body.position.y
                    for terrain_type, shapes in self.terrain_shapes.items():
                        for shape in shapes:
                            if isinstance(shape, pymunk.Circle):
                                dx = px - shape.body.position.x
                                dy = py - shape.body.position.y
                                if dx * dx + dy * dy <= shape.radius * shape.radius:
                                    terrain_shapes_at_pos.append(shape)
                    
                    # Check resource value
                    total_resource = 0.0
                    for shape in terrain_shapes_at_pos:
                        state = self.terrain_dynamic_state.get(shape, {})
                        if state:
                            total_resource += state.get('current_resource', 0.0)
                        elif hasattr(shape, 'resource_value'):
                            total_resource += shape.resource_value
                    if total_resource > 0.01:
                        c.episode_info['near_resource'] = True
                
                self._apply_terrain_env(c, dt)  # Still apply terrain effects (this sets gained_resource)
                c.update(dt, scene=self, nearby_creatures=nearby_creatures)
            else:
                # Regular gene-driven creatures
                self._apply_terrain_env(c, dt)
                
                # Use cached affinity if available, otherwise compute
                if should_update_affinity:
                    affinity = self._compute_affinity(c)
                    self._affinity_cache[c] = affinity
                    c.set_affinity(affinity)
                elif c in self._affinity_cache:
                    c.set_affinity(self._affinity_cache[c])
                else:
                    # Fallback: compute if not in cache
                    affinity = self._compute_affinity(c)
                    self._affinity_cache[c] = affinity
                    c.set_affinity(affinity)
                
                c.update(dt)

    def _cull_dead(self) -> None:
        """Remove dead creatures from simulation and physics space."""
        alive: List[Creature] = []
        new_map: Dict[pymunk.Body, Creature] = {}
        for c in self.creatures:
            if c.dead:
                # Track death
                self.stats['deaths_total'] += 1
                
                # Finalize RL episode and track fitness if needed
                if self.use_rl and isinstance(c, RLCreature) and c.use_rl:
                    # Update fitness metrics before death
                    if self.evolution_supervisor:
                        fitness_metrics = c.get_fitness_metrics()
                        self.evolution_supervisor.track_fitness(c.id, fitness_metrics)
                    c._finalize_episode(self, natural_death=(c.age >= c.genome.lifespan_secs))
                    # Record episode in visualizer
                    if self.rl_visualizer and hasattr(c, 'episode_reward') and c.episode_steps > 0:
                        self.rl_visualizer.record_episode(c.episode_reward, c.episode_steps)
                
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
        """Check for breeding opportunities between same-species creatures.
        
        Optimized with species comparison caching.
        """
        if len(self.creatures) >= self.pop_cap:
            return
        births = 0
        n = len(self.creatures)
        
        # Pre-filter: only check creatures that are potentially ready
        potential_breeders = [c for c in self.creatures 
                             if not c.dead and c._breed_cd <= BREED_WINDOW_SECONDS]
        
        if len(potential_breeders) < 2:
            return
        
        for i in range(len(potential_breeders)):
            if births >= MAX_BIRTHS_PER_FRAME or len(self.creatures) >= self.pop_cap:
                break
            a = potential_breeders[i]
            a_species_id = id(a.genome)  # Use object id as quick hash
            
            for j in range(i + 1, len(potential_breeders)):
                if births >= MAX_BIRTHS_PER_FRAME or len(self.creatures) >= self.pop_cap:
                    break
                b = potential_breeders[j]
                
                # Quick species check using cache
                cache_key = (a_species_id, id(b.genome))
                if cache_key not in self._species_cache:
                    self._species_cache[cache_key] = is_same_species(a.genome.to_dict(), b.genome.to_dict())
                
                if not self._species_cache[cache_key]:
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
                    # Breeding success influenced by beauty and breeding_timing (average of parents)
                    # Use cached GeneEffects instance
                    genome_a_dict = a.genome.to_dict()
                    genome_b_dict = b.genome.to_dict()
                    beauty_avg = 0.5 * (a.genome.beauty + b.genome.beauty)
                    timing_avg = 0.5 * (genome_a_dict.get("breeding_timing", 0.5) + genome_b_dict.get("breeding_timing", 0.5))
                    success_mod = self._gene_effects_cache.get_breeding_success_modifier(genome_a_dict) * 0.5 + \
                                 self._gene_effects_cache.get_breeding_success_modifier(genome_b_dict) * 0.5
                    base_p = 0.85 * success_mod  # Increased from 0.70 to 0.85 for higher breeding success
                    bonus = 0.5 * (beauty_avg - 0.5) + 0.2 * (timing_avg - 0.5)  # Range: [-0.35, +0.35]
                    success_p = max(0.5, min(0.98, base_p + bonus))  # Raise min to 0.5 to ensure births
                    # Inform RL: mating proximity reward signal
                    if isinstance(a, RLCreature) and a.use_rl:
                        a.episode_info['mating_proximity'] = True
                    if isinstance(b, RLCreature) and b.use_rl:
                        b.episode_info['mating_proximity'] = True
                    if random.random() < success_p:
                        # Pre-fetch parent genomes
                        a_gen = a.genome.to_dict()
                        b_gen = b.genome.to_dict()
                        
                        # Fitness-weighted breeding if available
                        if self.evolution_supervisor and isinstance(a, RLCreature) and isinstance(b, RLCreature):
                            fitness_a_raw = self.evolution_supervisor.get_fitness(a.id)
                            fitness_b_raw = self.evolution_supervisor.get_fitness(b.id)
                            max_fit = max(fitness_a_raw, fitness_b_raw, 1.0)
                            inv = 1.0 / max_fit
                            fitness_a = max(0.1, min(1.0, fitness_a_raw * inv))
                            fitness_b = max(0.1, min(1.0, fitness_b_raw * inv))
                            child_genome = Genome.breed(a.genome, b.genome, fitness_a, fitness_b)
                        else:
                            child_genome = Genome.breed(a.genome, b.genome)
                        
                        # Divergence check for new species
                        child_gen = child_genome.to_dict()
                        is_new_species = False
                        if check_species_divergence(child_gen, a_gen, b_gen):
                            child_genome = self._assign_new_species(child_gen)
                            is_new_species = True
                        
                        # Create child at midpoint between parents
                        mid_x = (pa.x + pb.x) * 0.5
                        mid_y = (pa.y + pb.y) * 0.5
                        child = RLCreature(
                            self.space,
                            (mid_x, mid_y),
                            genome=child_genome,
                            use_rl=True
                        )
                        
                        # Ensure child is RLCreature in RL mode
                        if self.use_rl and not isinstance(child, RLCreature):
                            # Convert to RL creature
                            child_genome = child.genome
                            child_body = child.body
                            child_shape = child.shape
                            # Remove old creature
                            self.space.remove(child_body, child_shape)
                            # Create RL creature
                            child = RLCreature(
                                self.space,
                                (child_body.position.x, child_body.position.y),
                                genome=child_genome,
                                use_rl=True
                            )
                        
                        self.creatures.append(child)
                        self.body_to_creature[child.body] = child
                        
                        # Update child's color based on species hue (if new species or existing)
                        try:
                            child_sid = child.genome.to_dict().get('species_id')
                            if child_sid is not None:
                                hue = self._existing_species.get(int(child_sid), child.genome.hue)
                                # Update genome hue if needed
                                if abs(child.genome.hue - hue) > 0.1:
                                    child.genome.hue = hue
                                # Update shape color immediately
                                from .creatures import hue_to_rgb
                                r, g, blue = hue_to_rgb(hue)
                                child.shape.color = (r, g, blue, 255)
                        except Exception:
                            pass
                        
                        # If it is a new species, automatically spawn a nearby clone to seed the population
                        if is_new_species:
                            # Prefer spacing by body size，以免重叠不明显
                            base = max(16.0, child.genome.body_radius * 2.5)
                            angle = random.uniform(0.0, 2.0 * math.pi)
                            dist = random.uniform(base, base + 24.0)
                            nx = mid_x + math.cos(angle) * dist
                            ny = mid_y + math.sin(angle) * dist
                            # Clamp to window to避免出界
                            nx = max(30.0, min(WINDOW_WIDTH - 30.0, nx))
                            ny = max(30.0, min(WINDOW_HEIGHT - 30.0, ny))
                            clone = RLCreature(
                                self.space,
                                (nx, ny),
                                genome=child_genome,
                                use_rl=True
                            )
                            # Update clone color to match new species
                            try:
                                clone_hue = child_genome.hue
                                from .creatures import hue_to_rgb
                                r, g, blue = hue_to_rgb(clone_hue)
                                clone.shape.color = (r, g, blue, 255)
                            except Exception:
                                pass
                            # 给予轻微初速度，避免与亲代重叠
                            jx = math.cos(angle) * 50.0
                            jy = math.sin(angle) * 50.0
                            clone.body.apply_impulse_at_local_point((jx, jy), (0, 0))
                            self.creatures.append(clone)
                            self.body_to_creature[clone.body] = clone
                            births += 1

                        # Fallback: if species id is new but divergence逻辑未触发，依然进行一次复制
                        if not is_new_species:
                            try:
                                sid = child.genome.to_dict().get('species_id', None)
                                if sid is not None:
                                    sid_int = int(sid)
                                    # Check if this is truly a new species (not in existing mapping)
                                    if sid_int not in self._existing_species:
                                        # Register this as new species with unique hue
                                        hue = child.genome.to_dict().get('hue', None)
                                        if hue is None or hue < 0.1:
                                            hue = self._generate_unique_hue()
                                        # Update child's genome hue
                                        child.genome.hue = float(hue)
                                        # Update shape color
                                        from .creatures import hue_to_rgb
                                        r, g, blue = hue_to_rgb(hue)
                                        child.shape.color = (r, g, blue, 255)
                                        # Register in mapping
                                        self._existing_species[sid_int] = float(hue)
                                        # Spawn clone if under population cap
                                        if len(self.creatures) < self.pop_cap:
                                            base = max(16.0, child.genome.body_radius * 2.5)
                                            angle = random.uniform(0.0, 2.0 * math.pi)
                                            dist = random.uniform(base, base + 24.0)
                                            nx = mid_x + math.cos(angle) * dist
                                            ny = mid_y + math.sin(angle) * dist
                                            nx = max(30.0, min(WINDOW_WIDTH - 30.0, nx))
                                            ny = max(30.0, min(WINDOW_HEIGHT - 30.0, ny))
                                            clone2 = RLCreature(self.space, (nx, ny), genome=child.genome, use_rl=True)
                                            # Update clone color too
                                            clone2.shape.color = (r, g, blue, 255)
                                            jx = math.cos(angle) * 50.0
                                            jy = math.sin(angle) * 50.0
                                            clone2.body.apply_impulse_at_local_point((jx, jy), (0, 0))
                                            self.creatures.append(clone2)
                                            self.body_to_creature[clone2.body] = clone2
                                            births += 1
                            except Exception as e:
                                print(f"[Debug] Fallback species check failed: {e}")
                                pass

                        # Track offspring for RL creatures
                        if isinstance(a, RLCreature) and a.use_rl:
                            a.episode_info['offspring_count'] = a.episode_info.get('offspring_count', 0) + 1
                        if isinstance(b, RLCreature) and b.use_rl:
                            b.episode_info['offspring_count'] = b.episode_info.get('offspring_count', 0) + 1
                        # Shorter cooldowns after breeding
                        a._breed_cd = random.uniform(2.0, 4.0)
                        b._breed_cd = random.uniform(2.0, 4.0)
                        births += 1
        
        # Update global births counter
        self.stats['births_total'] += births

    def _predation_encounters(self) -> None:
        """Check for predation opportunities between different species.
        
        Optimized with species comparison caching and early filtering.
        """
        # Pre-filter: only living creatures
        living = [c for c in self.creatures if not c.dead]
        if len(living) < 2:
            return
        
        n = len(living)
        for i in range(n):
            a = living[i]
            a_species_id = id(a.genome)
            
            for j in range(i + 1, n):
                b = living[j]
                
                # Quick species check using cache
                cache_key = (a_species_id, id(b.genome))
                if cache_key not in self._species_cache:
                    self._species_cache[cache_key] = is_same_species(a.genome.to_dict(), b.genome.to_dict())
                
                # Prevent same-species predation
                if self._species_cache[cache_key]:
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
                # Probability based on aggression vs prey caution (further reduced)
                p = 0.12 + 0.25 * predator.genome.aggression - 0.45 * prey.genome.caution
                if random.random() < max(0.02, min(0.70, p)):
                    # Track predation
                    self.stats['predations_total'] += 1
                    prey.dead = True
                    predator.devour(prey)
    
    def _generate_unique_hue(self, min_distance: float = 30.0) -> float:
        """Generate a unique hue that doesn't match existing species colors.
        
        Args:
            min_distance: Minimum hue distance from existing colors (degrees, default: 30.0)
            
        Returns:
            Unique hue value (0.0-360.0)
        """
        if not self._existing_species:
            # No existing species, return random hue
            return random.uniform(0.0, 360.0)
        
        existing_hues = list(self._existing_species.values())
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Try random hue
            candidate_hue = random.uniform(0.0, 360.0)
            
            # Check distance from all existing hues (circular distance)
            too_close = False
            for existing_hue in existing_hues:
                # Calculate circular distance (handle wrap-around at 360)
                diff = abs(candidate_hue - existing_hue)
                circular_diff = min(diff, 360.0 - diff)
                
                if circular_diff < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                return candidate_hue
        
        # If we couldn't find a unique hue, space them evenly
        # This happens when there are many species
        gap = 360.0 / (len(existing_hues) + 1)
        # Place new species at largest gap
        existing_sorted = sorted(existing_hues)
        max_gap = 0.0
        best_position = 0.0
        
        for i in range(len(existing_sorted)):
            next_hue = existing_sorted[(i + 1) % len(existing_sorted)]
            gap_size = (next_hue - existing_sorted[i]) % 360.0
            if gap_size > max_gap:
                max_gap = gap_size
                best_position = (existing_sorted[i] + gap_size / 2.0) % 360.0
        
        return best_position

    def _assign_new_species(self, child_gen: Dict[str, float]) -> Genome:
        """Assign a new species id and hue to a child genome and return Genome.
        
        Args:
            child_gen: Offspring genome dict to modify
        Returns:
            Genome with updated species_id and hue
        """
        new_id = self._next_species_id
        new_hue = self._generate_unique_hue()
        self._existing_species[new_id] = new_hue
        child_gen["species_id"] = float(new_id)
        child_gen["hue"] = new_hue
        self._next_species_id += 1
        return Genome(child_gen)

    def _aggregate_and_record_rl_stats(self) -> None:
        """Aggregate per-creature RL stats and record a visualizer step.
        
        Aggregates: average_reward, policy/value loss, entropy, clip_fraction,
        buffer size; prints a concise status line to console.
        """
        rl_creatures = [c for c in self.creatures if isinstance(c, RLCreature) and c.use_rl]
        if not rl_creatures:
            return
        
        total_trains = sum(c.train_counter for c in rl_creatures)
        avg_buffer_size = sum(len(c.experience_buffer) for c in rl_creatures) / len(rl_creatures)
        avg_reward = sum(c._total_lifetime_reward / max(1, c.age) for c in rl_creatures) / len(rl_creatures)
        recent_stats = [s for s in (c.get_recent_training_stats() for c in rl_creatures) if s]
        if recent_stats:
            mean_policy_loss = sum(s.get('policy_loss', 0.0) for s in recent_stats) / len(recent_stats)
            mean_value_loss = sum(s.get('value_loss', 0.0) for s in recent_stats) / len(recent_stats)
            mean_entropy = sum(s.get('entropy', 0.0) for s in recent_stats) / len(recent_stats)
            mean_clip = sum(s.get('clip_fraction', 0.0) for s in recent_stats) / len(recent_stats)
        else:
            mean_policy_loss = 0.0
            mean_value_loss = 0.0
            mean_entropy = 0.0
            mean_clip = 0.0
        
        latest_stats = {
            'average_reward': avg_reward,
            'policy_loss': mean_policy_loss,
            'value_loss': mean_value_loss,
            'entropy': mean_entropy,
            'clip_fraction': mean_clip,
            'buffer_size': int(avg_buffer_size),
        }
        self.rl_visualizer.record_training_step(latest_stats, total_trains)
        
        print(
            f"[RL] Independent Learning - Creatures: {len(rl_creatures)}, "
            f"Avg Buffer: {avg_buffer_size:.0f}, Avg Reward: {avg_reward:.3f}, "
            f"Policy: {mean_policy_loss:.3f}, Value: {mean_value_loss:.3f}, "
            f"Entropy: {mean_entropy:.3f}, Clip: {mean_clip:.3f}"
        )

    def _spawn_rl_creatures(self, count: int, area: Tuple[int, int, int, int]) -> List[Creature]:
        """Spawn RL-enabled creatures from three distinct species with high speed.
        
        Args:
            count: Number of creatures to spawn
            area: Tuple of (x0, y0, x1, y1) bounding box
            
        Returns:
            List of created RL creatures
        """
        if not self.use_rl or not RL_AVAILABLE:
            # Fallback to regular creatures (with 3 species, high speed)
            return spawn_creatures(self.space, count=count, area=area, num_species=3, high_speed=True)
        
        from .genetics import generate_base_species_genome, get_gene_def
        
        x0, y0, x1, y1 = area
        area_width = x1 - x0
        area_height = y1 - y0
        creatures: List[Creature] = []
        
        # Create base genomes for three species
        base_genomes = []
        for species_idx in range(3):
            base_genome = generate_base_species_genome(species_idx, high_speed=True)
            base_genomes.append(base_genome)
        
        # Distribute creatures evenly across species
        creatures_per_species = count // 3
        remainder = count % 3
        
        # Calculate cluster radius for species grouping
        cluster_radius = min(area_width, area_height) * 0.15  # 15% of smaller dimension
        
        for species_idx in range(3):
            num_this_species = creatures_per_species
            if species_idx < remainder:
                num_this_species += 1
            
            # Choose cluster center: left, center, or right region
            if species_idx == 0:
                center_x = x0 + area_width * 0.25
                center_y = y0 + area_height * 0.5
            elif species_idx == 1:
                center_x = x0 + area_width * 0.5
                center_y = y0 + area_height * 0.5
            else:  # species_idx == 2
                center_x = x0 + area_width * 0.75
                center_y = y0 + area_height * 0.5
            
            # Add small random offset
            center_x += random.uniform(-cluster_radius * 0.3, cluster_radius * 0.3)
            center_y += random.uniform(-cluster_radius * 0.3, cluster_radius * 0.3)
            center_x = max(x0 + cluster_radius, min(x1 - cluster_radius, center_x))
            center_y = max(y0 + cluster_radius, min(y1 - cluster_radius, center_y))
            
            base_genome = base_genomes[species_idx]
            
            # Spawn in cluster around center
            for _ in range(num_this_species):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, cluster_radius * 0.8)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                x = max(x0, min(x1, x))
                y = max(y0, min(y1, y))
                
                # Add small random variations to base genome
                varied_genome = base_genome.copy()
                for gene_name in varied_genome.keys():
                    if gene_name == "species_id":
                        continue
                    variation = random.uniform(-0.05, 0.05)
                    gene_def = get_gene_def(gene_name)
                    if gene_def:
                        min_val, max_val = gene_def.default_range
                        new_val = varied_genome[gene_name] * (1.0 + variation)
                        varied_genome[gene_name] = max(min_val, min(max_val, new_val))
                
                creature = RLCreature(
                    self.space,
                    (x, y),
                    genome=Genome(varied_genome),
                    use_rl=True
                )
                # Ensure color matches species_id (use hue from _existing_species mapping)
                try:
                    sid = int(varied_genome.get('species_id', species_idx))
                    if sid in self._existing_species:
                        hue = self._existing_species[sid]
                        creature.genome.hue = hue
                        from .creatures import hue_to_rgb
                        r, g, blue = hue_to_rgb(hue)
                        creature.shape.color = (r, g, blue, 255)
                except Exception:
                    pass
                creatures.append(creature)
                self.body_to_creature[creature.body] = creature
        
        return creatures
    
    def _cache_terrain_centers(self):
        """Cache terrain centers and approximate bounding boxes for fast lookup."""
        self._terrain_centers_cache.clear()
        self._terrain_bbox_cache.clear()
        
        for terrain_type, shapes in self.terrain_shapes.items():
            for shape in shapes:
                cx = shape.body.position.x
                cy = shape.body.position.y
                self._terrain_centers_cache[shape] = (cx, cy)
                
                # Approximate bounding box (using shape radius or size estimate)
                if isinstance(shape, pymunk.Circle):
                    radius = shape.radius
                else:
                    # Estimate from polygon vertices
                    verts = shape.get_vertices()
                    if verts:
                        max_dist = max(math.sqrt(v.x**2 + v.y**2) for v in verts)
                        radius = max_dist * 1.2  # Add 20% margin
                    else:
                        radius = 100.0  # Default estimate
                
                # Cache: (center_x, center_y, radius)
                self._terrain_bbox_cache[shape] = (cx, cy, radius)
    
    def _update_statistics(self):
        """Update statistics for analysis."""
        if len(self.creatures) == 0:
            return
        
        # Current time step (approximate)
        time_step = len(self.stats['time_steps']) * self.stats_update_interval
        
        # Population
        self.stats['time_steps'].append(time_step)
        self.stats['population'].append(len(self.creatures))
        
        # Average speed and age
        speeds = [math.sqrt(c.body.velocity.x**2 + c.body.velocity.y**2) for c in self.creatures if not c.dead]
        ages = [c.age for c in self.creatures if not c.dead]
        if speeds:
            self.stats['avg_speed'].append(sum(speeds) / len(speeds))
        if ages:
            self.stats['avg_age'].append(sum(ages) / len(ages))
        
        # Species count (approximate by counting different genome ids)
        species_ids = set()
        for c in self.creatures:
            if not c.dead:
                # Use a simplified species identifier (first few key genes)
                genome_dict = c.genome.to_dict()
                species_key = (
                    round(genome_dict.get('aggression', 0.5), 1),
                    round(genome_dict.get('curiosity', 0.5), 1),
                    round(genome_dict.get('sociability', 0.5), 1),
                )
                species_ids.add(species_key)
        self.stats['species_count'].append(len(species_ids))
        
        # Terrain resource and hazard levels
        for terrain_type, shapes in self.terrain_shapes.items():
            if not shapes:
                continue
            # Sample a few shapes to get average
            sample_shapes = list(shapes)[:min(5, len(shapes))]
            avg_resource = 0.0
            avg_hazard = 0.0
            count = 0
            for shape in sample_shapes:
                state = self.terrain_dynamic_state.get(shape, {})
                if state:
                    avg_resource += state.get('current_resource', 0.0)
                    avg_hazard += state.get('current_hazard', 0.0)
                    count += 1
            if count > 0:
                self.stats['terrain_resources'][terrain_type].append(avg_resource / count)
                self.stats['terrain_hazards'][terrain_type].append(avg_hazard / count)
    
    def get_statistics(self) -> Dict:
        """Get collected statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
    
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
        overlay.fill((0, 0, 0, 0))  # ensure fully transparent start

        # Draw terrains with transparency
        # Draw all terrain shapes (both sensor and non-sensor), but skip creature shapes
        creature_bodies = {c.body for c in self.creatures}
        
        # Compute hazard distribution percentiles (Q10, Q90) for robust normalization
        hazard_values = []
        for shape_iter in self.space.shapes:
            if shape_iter.body in creature_bodies:
                continue
            if not hasattr(shape_iter, "color"):
                continue
            hv = getattr(shape_iter, 'hazard_level', None)
            if hv is not None:
                try:
                    hazard_values.append(max(0.0, float(hv)))
                except Exception:
                    pass
        q10 = 0.0
        q90 = 1.0
        if hazard_values:
            vals = sorted(hazard_values)
            n = len(vals)
            i10 = max(0, min(n - 1, int(0.10 * (n - 1))))
            i90 = max(0, min(n - 1, int(0.90 * (n - 1))))
            q10 = vals[i10]
            q90 = vals[i90]
            if q90 - q10 < 1e-6:
                q10, q90 = 0.0, max(1.0, q90)

        for shape in self.space.shapes:
            # Skip creature shapes (they are drawn separately)
            if shape.body in creature_bodies:
                continue
            
            # Only draw shapes that have a color attribute (terrain shapes)
            if not hasattr(shape, "color"):
                continue
            # Skip walls/segments to avoid artifacts
            if isinstance(shape, pymunk.Segment):
                continue
            
            color = shape.color
            if not isinstance(color, (tuple, list)) or len(color) < 4:
                continue
            
            # Visualize dynamic resource/hazard: modulate brightness
            resource_val = getattr(shape, 'resource_value', 0.0)
            hazard_val = getattr(shape, 'hazard_level', 0.0)
            # Normalize values: resource linear; hazard via percentile-based normalization
            res_n = max(0.0, min(1.0, resource_val * 1.5))
            if q90 > q10:
                haz_n = (float(hazard_val) - q10) / (q90 - q10)
            else:
                haz_n = float(hazard_val)
            haz_n = max(0.0, min(1.0, haz_n))
            # Higher contrast mapping: more resource -> much brighter, hazard -> darker
            raw = 0.5 + 0.8 * (res_n - 0.6 * haz_n)
            factor = max(0.2, min(1.25, raw))
            r, g, b, a = color
            mod_color = (
                min(255, int(r * factor)),
                min(255, int(g * factor)),
                min(255, int(b * factor)),
                a,
            )
            
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
                pygame.draw.polygon(overlay, mod_color, pts, 0)
                # Draw outline with higher alpha
                outline_color = (mod_color[0], mod_color[1], mod_color[2], min(255, mod_color[3] + 40))
                pygame.draw.polygon(overlay, outline_color, pts, 2)

                # --- Hatch overlays: resource (horizontal), hazard (diagonal) ---
                # Build local rect
                min_x = min(p[0] for p in pts)
                max_x = max(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_y = max(p[1] for p in pts)
                rect_w = max(1, int(max_x - min_x))
                rect_h = max(1, int(max_y - min_y))

                # Skip off-screen minimal rects
                if rect_w > 2 and rect_h > 2:
                    # Create mask surface for polygon
                    poly_mask_surf = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                    # Draw polygon in local coords
                    local_pts = [(int(p[0] - min_x), int(p[1] - min_y)) for p in pts]
                    pygame.draw.polygon(poly_mask_surf, (255, 255, 255, 255), local_pts, 0)
                    poly_mask = pygame.mask.from_surface(poly_mask_surf)

                    # Resource hatch: horizontal lines
                    res_alpha = int(90 + 140 * res_n)  # 90..230
                    res_spacing = int(18 - 12 * res_n)  # 18..6
                    if res_n > 0.02 and res_alpha > 0 and res_spacing >= 4:
                        hatch_res = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                        y = 0
                        while y < rect_h:
                            pygame.draw.line(hatch_res, (255, 255, 255, res_alpha), (0, y), (rect_w, y), 1)
                            y += res_spacing
                        # Clip hatch to polygon mask
                        mask_surface = poly_mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
                        hatch_res.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                        overlay.blit(hatch_res, (int(min_x), int(min_y)), special_flags=pygame.BLEND_RGBA_ADD)

                    # Hazard hatch: 45-degree diagonal lines
                    haz_alpha = int(120 + 135 * haz_n)  # 120..255
                    haz_spacing = int(20 - 12 * haz_n)  # 20..8
                    if haz_n > 0.02 and haz_alpha > 0 and haz_spacing >= 4:
                        hatch_haz = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                        # Draw diagonals by tiling lines across extended range
                        # We draw lines with slope +1 across the rect
                        # Start from negative offset to cover fully
                        start = -rect_h
                        end = rect_w
                        x = start
                        while x < end + rect_h:
                            p1 = (x, 0)
                            p2 = (x + rect_h, rect_h)
                            pygame.draw.line(hatch_haz, (255, 0, 0, haz_alpha), p1, p2, 1)
                            x += haz_spacing
                        # Clip to polygon
                        mask_surface = poly_mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
                        hatch_haz.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                        overlay.blit(hatch_haz, (int(min_x), int(min_y)), special_flags=pygame.BLEND_RGBA_ADD)
            
            elif isinstance(shape, pymunk.Circle):
                bx, by = shape.body.position.x, shape.body.position.y
                pygame.draw.circle(overlay, mod_color, (int(bx), int(by)), int(shape.radius))
                # Draw outline with higher alpha
                outline_color = (mod_color[0], mod_color[1], mod_color[2], min(255, mod_color[3] + 40))
                pygame.draw.circle(overlay, outline_color, (int(bx), int(by)), int(shape.radius), 2)

                # --- Hatch overlays for circle ---
                rect_w = rect_h = int(shape.radius * 2)
                if rect_w > 2 and rect_h > 2:
                    min_x = int(bx - shape.radius)
                    min_y = int(by - shape.radius)
                    circle_mask_surf = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                    pygame.draw.circle(circle_mask_surf, (255, 255, 255, 255), (int(shape.radius), int(shape.radius)), int(shape.radius))
                    circle_mask = pygame.mask.from_surface(circle_mask_surf)

                    # Resource horizontal lines
                    res_alpha = int(90 + 140 * res_n)
                    res_spacing = int(18 - 12 * res_n)
                    if res_n > 0.02 and res_alpha > 0 and res_spacing >= 4:
                        hatch_res = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                        y = 0
                        while y < rect_h:
                            pygame.draw.line(hatch_res, (255, 255, 255, res_alpha), (0, y), (rect_w, y), 1)
                            y += res_spacing
                        mask_surface = circle_mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
                        hatch_res.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                        overlay.blit(hatch_res, (min_x, min_y), special_flags=pygame.BLEND_RGBA_ADD)

                    # Hazard diagonal lines (45 degrees)
                    haz_alpha = int(120 + 135 * haz_n)
                    haz_spacing = int(20 - 12 * haz_n)
                    if haz_n > 0.02 and haz_alpha > 0 and haz_spacing >= 4:
                        hatch_haz = pygame.Surface((rect_w, rect_h), pygame.SRCALPHA)
                        start = -rect_h
                        end = rect_w
                        x = start
                        while x < end + rect_h:
                            p1 = (x, 0)
                            p2 = (x + rect_h, rect_h)
                            pygame.draw.line(hatch_haz, (255, 0, 0, haz_alpha), p1, p2, 1)
                            x += haz_spacing
                        mask_surface = circle_mask.to_surface(setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0))
                        hatch_haz.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                        overlay.blit(hatch_haz, (min_x, min_y), special_flags=pygame.BLEND_RGBA_ADD)
        
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
        
        # Realtime population/species counters
        alive_count = sum(1 for c in self.creatures if not c.dead)
        # Prefer species_id if present; fallback to approximate key
        species_ids = set()
        for c in self.creatures:
            if c.dead:
                continue
            g = c.genome.to_dict()
            sid = g.get('species_id')
            if sid is None:
                sid = (
                    round(g.get('aggression', 0.5), 1),
                    round(g.get('curiosity', 0.5), 1),
                    round(g.get('sociability', 0.5), 1),
                )
            species_ids.add(sid)
        lines.append("")
        lines.append("Population:")
        lines.append(f"  Current: {alive_count} / Cap {self.pop_cap}")
        lines.append(f"  Species: {len(species_ids)}")
        lines.append(f"  Births: {self.stats['births_total']}  Deaths: {self.stats['deaths_total']}  Predations: {self.stats['predations_total']}")
        
        # Add RL statistics if RL mode is enabled
        if self.use_rl and self.rl_visualizer:
            rl_stats = self.rl_visualizer.get_recent_statistics()
            lines.append("")
            lines.append("RL Training:")
            if rl_stats['total_updates'] > 0:
                lines.append(f"  Updates: {rl_stats['total_updates']}")
                lines.append(f"  Avg Reward: {rl_stats['avg_reward']:.2f}")
                if rl_stats['recent_episode_reward'] > 0:
                    lines.append(f"  Episode Reward: {rl_stats['recent_episode_reward']:.2f}")
        
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
            
            # RL training and reporting (independent learning)
            if self.use_rl:
                self._rl_train_counter += 1
                
                # Aggregate statistics from all creatures (for visualization)
                if self.rl_visualizer and self._rl_train_counter % self._rl_report_interval == 0:
                    self._aggregate_and_record_rl_stats()
                
                # Periodic evolution tracking (every 1000 steps)
                if self.evolution_supervisor and self._rl_train_counter % self._evolution_record_interval == 0:
                    # Update fitness tracking for all alive RL creatures
                    rl_creatures = [c for c in self.creatures 
                                  if isinstance(c, RLCreature) and c.use_rl]
                    for creature in rl_creatures:
                        fitness_metrics = creature.get_fitness_metrics()
                        self.evolution_supervisor.track_fitness(creature.id, fitness_metrics)
                    
                    # Record generation statistics
                    self.evolution_supervisor.record_generation(rl_creatures)
                    
                    # Print evolution stats occasionally
                    if self._rl_train_counter % self._evolution_verbose_interval == 0:
                        evo_stats = self.evolution_supervisor.get_evolution_stats()
                        health = self.evolution_supervisor.get_population_health(rl_creatures)
                        # Cache current diversity for RL global bonus module
                        self._current_diversity = float(health.get('diversity', 0.0))
                        print(f"[Evolution] Gen: {evo_stats['generation']}, "
                              f"Fitness: {health['avg_fitness']:.2f}, "
                              f"Diversity: {health['diversity']:.3f}, "
                              f"Lifespan: {health['avg_lifespan']:.1f}s")
                        
                        # Record episode stats from RL creatures
                        if self.rl_visualizer:
                            for c in self.creatures:
                                if isinstance(c, RLCreature) and c.use_rl and hasattr(c, 'episode_reward'):
                                    if c.episode_steps > 0:
                                        self.rl_visualizer.record_episode(c.episode_reward, c.episode_steps)
            self._breed_encounters()
            self._predation_encounters()
            self._cull_dead()
            self._maybe_rebuild_terrains(dt)
            
            # Update statistics periodically
            self.stats_timer += dt
            if self.stats_timer >= self.stats_update_interval:
                self._update_statistics()
                self.stats_timer = 0.0
            
            # Clear species cache periodically to prevent memory growth
            if hasattr(self, '_species_cache_clear_timer'):
                self._species_cache_clear_timer += dt
                if self._species_cache_clear_timer >= 10.0:  # Clear every 10 seconds
                    self._species_cache_clear_timer = 0.0
                    # Keep cache size reasonable (keep last 1000 entries)
                    if len(self._species_cache) > 1000:
                        # Clear half of cache (simple FIFO replacement)
                        keys_to_remove = list(self._species_cache.keys())[:500]
                        for key in keys_to_remove:
                            del self._species_cache[key]
            else:
                self._species_cache_clear_timer = 0.0

            self.space.step(dt)

            self.screen.fill(BACKGROUND_COLOR)
            if self.use_debug_draw:
                self.space.debug_draw(self.draw_options)
            else:
                self._draw_custom()
            self._draw_hud()

            pygame.display.flip()
            self.clock.tick(FPS)

