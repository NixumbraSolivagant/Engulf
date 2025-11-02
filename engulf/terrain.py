"""Terrain generation and management for the simulation."""

from typing import Iterable, Tuple

import math
import random

import pymunk

from .settings import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    COLOR_WALL,
    COLOR_ICE,
    COLOR_SAND,
    COLOR_WATER,
    COLOR_BOUNCE,
    COLLTYPE_WATER_SENSOR,
    COLOR_MUD,
    COLOR_BOOST,
    COLOR_TOXIC,
    COLOR_MEADOW,
    COLOR_CRYSTAL,
    COLOR_FOREST,
    COLOR_GEYSER,
    COLOR_LAVA,
    COLOR_SPIKE,
    COLOR_STORM,
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


def add_segment(
    space: pymunk.Space,
    a: Tuple[float, float],
    b: Tuple[float, float],
    radius: float,
    friction: float,
    elasticity: float,
    color: Tuple[int, int, int, int],
) -> pymunk.Segment:
    """Add a segment (wall) to the physics space.

    Args:
        space: Pymunk physics space
        a: Start point
        b: End point
        radius: Segment radius
        friction: Friction coefficient
        elasticity: Elasticity coefficient
        color: RGBA color tuple

    Returns:
        Created segment shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, a, b, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.color = color
    space.add(body, shape)
    return shape


essential_walls = (
    ((20, 20), (WINDOW_WIDTH - 20, 20)),      # top
    ((WINDOW_WIDTH - 20, 20), (WINDOW_WIDTH - 20, WINDOW_HEIGHT - 20)),  # right
    ((20, WINDOW_HEIGHT - 20), (WINDOW_WIDTH - 20, WINDOW_HEIGHT - 20)),  # bottom
    ((20, 20), (20, WINDOW_HEIGHT - 20)),      # left
)


def build_walls(space: pymunk.Space, cfg: Config) -> None:
    """Build boundary walls around the simulation area.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
    """
    for a, b in essential_walls:
        add_segment(space, a, b, radius=cfg.wall_radius, friction=0.9, elasticity=0.2, color=COLOR_WALL)


def add_poly(
    space: pymunk.Space,
    verts: Iterable[Tuple[float, float]],
    pos: Tuple[float, float],
    friction: float,
    elasticity: float,
    color: Tuple[int, int, int, int],
    sensor: bool = False,
    collision_type: int = 0,
    angle: float = 0.0,
) -> pymunk.Poly:
    """Add a polygon shape to the physics space.

    Args:
        space: Pymunk physics space
        verts: Polygon vertices (relative to center)
        pos: Center position
        friction: Friction coefficient
        elasticity: Elasticity coefficient
        color: RGBA color tuple
        sensor: Whether shape is a sensor (no collision response)
        collision_type: Collision type identifier
        angle: Rotation angle in radians

    Returns:
        Created polygon shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    body.angle = angle
    poly = pymunk.Poly(body, verts)
    poly.friction = friction
    poly.elasticity = elasticity
    poly.color = color
    poly.sensor = sensor
    poly.collision_type = collision_type
    space.add(body, poly)
    return poly


def build_ice(space: pymunk.Space, cfg: Config):
    w, h = 360, 180
    cx, cy = 320, 200
    verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    add_poly(space, verts, (cx, cy), friction=cfg.ice_friction, elasticity=cfg.ice_elasticity, color=COLOR_ICE)


def build_sand(space: pymunk.Space, cfg: Config):
    verts = [(-180, -60), (140, -40), (160, 30), (60, 80), (-140, 50), (-160, -10)]
    pos = (860, 220)
    add_poly(space, verts, pos, friction=cfg.sand_friction, elasticity=cfg.sand_elasticity, color=COLOR_SAND)


def build_bounce(space: pymunk.Space, cfg: Config):
    pads = [
        ((520, 540), 100, 40, 0.0),
        ((820, 520), 120, 30, 0.2),
    ]
    for (cx, cy), w, h, angle in pads:
        verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        body.angle = angle
        poly = pymunk.Poly(body, verts)
        poly.friction = cfg.bounce_friction
        poly.elasticity = cfg.bounce_elasticity
        poly.color = COLOR_BOUNCE
        space.add(body, poly)


def build_water(space: pymunk.Space, cfg: Config):
    w, h = 300, 220
    cx, cy = 360, 500
    verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    add_poly(space, verts, (cx, cy), friction=0.2, elasticity=0.0, color=COLOR_WATER,
             sensor=True, collision_type=COLLTYPE_WATER_SENSOR)


def build_effect_terrains(space: pymunk.Space):
    verts = [(-200, -80), (200, -60), (180, 80), (-220, 70)]
    add_poly(space, verts, (260, 360), friction=0.9, elasticity=0.0, color=COLOR_MUD,
             sensor=True, collision_type=COLLTYPE_TERRAIN_MUD)
    w, h = 220, 120
    verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    add_poly(space, verts, (880, 420), friction=0.9, elasticity=0.0, color=COLOR_BOOST,
             sensor=True, collision_type=COLLTYPE_TERRAIN_BOOST)
    w2, h2 = 260, 160
    verts2 = [(-w2/2, -h2/2), (w2/2, -h2/2), (w2/2, h2/2), (-w2/2, h2/2)]
    add_poly(space, verts2, (640, 200), friction=0.9, elasticity=0.0, color=COLOR_TOXIC,
             sensor=True, collision_type=COLLTYPE_TERRAIN_TOXIC)
    verts3 = [(-180, -90), (180, -70), (200, 60), (-160, 70)]
    add_poly(space, verts3, (980, 280), friction=0.9, elasticity=0.0, color=COLOR_MEADOW,
             sensor=True, collision_type=COLLTYPE_TERRAIN_MEADOW)


def build_all(space: pymunk.Space, cfg: Config):
    build_walls(space, cfg)
    build_ice(space, cfg)
    build_sand(space, cfg)
    build_bounce(space, cfg)
    build_water(space, cfg)
    build_effect_terrains(space)


# Randomized builders

def _rand_rect(wmin: float, wmax: float, hmin: float, hmax: float) -> list[Tuple[float, float]]:
    """Generate random rectangle vertices.

    Args:
        wmin: Minimum width
        wmax: Maximum width
        hmin: Minimum height
        hmax: Maximum height

    Returns:
        List of 4 vertices forming a rectangle
    """
    w = random.uniform(wmin, wmax)
    h = random.uniform(hmin, hmax)
    return [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]


def build_random_ice(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random ice patches and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        verts = _rand_rect(200, 420, 120, 220)
        cx = rng.uniform(120, WINDOW_WIDTH - 120)
        cy = rng.uniform(120, WINDOW_HEIGHT - 120)
        ang = rng.uniform(-math.pi/8, math.pi/8)
        poly = add_poly(space, verts, (cx, cy), friction=cfg.ice_friction, elasticity=cfg.ice_elasticity,
                        color=COLOR_ICE, angle=ang)
        shapes.append(poly)
    return shapes


def build_random_sand(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Circle]:
    """Build random sand areas as circles and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        radius = rng.uniform(80, 150)
        cx = rng.uniform(120, WINDOW_WIDTH - 120)
        cy = rng.uniform(120, WINDOW_HEIGHT - 120)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        circle = pymunk.Circle(body, radius)
        circle.friction = cfg.sand_friction
        circle.elasticity = cfg.sand_elasticity
        circle.color = COLOR_SAND
        space.add(body, circle)
        shapes.append(circle)
    return shapes


def build_random_bounce(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random bounce pads and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        w = rng.uniform(120, 180)
        h = rng.uniform(40, 80)
        verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        cx = rng.uniform(160, WINDOW_WIDTH - 160)
        cy = rng.uniform(160, WINDOW_HEIGHT - 160)
        ang = rng.uniform(-math.pi/8, math.pi/8)
        poly = add_poly(space, verts, (cx, cy), friction=cfg.bounce_friction, elasticity=cfg.bounce_elasticity,
                        color=COLOR_BOUNCE, angle=ang)
        shapes.append(poly)
    return shapes


def build_random_water(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random water areas as ellipses (polygon approximation) and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        # Create ellipse as polygon with many vertices
        rx = rng.uniform(110, 190)
        ry = rng.uniform(70, 120)
        num_points = 16
        verts = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = rx * math.cos(angle)
            y = ry * math.sin(angle)
            verts.append((x, y))
        cx = rng.uniform(140, WINDOW_WIDTH - 140)
        cy = rng.uniform(140, WINDOW_HEIGHT - 140)
        poly = add_poly(space, verts, (cx, cy), friction=0.2, elasticity=0.0, color=COLOR_WATER,
                        sensor=True, collision_type=COLLTYPE_WATER_SENSOR)
        shapes.append(poly)
    return shapes


def build_random_mud(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random mud areas and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        verts = _rand_rect(220, 360, 100, 200)
        cx = rng.uniform(140, WINDOW_WIDTH - 140)
        cy = rng.uniform(140, WINDOW_HEIGHT - 140)
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0, color=COLOR_MUD,
                        sensor=True, collision_type=COLLTYPE_TERRAIN_MUD, angle=rng.uniform(-0.4, 0.4))
        shapes.append(poly)
    return shapes


def build_random_boost(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random boost areas as stars and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        # Create star shape (5-pointed star)
        outer_radius = rng.uniform(80, 120)
        inner_radius = outer_radius * 0.5
        num_points = 5
        verts = []
        for i in range(num_points * 2):
            angle = (i / (num_points * 2)) * 2 * math.pi - math.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            verts.append((x, y))
        cx = rng.uniform(140, WINDOW_WIDTH - 140)
        cy = rng.uniform(140, WINDOW_HEIGHT - 140)
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0, color=COLOR_BOOST,
                        sensor=True, collision_type=COLLTYPE_TERRAIN_BOOST, angle=rng.uniform(-0.4, 0.4))
        shapes.append(poly)
    return shapes


def build_random_toxic(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random toxic areas as triangles and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        # Create triangle
        size = rng.uniform(120, 180)
        verts = [
            (0, -size * 0.866),
            (-size * 0.5, size * 0.433),
            (size * 0.5, size * 0.433),
        ]
        cx = rng.uniform(140, WINDOW_WIDTH - 140)
        cy = rng.uniform(140, WINDOW_HEIGHT - 140)
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0, color=COLOR_TOXIC,
                        sensor=True, collision_type=COLLTYPE_TERRAIN_TOXIC, angle=rng.uniform(-0.4, 0.4))
        shapes.append(poly)
    return shapes


def build_random_meadow(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list[pymunk.Poly]:
    """Build random meadow areas as hexagons and return list of shapes."""
    rng = random.Random(seed)
    shapes = []
    for _ in range(rng.randint(1, 2)):
        # Create hexagon
        radius = rng.uniform(80, 120)
        verts = []
        for i in range(6):
            angle = (i / 6) * 2 * math.pi - math.pi / 6
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            verts.append((x, y))
        cx = rng.uniform(140, WINDOW_WIDTH - 140)
        cy = rng.uniform(140, WINDOW_HEIGHT - 140)
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0, color=COLOR_MEADOW,
                        sensor=True, collision_type=COLLTYPE_TERRAIN_MEADOW, angle=rng.uniform(-0.4, 0.4))
        shapes.append(poly)
    return shapes


def build_all_random(space: pymunk.Space, cfg: Config, seed: int | None = None) -> None:
    """Build all terrain types randomly using hexagonal grid for full coverage.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
        seed: Random seed (None for random)
    """
    rng = random.Random(seed)
    build_walls(space, cfg)
    # Use hexagonal grid (honeycomb pattern) - each hex gets one terrain type
    # Use larger grid for better coverage
    build_tiled_terrain(space, cfg, 'all', 12, 8, seed)


def build_tiled_terrain(space: pymunk.Space, cfg: Config, terrain_type: str,
                        grid_cols: int = 6, grid_rows: int = 4, seed: int | None = None) -> list:
    """Build terrain using hexagonal grid (honeycomb pattern) for full coverage.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
        terrain_type: Type of terrain ('ice', 'sand', etc., or 'all' for random mix)
        grid_cols: Approximate number of columns
        grid_rows: Approximate number of rows
        seed: Random seed

    Returns:
        List of created terrain shapes
    """
    rng = random.Random(seed)
    shapes = []
    
    margin = 20
    playable_width = WINDOW_WIDTH - 2 * margin
    playable_height = WINDOW_HEIGHT - 2 * margin
    
    # All terrain types to randomly select from
    all_terrain_types = ['ice', 'sand', 'bounce', 'water', 'mud', 'boost', 'toxic', 'meadow',
                         'crystal', 'forest', 'geyser', 'lava', 'spike', 'storm']
    
    # Calculate hexagon size for flat-top hexagonal tiling
    # Horizontal spacing: sqrt(3) * radius
    # Vertical spacing: 1.5 * radius
    # Use smaller radius to ensure full coverage with overlap
    hex_radius_w = playable_width / ((grid_cols + 0.5) * math.sqrt(3))
    hex_radius_h = playable_height / ((grid_rows + 0.5) * 1.5)
    hex_radius = min(hex_radius_w, hex_radius_h) * 1.05  # Slightly larger for overlap
    
    # Generate hexagonal grid with honeycomb pattern
    hex_width = hex_radius * math.sqrt(3)  # Horizontal spacing
    hex_height = hex_radius * 1.5  # Vertical spacing
    
    # Calculate how many rows and cols we need to fully cover
    num_rows = int(math.ceil(playable_height / hex_height)) + 2
    num_cols_base = int(math.ceil(playable_width / hex_width)) + 2
    
    # Generate hexagonal grid - start from negative margin to ensure edge coverage
    start_y = margin - hex_radius
    start_x = margin - hex_radius
    
    for row in range(num_rows + 1):
        # Calculate number of hexagons in this row (odd rows have one more in offset pattern)
        cols_in_row = num_cols_base + 1 if row % 2 == 0 else num_cols_base + 2
        
        for col in range(cols_in_row):
            # Offset odd rows for honeycomb pattern
            offset_x = hex_width / 2 if (row % 2 == 1) else 0
            
            # Calculate hexagon center
            cx = start_x + col * hex_width + offset_x
            cy = start_y + row * hex_height + hex_radius
            
            # Only skip if completely outside screen (allow overlap at edges)
            if cx + hex_radius < 0 or cx - hex_radius > WINDOW_WIDTH:
                continue
            if cy + hex_radius < 0 or cy - hex_radius > WINDOW_HEIGHT:
                continue
            
            # Randomly assign terrain type
            patch_terrain = rng.choice(all_terrain_types) if terrain_type == 'all' else terrain_type
            
            # Create hexagon shape
            shape = _create_hexagon_terrain(
                space, cfg, patch_terrain, (cx, cy), hex_radius, rng
            )
            if shape:
                shapes.append(shape)
                shape.terrain_type = patch_terrain
    
    return shapes


# Terrain resource and hazard definitions
# Each terrain type has: (resource_value_per_sec, hazard_level)
# resource_value: growth/speed/breeding benefits per second while on terrain
# hazard_level: probability of negative effects per second while on terrain
# Base terrain properties (resource_value, hazard_level)
# These are base values that will be modified by dynamic systems
# Base terrain properties (resource_value, hazard_level)
# Optimized for better resource gameplay - higher values, clearer distinction
TERRAIN_PROPERTIES = {
    # Neutral/Low Resource Terrains
    'ice': (0.02, 0.03),      # Minimal resources, reduced hazard (from 0.05 to 0.03)
    'sand': (0.25, 0.0),      # Moderate resources, no hazard
    'bounce': (0.12, 0.0),    # Low resources, no hazard
    
    # Moderate Resource Terrains
    'water': (0.35, 0.06),    # Good resources, reduced hazard (from 0.10 to 0.06)
    'mud': (0.18, 0.05),      # Moderate resources, reduced hazard (from 0.08 to 0.05)
    'boost': (0.30, 0.0),     # Good resources, no hazard
    'forest': (0.40, 0.03),   # Good resources, reduced hazard (from 0.05 to 0.03)
    
    # High Resource Terrains (Resource-Type)
    'meadow': (0.55, 0.0),    # Excellent resources, no hazard
    'crystal': (0.65, 0.0),   # Very high resources, no hazard
    'geyser': (0.45, 0.10),   # High resources, reduced hazard (from 0.15 to 0.10)
    
    # Hazard-Only Terrains (Hazard-Type) - Reduced hazards for better survival
    'toxic': (0.05, 0.25),    # Minimal resources, reduced hazard (from 0.35 to 0.25)
    'lava': (0.0, 0.30),      # No resources, reduced hazard (from 0.45 to 0.30)
    'spike': (0.0, 0.28),     # No resources, reduced hazard (from 0.40 to 0.28)
    'storm': (0.05, 0.25),    # Minimal resources, reduced hazard (from 0.35 to 0.25)
}


def _create_hexagon_terrain(space: pymunk.Space, cfg: Config, terrain_type: str,
                            pos: Tuple[float, float], radius: float,
                            rng: random.Random) -> pymunk.Shape | None:
    """Create a hexagon-shaped terrain with resource and hazard properties.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
        terrain_type: Type of terrain
        pos: Center position
        radius: Hexagon radius (center to vertex)
        rng: Random number generator

    Returns:
        Created terrain shape or None
    """
    cx, cy = pos
    
    # Create regular hexagon vertices (flat-top)
    verts = []
    for i in range(6):
        angle = (i / 6) * 2 * math.pi - math.pi / 6  # Start from top-left for flat-top
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append((x, y))
    
    # Small random rotation for visual variation
    ang = rng.uniform(-math.pi / 12, math.pi / 12)
    
    # Get terrain properties (resource value, hazard level)
    resource_value, hazard_level = TERRAIN_PROPERTIES.get(terrain_type, (0.0, 0.0))
    
    if terrain_type == 'ice':
        poly = add_poly(space, verts, (cx, cy), friction=cfg.ice_friction,
                        elasticity=cfg.ice_elasticity, color=COLOR_ICE, angle=ang)
        poly.terrain_type = 'ice'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'sand':
        poly = add_poly(space, verts, (cx, cy), friction=cfg.sand_friction,
                        elasticity=cfg.sand_elasticity, color=COLOR_SAND, angle=ang)
        poly.terrain_type = 'sand'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'bounce':
        poly = add_poly(space, verts, (cx, cy), friction=cfg.bounce_friction,
                        elasticity=cfg.bounce_elasticity, color=COLOR_BOUNCE, angle=ang)
        poly.terrain_type = 'bounce'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'water':
        poly = add_poly(space, verts, (cx, cy), friction=0.2, elasticity=0.0,
                        color=COLOR_WATER, sensor=True, collision_type=COLLTYPE_WATER_SENSOR, angle=ang)
        poly.terrain_type = 'water'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'mud':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MUD, sensor=True, collision_type=COLLTYPE_TERRAIN_MUD, angle=ang)
        poly.terrain_type = 'mud'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'boost':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_BOOST, sensor=True, collision_type=COLLTYPE_TERRAIN_BOOST, angle=ang)
        poly.terrain_type = 'boost'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'toxic':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_TOXIC, sensor=True, collision_type=COLLTYPE_TERRAIN_TOXIC, angle=ang)
        poly.terrain_type = 'toxic'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'meadow':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MEADOW, sensor=True, collision_type=COLLTYPE_TERRAIN_MEADOW, angle=ang)
        poly.terrain_type = 'meadow'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'crystal':
        poly = add_poly(space, verts, (cx, cy), friction=0.1, elasticity=0.3,
                        color=COLOR_CRYSTAL, sensor=True, collision_type=COLLTYPE_TERRAIN_CRYSTAL, angle=ang)
        poly.terrain_type = 'crystal'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'forest':
        poly = add_poly(space, verts, (cx, cy), friction=0.7, elasticity=0.0,
                        color=COLOR_FOREST, sensor=True, collision_type=COLLTYPE_TERRAIN_FOREST, angle=ang)
        poly.terrain_type = 'forest'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'geyser':
        poly = add_poly(space, verts, (cx, cy), friction=0.5, elasticity=0.1,
                        color=COLOR_GEYSER, sensor=True, collision_type=COLLTYPE_TERRAIN_GEYSER, angle=ang)
        poly.terrain_type = 'geyser'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'lava':
        poly = add_poly(space, verts, (cx, cy), friction=0.3, elasticity=0.0,
                        color=COLOR_LAVA, sensor=True, collision_type=COLLTYPE_TERRAIN_LAVA, angle=ang)
        poly.terrain_type = 'lava'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'spike':
        # Spike is NOT a sensor - it has physical collision
        poly = add_poly(space, verts, (cx, cy), friction=0.8, elasticity=0.0,
                        color=COLOR_SPIKE, sensor=False, collision_type=COLLTYPE_TERRAIN_SPIKE, angle=ang)
        poly.terrain_type = 'spike'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    elif terrain_type == 'storm':
        poly = add_poly(space, verts, (cx, cy), friction=0.6, elasticity=0.0,
                        color=COLOR_STORM, sensor=True, collision_type=COLLTYPE_TERRAIN_STORM, angle=ang)
        poly.terrain_type = 'storm'
        poly.resource_value = resource_value
        poly.hazard_level = hazard_level
        return poly
    
    return None


def _create_natural_terrain_shape(space: pymunk.Space, cfg: Config, terrain_type: str,
                                  pos: Tuple[float, float], base_w: float, base_h: float,
                                  rng: random.Random) -> pymunk.Shape | None:
    """Create a natural-looking irregular terrain shape.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
        terrain_type: Type of terrain
        pos: Center position
        base_w: Base width
        base_h: Base height
        rng: Random number generator

    Returns:
        Created terrain shape or None
    """
    cx, cy = pos
    
    # Create irregular blob using multiple vertices with noise
    num_points = rng.randint(10, 20)
    base_radius_w = base_w * 0.5
    base_radius_h = base_h * 0.5
    
    verts = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        # More variation for natural look
        noise_r = rng.uniform(0.65, 1.0)
        # Make it more irregular with wider variation
        r_w = base_radius_w * noise_r * rng.uniform(0.75, 1.25)
        r_h = base_radius_h * noise_r * rng.uniform(0.75, 1.25)
        
        # More angle variation
        angle_offset = rng.uniform(-0.15, 0.15)
        final_angle = angle + angle_offset
        
        x = r_w * math.cos(final_angle)
        y = r_h * math.sin(final_angle)
        verts.append((x, y))
    
    # Add small rotation
    ang = rng.uniform(-math.pi / 6, math.pi / 6)
    
    if terrain_type == 'ice':
        poly = add_poly(space, verts, (cx, cy), friction=cfg.ice_friction,
                        elasticity=cfg.ice_elasticity, color=COLOR_ICE, angle=ang)
        poly.terrain_type = 'ice'
        return poly
    
    elif terrain_type == 'sand':
        # Use polygon for sand too (irregular)
        poly = add_poly(space, verts, (cx, cy), friction=cfg.sand_friction,
                        elasticity=cfg.sand_elasticity, color=COLOR_SAND, angle=ang)
        poly.terrain_type = 'sand'
        return poly
    
    elif terrain_type == 'bounce':
        poly = add_poly(space, verts, (cx, cy), friction=cfg.bounce_friction,
                        elasticity=cfg.bounce_elasticity, color=COLOR_BOUNCE, angle=ang)
        poly.terrain_type = 'bounce'
        return poly
    
    elif terrain_type == 'water':
        poly = add_poly(space, verts, (cx, cy), friction=0.2, elasticity=0.0,
                        color=COLOR_WATER, sensor=True, collision_type=COLLTYPE_WATER_SENSOR, angle=ang)
        poly.terrain_type = 'water'
        return poly
    
    elif terrain_type == 'mud':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MUD, sensor=True, collision_type=COLLTYPE_TERRAIN_MUD, angle=ang)
        poly.terrain_type = 'mud'
        return poly
    
    elif terrain_type == 'boost':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_BOOST, sensor=True, collision_type=COLLTYPE_TERRAIN_BOOST, angle=ang)
        poly.terrain_type = 'boost'
        return poly
    
    elif terrain_type == 'toxic':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_TOXIC, sensor=True, collision_type=COLLTYPE_TERRAIN_TOXIC, angle=ang)
        poly.terrain_type = 'toxic'
        return poly
    
    elif terrain_type == 'meadow':
        poly = add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MEADOW, sensor=True, collision_type=COLLTYPE_TERRAIN_MEADOW, angle=ang)
        poly.terrain_type = 'meadow'
        return poly
    
    return None


def _create_terrain_shape_for_type(space: pymunk.Space, cfg: Config, terrain_type: str,
                                   pos: Tuple[float, float], width: float, height: float,
                                   rng: random.Random) -> pymunk.Shape | None:
    """Create a terrain shape of specific type at position with given dimensions.

    Args:
        space: Pymunk physics space
        cfg: Configuration object
        terrain_type: Type of terrain
        pos: Center position
        width: Approximate width
        height: Approximate height
        rng: Random number generator

    Returns:
        Created terrain shape or None
    """
    cx, cy = pos
    ang = rng.uniform(-math.pi/12, math.pi/12)
    
    if terrain_type == 'ice':
        # Ice: Irregular polygon that covers the cell
        verts = [
            (-width*0.5, -height*0.5),
            (width*0.5, -height*0.5),
            (width*0.5, height*0.5),
            (-width*0.5, height*0.5),
        ]
        return add_poly(space, verts, (cx, cy), friction=cfg.ice_friction,
                        elasticity=cfg.ice_elasticity, color=COLOR_ICE, angle=ang)
    
    elif terrain_type == 'sand':
        # Sand: Circle (radius ensures coverage of diagonal)
        radius = math.sqrt(width*width + height*height) * 0.36  # Ensures cell coverage
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (cx, cy)
        circle = pymunk.Circle(body, radius)
        circle.friction = cfg.sand_friction
        circle.elasticity = cfg.sand_elasticity
        circle.color = COLOR_SAND
        space.add(body, circle)
        return circle
    
    elif terrain_type == 'bounce':
        # Bounce: Rectangle
        w, h = width * 0.95, height * 0.95
        verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        return add_poly(space, verts, (cx, cy), friction=cfg.bounce_friction,
                        elasticity=cfg.bounce_elasticity, color=COLOR_BOUNCE, angle=ang)
    
    elif terrain_type == 'water':
        # Water: Ellipse (covers cell)
        rx, ry = width * 0.5, height * 0.5
        num_points = 16
        verts = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            verts.append((rx * math.cos(angle), ry * math.sin(angle)))
        return add_poly(space, verts, (cx, cy), friction=0.2, elasticity=0.0,
                        color=COLOR_WATER, sensor=True, collision_type=COLLTYPE_WATER_SENSOR)
    
    elif terrain_type == 'mud':
        # Mud: Rectangle with slightly irregular edges
        w, h = width * 0.95, height * 0.95
        verts = [
            (-w/2, -h/2),
            (w/2, -h/2),
            (w/2, h/2),
            (-w/2, h/2),
        ]
        return add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MUD, sensor=True, collision_type=COLLTYPE_TERRAIN_MUD, angle=ang)
    
    elif terrain_type == 'boost':
        # Boost: Octagon (more coverage than star)
        radius = min(width, height) * 0.5
        num_points = 8
        verts = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi - math.pi / 8
            verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        return add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_BOOST, sensor=True, collision_type=COLLTYPE_TERRAIN_BOOST, angle=ang)
    
    elif terrain_type == 'toxic':
        # Toxic: Triangle (large enough to cover cell)
        size = math.sqrt(width*width + height*height) * 0.5
        verts = [
            (0, -size * 0.866),
            (-size * 0.5, size * 0.433),
            (size * 0.5, size * 0.433),
        ]
        return add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_TOXIC, sensor=True, collision_type=COLLTYPE_TERRAIN_TOXIC, angle=ang)
    
    elif terrain_type == 'meadow':
        # Meadow: Hexagon (large enough to cover cell)
        radius = min(width, height) * 0.52
        verts = []
        for i in range(6):
            angle = (i / 6) * 2 * math.pi - math.pi / 6
            verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        return add_poly(space, verts, (cx, cy), friction=0.9, elasticity=0.0,
                        color=COLOR_MEADOW, sensor=True, collision_type=COLLTYPE_TERRAIN_MEADOW, angle=ang)
    
    return None


def build_random_ice_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build ice terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'ice']


def build_random_sand_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build sand terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'sand']


def build_random_bounce_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build bounce terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'bounce']


def build_random_water_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build water terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'water']


def build_random_mud_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build mud terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'mud']


def build_random_boost_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build boost terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'boost']


def build_random_toxic_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build toxic terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'toxic']


def build_random_meadow_grid(space: pymunk.Space, cfg: Config, seed: int | None = None) -> list:
    """Build meadow terrain using grid system - returns shapes for this type only."""
    all_shapes = build_tiled_terrain(space, cfg, 'all', 6, 4, seed)
    return [s for s in all_shapes if _get_terrain_type_from_shape(s) == 'meadow']


def _get_terrain_type_from_shape(shape: pymunk.Shape) -> str:
    """Get terrain type from shape.
    
    First checks terrain_type attribute, then falls back to color matching.
    """
    # Check if terrain_type is stored on shape
    if hasattr(shape, 'terrain_type'):
        return shape.terrain_type
    
    # Fall back to color matching
    color = getattr(shape, "color", None)
    if not color:
        return 'ice'
    
    # Compare RGB (ignore alpha)
    r, g, b = color[0], color[1], color[2] if len(color) > 2 else 0
    
    if abs(r - COLOR_ICE[0]) < 10 and abs(g - COLOR_ICE[1]) < 10 and abs(b - COLOR_ICE[2]) < 10:
        return 'ice'
    elif abs(r - COLOR_SAND[0]) < 10 and abs(g - COLOR_SAND[1]) < 10 and abs(b - COLOR_SAND[2]) < 10:
        return 'sand'
    elif abs(r - COLOR_WATER[0]) < 10 and abs(g - COLOR_WATER[1]) < 10 and abs(b - COLOR_WATER[2]) < 10:
        return 'water'
    elif abs(r - COLOR_BOUNCE[0]) < 10 and abs(g - COLOR_BOUNCE[1]) < 10 and abs(b - COLOR_BOUNCE[2]) < 10:
        return 'bounce'
    elif abs(r - COLOR_MUD[0]) < 10 and abs(g - COLOR_MUD[1]) < 10 and abs(b - COLOR_MUD[2]) < 10:
        return 'mud'
    elif abs(r - COLOR_BOOST[0]) < 10 and abs(g - COLOR_BOOST[1]) < 10 and abs(b - COLOR_BOOST[2]) < 10:
        return 'boost'
    elif abs(r - COLOR_TOXIC[0]) < 10 and abs(g - COLOR_TOXIC[1]) < 10 and abs(b - COLOR_TOXIC[2]) < 10:
        return 'toxic'
    elif abs(r - COLOR_MEADOW[0]) < 10 and abs(g - COLOR_MEADOW[1]) < 10 and abs(b - COLOR_MEADOW[2]) < 10:
        return 'meadow'
    
    return 'ice'  # Default

