"""Hazard spawning and management."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import pymunk

from .settings import COLOR_HAZARD, COLLTYPE_HAZARD


def spawn_hazard_polygon(space: pymunk.Space, verts: List[Tuple[float, float]], 
                         pos: Tuple[float, float], angle: float = 0.0) -> pymunk.Poly:
    """Spawn a polygon-shaped hazard.

    Args:
        space: Pymunk physics space
        verts: Polygon vertices (relative to center)
        pos: Center position
        angle: Rotation angle

    Returns:
        Created hazard polygon shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    body.angle = angle
    poly = pymunk.Poly(body, verts)
    poly.sensor = True
    poly.color = COLOR_HAZARD
    poly.collision_type = COLLTYPE_HAZARD
    space.add(body, poly)
    return poly


def spawn_hazard_circle(space: pymunk.Space, pos: Tuple[float, float], radius: float = 14.0) -> pymunk.Circle:
    """Spawn a circle-shaped hazard.

    Args:
        space: Pymunk physics space
        pos: Position tuple (x, y)
        radius: Hazard radius

    Returns:
        Created hazard circle shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.sensor = True
    shape.color = COLOR_HAZARD
    shape.collision_type = COLLTYPE_HAZARD
    space.add(body, shape)
    return shape


def spawn_hazard_for_terrain(space: pymunk.Space, terrain_type: str, 
                             pos: Tuple[float, float]) -> pymunk.Shape:
    """Spawn a hazard matching the terrain type.

    Args:
        space: Pymunk physics space
        terrain_type: Type of terrain ('ice', 'sand', 'water', etc.)
        pos: Position tuple (x, y)

    Returns:
        Created hazard shape
    """
    rng = random.Random()
    
    if terrain_type == 'ice':
        # Ice: Sharp spikes (diamond)
        size = random.uniform(10, 16)
        verts = [(-size, 0), (0, -size), (size, 0), (0, size)]
        return spawn_hazard_polygon(space, verts, pos, math.pi / 4)
    
    elif terrain_type == 'sand':
        # Sand: Quicksand circles
        radius = random.uniform(10, 16)
        return spawn_hazard_circle(space, pos, radius)
    
    elif terrain_type == 'water':
        # Water: Whirlpool (spiral approximation with ellipse)
        rx, ry = random.uniform(12, 18), random.uniform(8, 12)
        num_points = 16
        verts = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Slight spiral effect
            r_factor = 0.9 + 0.2 * (i / num_points)
            verts.append((rx * r_factor * math.cos(angle), ry * r_factor * math.sin(angle)))
        return spawn_hazard_polygon(space, verts, pos)
    
    elif terrain_type == 'bounce':
        # Bounce: Large spikes
        w, h = random.uniform(12, 18), random.uniform(8, 14)
        verts = [(-w/2, -h/2), (0, h/2), (w/2, -h/2)]
        return spawn_hazard_polygon(space, verts, pos, random.uniform(0, math.pi/4))
    
    elif terrain_type == 'mud':
        # Mud: Sinking pits (irregular circles)
        radius = random.uniform(10, 16)
        verts = []
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            r = radius * random.uniform(0.8, 1.0)
            verts.append((r * math.cos(angle), r * math.sin(angle)))
        return spawn_hazard_polygon(space, verts, pos)
    
    elif terrain_type == 'boost':
        # Boost: Sharp stars
        outer_r = random.uniform(10, 16)
        inner_r = outer_r * 0.4
        num_points = 5
        verts = []
        for i in range(num_points * 2):
            angle = (i / (num_points * 2)) * 2 * math.pi - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            verts.append((r * math.cos(angle), r * math.sin(angle)))
        return spawn_hazard_polygon(space, verts, pos)
    
    elif terrain_type == 'toxic':
        # Toxic: Large warning triangles
        size = random.uniform(14, 20)
        verts = [
            (0, -size * 0.866),
            (-size * 0.5, size * 0.433),
            (size * 0.5, size * 0.433),
        ]
        return spawn_hazard_polygon(space, verts, pos)
    
    elif terrain_type == 'meadow':
        # Meadow: Thorns (spiked hexagon)
        radius = random.uniform(10, 16)
        verts = []
        for i in range(6):
            angle = (i / 6) * 2 * math.pi - math.pi / 6
            # Alternate outer and inner points for spikes
            r = radius if i % 2 == 0 else radius * 0.7
            verts.append((r * math.cos(angle), r * math.sin(angle)))
        return spawn_hazard_polygon(space, verts, pos)
    
    else:
        # Default: Circle
        return spawn_hazard_circle(space, pos, random.uniform(10, 16))
