"""Resource spawning and management."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import pymunk

from .settings import COLOR_RESOURCE, COLLTYPE_RESOURCE


def spawn_resource_polygon(space: pymunk.Space, verts: List[Tuple[float, float]], 
                           pos: Tuple[float, float], angle: float = 0.0) -> pymunk.Poly:
    """Spawn a polygon-shaped resource.

    Args:
        space: Pymunk physics space
        verts: Polygon vertices (relative to center)
        pos: Center position
        angle: Rotation angle

    Returns:
        Created resource polygon shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    body.angle = angle
    poly = pymunk.Poly(body, verts)
    poly.sensor = True
    poly.color = COLOR_RESOURCE
    poly.collision_type = COLLTYPE_RESOURCE
    space.add(body, poly)
    return poly


def spawn_resource_circle(space: pymunk.Space, pos: Tuple[float, float], radius: float = 8.0) -> pymunk.Circle:
    """Spawn a circle-shaped resource.

    Args:
        space: Pymunk physics space
        pos: Position tuple (x, y)
        radius: Resource radius

    Returns:
        Created resource circle shape
    """
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Circle(body, radius)
    body.position = pos
    shape.sensor = True
    shape.color = COLOR_RESOURCE
    shape.collision_type = COLLTYPE_RESOURCE
    space.add(body, shape)
    return shape


def spawn_resource_for_terrain(space: pymunk.Space, terrain_type: str, 
                               pos: Tuple[float, float]) -> pymunk.Shape:
    """Spawn a resource matching the terrain type.

    Args:
        space: Pymunk physics space
        terrain_type: Type of terrain ('ice', 'sand', 'water', etc.)
        pos: Position tuple (x, y)

    Returns:
        Created resource shape
    """
    rng = random.Random()
    
    if terrain_type == 'ice':
        # Ice: Diamond shape (square rotated 45 degrees)
        size = random.uniform(6, 12)
        verts = [(-size, 0), (0, -size), (size, 0), (0, size)]
        return spawn_resource_polygon(space, verts, pos, math.pi / 4)
    
    elif terrain_type == 'sand':
        # Sand: Circle
        radius = random.uniform(6, 10)
        return spawn_resource_circle(space, pos, radius)
    
    elif terrain_type == 'water':
        # Water: Ellipse (polygon approximation)
        rx, ry = random.uniform(8, 14), random.uniform(5, 9)
        num_points = 12
        verts = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            verts.append((rx * math.cos(angle), ry * math.sin(angle)))
        return spawn_resource_polygon(space, verts, pos)
    
    elif terrain_type == 'bounce':
        # Bounce: Rectangle
        w, h = random.uniform(8, 14), random.uniform(5, 9)
        verts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        return spawn_resource_polygon(space, verts, pos, random.uniform(0, math.pi/4))
    
    elif terrain_type == 'mud':
        # Mud: Irregular blob (5-sided polygon)
        radius = random.uniform(6, 10)
        verts = []
        for i in range(5):
            angle = (i / 5) * 2 * math.pi
            r = radius * random.uniform(0.7, 1.0)
            verts.append((r * math.cos(angle), r * math.sin(angle)))
        return spawn_resource_polygon(space, verts, pos)
    
    elif terrain_type == 'boost':
        # Boost: Star shape
        outer_r = random.uniform(6, 10)
        inner_r = outer_r * 0.5
        num_points = 5
        verts = []
        for i in range(num_points * 2):
            angle = (i / (num_points * 2)) * 2 * math.pi - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            verts.append((r * math.cos(angle), r * math.sin(angle)))
        return spawn_resource_polygon(space, verts, pos)
    
    elif terrain_type == 'toxic':
        # Toxic: Triangle (warning shape)
        size = random.uniform(7, 12)
        verts = [
            (0, -size * 0.866),
            (-size * 0.5, size * 0.433),
            (size * 0.5, size * 0.433),
        ]
        return spawn_resource_polygon(space, verts, pos)
    
    elif terrain_type == 'meadow':
        # Meadow: Hexagon
        radius = random.uniform(6, 10)
        verts = []
        for i in range(6):
            angle = (i / 6) * 2 * math.pi - math.pi / 6
            verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        return spawn_resource_polygon(space, verts, pos)
    
    else:
        # Default: Circle
        return spawn_resource_circle(space, pos, random.uniform(6, 10))
