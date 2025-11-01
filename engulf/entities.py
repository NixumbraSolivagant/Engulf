from typing import Tuple

import pymunk

from .settings import PLAYER_RADIUS, PLAYER_MASS


def create_player(space: pymunk.Space, pos: Tuple[float, float]) -> pymunk.Shape:
    radius = PLAYER_RADIUS
    mass = PLAYER_MASS
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    body.angular_velocity = 0
    body.angle = 0
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.2
    shape.friction = 0.8
    space.add(body, shape)
    return shape


def create_puck(space: pymunk.Space, pos: Tuple[float, float], radius: float = 14,
                 mass: float = 1.0, elasticity: float = 0.3) -> pymunk.Shape:
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.elasticity = elasticity
    shape.friction = 0.4
    space.add(body, shape)
    return shape


def create_box(space: pymunk.Space, pos: Tuple[float, float], size: Tuple[int, int] = (28, 28),
               mass: float = 1.2, elasticity: float = 0.2) -> pymunk.Shape:
    w, h = size
    points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    moment = pymunk.moment_for_poly(mass, points)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Poly(body, points)
    shape.elasticity = elasticity
    shape.friction = 0.7
    space.add(body, shape)
    return shape

