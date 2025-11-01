"""Configuration settings for the simulation."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for terrain and UI settings.

    Attributes:
        language: Display language ("en" or "zh")
        wall_radius: Wall segment radius
        ice_friction: Ice terrain friction coefficient
        ice_elasticity: Ice terrain elasticity coefficient
        sand_friction: Sand terrain friction coefficient
        sand_elasticity: Sand terrain elasticity coefficient
        bounce_friction: Bounce pad friction coefficient
        bounce_elasticity: Bounce pad elasticity coefficient
        water_damping_linear: Water linear velocity damping
        water_damping_angular: Water angular velocity damping
    """
    # UI
    language: str = "zh"  # "en" or "zh"

    # Terrain parameters
    wall_radius: float = 6.0

    ice_friction: float = 0.02
    ice_elasticity: float = 0.05

    sand_friction: float = 1.5
    sand_elasticity: float = 0.05

    bounce_friction: float = 0.4
    bounce_elasticity: float = 1.2

    water_damping_linear: float = 0.90
    water_damping_angular: float = 0.90
