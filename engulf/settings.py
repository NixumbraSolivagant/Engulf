WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS = 120
BACKGROUND_COLOR = (16, 18, 22)

# Physics
GRAVITY = (0, 0)  # Top-down
SPACE_ITERATIONS = 20

# Player
PLAYER_RADIUS = 18
PLAYER_MASS = 2.0
PLAYER_FORCE = 2200.0
PLAYER_LINEAR_DAMPING = 0.9

# Terrain colors (RGBA)
COLOR_WALL = (230, 230, 230, 255)
COLOR_ICE = (120, 200, 255, 90)
COLOR_SAND = (220, 200, 120, 90)
COLOR_WATER = (120, 180, 220, 70)
COLOR_BOUNCE = (180, 120, 220, 90)
COLOR_RESOURCE = (120, 220, 150, 110)
COLOR_HAZARD = (230, 80, 80, 110)
# New effect terrains (lower alpha so debug draw remains visible)
COLOR_MUD = (140, 100, 70, 85)       # slow
COLOR_BOOST = (120, 240, 240, 85)    # speed up
COLOR_TOXIC = (180, 255, 120, 80)    # ambient hazard
COLOR_MEADOW = (160, 255, 180, 85)   # faster breeding

# New resource terrains
COLOR_CRYSTAL = (255, 150, 255, 90)  # high resource, variable
COLOR_FOREST = (50, 150, 50, 90)     # medium resource, safe
COLOR_GEYSER = (255, 180, 80, 85)    # high resource, periodic burst

# New hazard terrains
COLOR_LAVA = (255, 80, 40, 90)       # extreme hazard, accumulating
COLOR_SPIKE = (100, 20, 20, 90)      # high hazard, speed penalty
COLOR_STORM = (80, 60, 140, 85)      # medium-high hazard, random

# Collision types
COLLTYPE_DEFAULT = 0
COLLTYPE_WATER_SENSOR = 1
COLLTYPE_RESOURCE = 2
COLLTYPE_HAZARD = 3
COLLTYPE_TERRAIN_MUD = 10
COLLTYPE_TERRAIN_BOOST = 11
COLLTYPE_TERRAIN_TOXIC = 12
COLLTYPE_TERRAIN_MEADOW = 13
COLLTYPE_TERRAIN_CRYSTAL = 14
COLLTYPE_TERRAIN_FOREST = 15
COLLTYPE_TERRAIN_GEYSER = 16
COLLTYPE_TERRAIN_LAVA = 17
COLLTYPE_TERRAIN_SPIKE = 18
COLLTYPE_TERRAIN_STORM = 19

