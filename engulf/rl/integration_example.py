"""Example of how to integrate RL system into TopDownScene.

This file shows how to modify scene_topdown.py to use RL creatures.
"""

# In scene_topdown.py __init__:
"""
from .rl import GlobalRLManager, RLCreature

class TopDownScene:
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize RL system
        self.use_rl = True  # Flag to enable/disable RL
        if self.use_rl:
            self.rl_manager = GlobalRLManager(
                state_dim=210,  # Will be auto-adjusted
                action_dim=12,
            )
        else:
            self.rl_manager = None
        
        # Spawn creatures (use RLCreature instead of Creature)
        if self.use_rl:
            from .rl import RLCreature
            self.creatures = self._spawn_rl_creatures(...)
        else:
            from .creatures import spawn_creatures
            self.creatures = spawn_creatures(...)
"""

# In scene_topdown.py _update_creatures:
"""
def _update_creatures(self, dt: float):
    # ... existing terrain/affinity code ...
    
    # Update creatures
    for c in self.creatures:
        if c.dead:
            continue
        
        if isinstance(c, RLCreature) and c.use_rl:
            # RL update
            nearby = self._get_nearby_creatures(c, radius=300)
            c.update(dt, scene=self, nearby_creatures=nearby)
        else:
            # Original gene-driven update
            c.update(dt)
        
        # ... rest of update logic ...
"""

# Helper method for getting nearby creatures:
"""
def _get_nearby_creatures(self, creature, radius: float = 300.0):
    nearby = []
    px, py = creature.body.position.x, creature.body.position.y
    
    for other in self.creatures:
        if other.dead or other is creature:
            continue
        ox, oy = other.body.position.x, other.body.position.y
        dist = math.sqrt((px - ox)**2 + (py - oy)**2)
        if dist < radius:
            nearby.append(other)
    
    return nearby
"""

# In scene_topdown.py update method:
"""
def update(self, dt: float):
    # ... existing code ...
    
    # Periodic RL training and syncing
    if self.use_rl and self.rl_manager:
        # Train every N steps
        if hasattr(self, '_rl_train_counter'):
            self._rl_train_counter += 1
        else:
            self._rl_train_counter = 0
        
        if self._rl_train_counter % 100 == 0:
            # Sync parameters to all RL creatures
            rl_creatures = [c for c in self.creatures 
                          if isinstance(c, RLCreature) and c.use_rl]
            if rl_creatures:
                self.rl_manager.sync_to_agents(rl_creatures)
        
        # Print training stats occasionally
        if self._rl_train_counter % 500 == 0:
            stats = self.rl_manager.get_statistics()
            print(f"RL Stats: Updates={stats['total_updates']}, "
                  f"Buffer={stats['buffer_size']}, "
                  f"AvgReward={stats.get('recent_avg_reward', 0.0):.2f}")
"""

# Spawn RL creatures:
"""
def _spawn_rl_creatures(self, count: int):
    from .rl import RLCreature
    creatures = []
    for _ in range(count):
        x = random.uniform(50, self.cfg.window_width - 50)
        y = random.uniform(50, self.cfg.window_height - 50)
        creature = RLCreature(
            self.space,
            (x, y),
            genome=None,  # Random genome
            global_manager=self.rl_manager,
            use_rl=True
        )
        creatures.append(creature)
        self.body_to_creature[creature.body] = creature
    return creatures
"""


