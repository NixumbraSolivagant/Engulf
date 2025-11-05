"""Configuration settings for the simulation."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration parameters for the simulation.
    
    Attributes:
        terrain_friction_ice: Friction coefficient for ice terrain
        terrain_elasticity_ice: Elasticity coefficient for ice terrain
        terrain_friction_sand: Friction coefficient for sand terrain
        terrain_elasticity_sand: Elasticity coefficient for sand terrain
        terrain_friction_bounce: Friction coefficient for bounce terrain
        terrain_elasticity_bounce: Elasticity coefficient for bounce terrain
        language: Current language setting ('en' or 'zh')
        use_rl: Whether to use reinforcement learning for all creatures
    """
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
    
    # RL settings
    use_rl: bool = True  # Enable RL mode (requires RL_AVAILABLE)

    # RL reward tuning (optional knobs)
    rl_cell_size: float = 60.0                   # RL 内部网格大小（像素），用于新奇度/访问计数
    rl_novelty_coeff: float = 0.6                # 新奇度奖励系数
    rl_novelty_time_const: float = 500.0         # 新奇度时间常数（步），时间越长“再新”越慢
    rl_energy_cost_per_px: float = 0.005         # 能量代价/像素（下调，减轻负担）
    rl_energy_intake_coeff: float = 0.5          # 资源摄入的能量回补系数
    rl_hazard_penalty_scale: float = 0.4         # 危险连续惩罚强度系数（增强以避免灭绝）
    rl_hazard_event_penalty: float = 0.5         # 当帧受险事件的离散惩罚（增强）
    rl_hazard_penalty_cap: float = 5.0           # 危险连续惩罚的强度上限（提高上限以惩罚高危险）
    rl_hazard_streak_threshold: int = 3          # 连续受险步数阈值（达到后额外惩罚）
    rl_hazard_streak_penalty: float = 3.0        # 连续受险额外惩罚（增强）
    rl_leave_danger_bonus: float = 2.0          # 离开危险区的学习奖励（增强以鼓励离开）
    rl_safe_chain_steps: int = 200               # 安全长链判定步数阈值
    rl_safe_chain_bonus_scale: float = 3.0       # 安全长链奖励系数（乘以 caution 基因）
    rl_territory_radius: float = 120.0           # 领地半径（像素，以出生点为圆心）
    rl_territory_bonus: float = 1.5              # 自身领地奖励
    rl_intrude_penalty: float = 1.0              # 侵入他者领地惩罚
    rl_diversity_center: float = 0.6             # 群体多样性全局奖励中心点（高于此开始加分）
    rl_diversity_scale: float = 2.0              # 群体多样性全局奖励缩放
    rl_forward_unvisited_bonus: float = 1.0      # 面朝未访问格子的方向性奖励
    rl_repeat_visit_penalty: float = 0.05        # 重复访问网格的惩罚系数/次（下调）
    rl_resource_switch_bonus: float = 1.0        # 资源类型切换的多样性奖励（下调）
    rl_survival_decay_factor: float = 0.6        # 生存奖励的指数衰减因子（越小衰减越快）

    # 奖励总体缩放与裁剪（稳定训练）
    rl_reward_scale: float = 0.1                 # 所有单步奖励的整体缩放
    rl_reward_clip: float = 2.0                  # 单步奖励裁剪区间 [-clip, +clip]

    # 训练频率
    rl_train_frequency: int = 20                 # 个体每收集多少条经验更新一次

    # 预加载权重
    rl_load_weights_dir: str = ""               # 如设置为 outputs/<timestamp>/weights 则启动时加载分配
