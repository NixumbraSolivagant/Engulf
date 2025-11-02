# 强化学习系统 (Reinforcement Learning System)

## 概述

这是一个完整的强化学习系统，实现**全RL模式**，所有生物都使用神经网络进行决策。系统针对CPU优化，不需要GPU。

## 系统架构

```
engulf/rl/
├── __init__.py              # 模块导出
├── network.py               # 神经网络实现（NumPy，CPU优化）
├── ppo.py                   # PPO算法实现
├── state_encoder.py         # 状态编码器（环境→状态向量）
├── action_executor.py       # 动作执行器（动作→行为）
├── reward_shaping.py        # 奖励塑形和个性化奖励
├── experience_buffer.py     # 经验缓冲区
├── global_manager.py        # 全局训练管理器
├── rl_creature.py           # RL生物类
├── utils.py                 # 工具函数
└── integration_example.py   # 集成示例
```

## 核心特性

### 1. CPU优化的神经网络
- 使用NumPy实现，无GPU依赖
- 轻量级，适合实时训练
- 支持参数保存/加载

### 2. 完整的PPO实现
- 支持GAE（Generalized Advantage Estimation）
- 裁剪目标函数防止策略更新过大
- 熵正则化鼓励探索

### 3. 详细的状态编码（210维）
- 自身状态（25维）
- 基因特征（30维）
- 局部地形（60维）
- 附近生物（40维）
- 全局环境（15维）
- 历史信息（25维）
- 计算特征（15维）

### 4. 连续动作空间（12维）
- 基础移动：方向、速度、角速度
- 行为策略：探索/利用、资源/安全/社交权重
- 高级决策：风险承受、竞争性、适应速度

### 5. 个性化奖励系统
- 基于基因的奖励权重调整
- 多样化的奖励类型（生存、资源、探索、繁殖等）
- Episode级别的奖励计算

### 6. 全局训练管理
- 所有生物共享经验
- 集中训练，参数同步
- 训练统计跟踪

## 使用方法

### 基本集成

在 `scene_topdown.py` 中：

```python
from .rl import GlobalRLManager, RLCreature

class TopDownScene:
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 初始化RL系统
        self.use_rl = True  # 启用/禁用RL的标志
        if self.use_rl:
            self.rl_manager = GlobalRLManager(state_dim=210, action_dim=12)
        else:
            self.rl_manager = None
        
        # 生成RL生物
        if self.use_rl:
            self.creatures = self._spawn_rl_creatures(INITIAL_CREATURE_COUNT)
        else:
            from .creatures import spawn_creatures
            self.creatures = spawn_creatures(...)
    
    def _spawn_rl_creatures(self, count):
        creatures = []
        for _ in range(count):
            x = random.uniform(50, WINDOW_WIDTH - 50)
            y = random.uniform(50, WINDOW_HEIGHT - 50)
            creature = RLCreature(
                self.space,
                (x, y),
                genome=None,
                global_manager=self.rl_manager,
                use_rl=True
            )
            creatures.append(creature)
            self.body_to_creature[creature.body] = creature
        return creatures
    
    def _update_creatures(self, dt):
        for c in self.creatures:
            if c.dead:
                continue
            
            if isinstance(c, RLCreature) and c.use_rl:
                nearby = self._get_nearby_creatures(c)
                c.update(dt, scene=self, nearby_creatures=nearby)
            else:
                c.update(dt)
    
    def _get_nearby_creatures(self, creature, radius=300):
        nearby = []
        px, py = creature.body.position.x, creature.body.position.y
        for other in self.creatures:
            if other.dead or other is creature:
                continue
            dist = math.sqrt((px - other.body.position.x)**2 + 
                           (py - other.body.position.y)**2)
            if dist < radius:
                nearby.append(other)
        return nearby
```

### 训练和同步

在 `update` 方法中定期同步：

```python
def update(self, dt):
    # ... 现有更新逻辑 ...
    
    # RL训练和同步
    if self.use_rl and self.rl_manager:
        if not hasattr(self, '_rl_train_counter'):
            self._rl_train_counter = 0
        self._rl_train_counter += 1
        
        # 每100步同步一次
        if self._rl_train_counter % 100 == 0:
            rl_creatures = [c for c in self.creatures 
                          if isinstance(c, RLCreature) and c.use_rl]
            if rl_creatures:
                self.rl_manager.sync_to_agents(rl_creatures)
        
        # 打印统计信息
        if self._rl_train_counter % 500 == 0:
            stats = self.rl_manager.get_statistics()
            print(f"RL: Updates={stats['total_updates']}, "
                  f"Buffer={stats['buffer_size']}")
```

## 配置参数

### PPO配置
```python
from .rl import PPOConfig

config = PPOConfig(
    gamma=0.99,           # 折扣因子
    lambda_=0.95,         # GAE lambda
    clip_epsilon=0.2,     # 裁剪范围
    learning_rate=3e-4,    # 学习率
    batch_size=64,        # 批次大小
    n_epochs=10,          # 每次更新的迭代次数
)
```

### 网络结构
```python
policy = PolicyNetwork(
    state_dim=210,
    action_dim=12,
    hidden_sizes=(128, 64, 32),  # 隐藏层大小
    activation='tanh'
)
```

## 基因与RL的融合

1. **初始偏置**：基因影响网络的初始参数
2. **物理约束**：基因控制速度、体型等物理属性上限
3. **奖励权重**：不同基因有不同的奖励偏好
4. **网络参数**：训练后可通过基因偏置进行个性化调整

## 性能优化（CPU）

1. **轻量级网络**：使用NumPy而非PyTorch/TensorFlow
2. **经验缓冲限制**：限制为50,000条经验（而非100万+）
3. **批量训练**：积累足够经验后才训练
4. **异步更新**：定期同步而非每步同步

## 训练监控

```python
stats = rl_manager.get_statistics()
print(f"Total Updates: {stats['total_updates']}")
print(f"Buffer Size: {stats['buffer_size']}")
print(f"Average Reward: {stats.get('recent_avg_reward', 0.0):.2f}")
```

## 保存和加载

```python
# 保存
rl_manager.save('rl_model.json')

# 加载
rl_manager.load('rl_model.json')
```

## 注意事项

1. **状态维度**：当前固定为210维，根据实际编码可能略有变化
2. **训练频率**：建议每100-200步训练一次，避免过于频繁
3. **同步频率**：建议每50-100步同步一次参数
4. **CPU性能**：网络较小（~10K参数），CPU可流畅运行

## 下一步优化

- [ ] 实现完整的反向传播（当前为简化版本）
- [ ] 添加自适应学习率
- [ ] 实现课程学习
- [ ] 添加可视化工具
- [ ] 性能profiling和优化

