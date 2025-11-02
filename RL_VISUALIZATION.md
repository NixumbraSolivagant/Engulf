# RL训练可视化指南

## 概述

RL训练可视化系统提供了实时监控和离线分析RL学习过程的功能，包括训练曲线、损失函数、奖励分布等。

## 功能特性

### 1. 实时监控（游戏中显示）

在游戏运行时，RL训练统计信息会显示在屏幕左上角的HUD中：

```
RL Training:
  Updates: 150
  Avg Reward: 2.45
  Episode Reward: 15.30
```

**显示内容**：
- **Updates**: 总训练更新次数
- **Avg Reward**: 最近的平均奖励值
- **Episode Reward**: 最近完成的episode的平均奖励

### 2. 离线报告（训练结束后）

当场景结束时，会自动生成详细的训练报告 `rl_training_report.png`，包含：

#### 报告内容

1. **平均奖励曲线** (Average Reward Over Time)
   - 显示训练过程中的平均奖励变化
   - 包含移动平均线（MA）显示趋势
   
2. **策略损失** (Policy Loss)
   - PPO算法的策略网络损失
   - 反映策略优化的进展情况

3. **价值损失** (Value Loss)
   - 价值网络的损失
   - 反映价值函数估计的准确性

4. **策略熵** (Policy Entropy)
   - 策略的随机性/探索性
   - 高熵 = 更多探索，低熵 = 更多利用

5. **PPO裁剪比例** (Clip Fraction)
   - PPO算法中策略更新被裁剪的比例
   - 反映策略更新幅度

6. **经验缓冲区大小** (Buffer Size)
   - 当前存储的经验数量
   - 反映学习数据的积累情况

7. **Episode奖励分布** (Episode Reward Distribution)
   - 所有episode奖励的直方图
   - 显示奖励的分布特征

8. **Episode长度分布** (Episode Length Distribution)
   - 所有episode的长度分布
   - 反映生物的生存时长

9. **训练摘要** (Training Summary)
   - 关键统计信息汇总
   - 包括总更新次数、最佳/最差奖励等

## 使用方法

### 基本使用

1. **启用RL模式**（如果还未启用）：
   ```python
   # 在 config.py 中
   use_rl: bool = True
   ```

2. **运行游戏**：
   ```bash
   python main.py
   ```

3. **观察实时统计**：
   - 在游戏左上角查看RL训练统计
   - 控制台每500步会打印训练信息

4. **查看离线报告**：
   - 退出场景后，自动生成 `rl_training_report.png`
   - 在项目根目录查看

### 手动生成报告

如果需要手动生成报告：

```python
from engulf.rl import RLTrainingVisualizer

# 如果已有visualizer实例
if scene.rl_visualizer:
    scene.rl_visualizer.generate_report("my_rl_report.png")
```

## 数据收集

可视化系统会自动收集以下数据：

### 训练步骤数据
- 每次训练更新时记录：
  - 平均奖励
  - 策略损失
  - 价值损失
  - 熵值
  - 裁剪比例
  - 缓冲区大小

### Episode数据
- 每个episode结束时记录：
  - 总奖励
  - Episode长度（步数）

### 历史记录
- 默认保留最近10000个训练步骤的数据
- Episode奖励和长度各保留最近1000个

## 性能考虑

- **内存使用**：历史数据限制在10000步以内，防止内存过度使用
- **更新频率**：训练数据在每次训练更新时记录（约每100个经验）
- **报告生成**：只在场景结束时生成，不影响实时性能

## 解读报告

### 健康的学习信号

✅ **奖励曲线上升**：说明生物在学习和改进
✅ **损失下降**：说明网络正在收敛
✅ **适中的熵值**：平衡探索和利用
✅ **Episode奖励增长**：说明策略越来越好

### 需要关注的问题

⚠️ **奖励不增长或下降**：可能需要调整学习率或奖励设计
⚠️ **损失震荡严重**：学习率可能过高
⚠️ **熵值过低**：探索不足，可能陷入局部最优
⚠️ **裁剪比例过高**：策略更新幅度过大，可能需要更小的学习率

## 自定义可视化

### 修改历史记录长度

```python
# 创建visualizer时指定
from engulf.rl import RLTrainingVisualizer

visualizer = RLTrainingVisualizer(max_history=20000)  # 保留20000步
```

### 获取统计数据文本

```python
stats_text = visualizer.get_simple_stats_text()
print(stats_text)  # "Updates: 150 | Avg Reward: 2.45 | ..."
```

### 获取详细统计

```python
stats = visualizer.get_recent_statistics()
print(stats)
# {
#     'avg_reward': 2.45,
#     'avg_policy_loss': 0.123,
#     'total_updates': 150,
#     ...
# }
```

## 故障排除

**问题**：报告中没有数据
- **原因**：训练还没开始或数据不足
- **解决**：让游戏运行更长时间，等待训练更新

**问题**：HUD中看不到RL统计
- **检查**：确认 `use_rl = True` 且 RL模块可用
- **检查**：确认 `rl_visualizer` 已初始化

**问题**：报告生成失败
- **检查**：确认已安装 matplotlib: `pip install matplotlib`
- **检查**：查看控制台错误信息

**问题**：报告图片太大
- **调整**：修改 `visualizer.py` 中的 `figsize=(16, 12)` 参数
- **调整**：修改 `dpi=150` 参数降低分辨率

## 最佳实践

1. **定期查看报告**：每次长时间训练后查看报告，了解学习进展
2. **对比不同运行**：保存不同配置下的报告，对比学习效果
3. **关注趋势**：关注奖励曲线的整体趋势，而非短期波动
4. **调整超参数**：根据报告中的指标调整学习率、批次大小等超参数

## 技术细节

- **后端**：使用 matplotlib 的 'Agg' 后端（非交互式）
- **数据存储**：使用 Python 列表和 deque 存储历史数据
- **更新频率**：训练步骤数据在每次训练更新时记录
- **报告格式**：PNG格式，150 DPI分辨率

---

**提示**：首次运行RL模式时，训练需要一些时间才能积累足够的数据。建议至少运行几分钟后再查看报告。

