# Engulf - 生态系统物理模拟

一个使用 Pygame（渲染）和 Pymunk（2D 物理引擎）构建的生态系统模拟器。包含具有基因系统的生物、繁衍机制、捕食关系、动态地形等特性。

An ecosystem simulation built with Pygame (rendering) and Pymunk (2D physics engine). Features creatures with genetics, reproduction, predation, and dynamic terrain.

## 功能特性 / Features

### 生物系统 / Creature System
- **基因系统** / **Genetics**: 每个生物都有包含14种特征的基因（外观、性格、能力、调节因子等）
- **繁衍机制** / **Reproduction**: 
  - 同种生物相遇可以繁衍，后代继承并变异父母基因
  - 考虑美丽度（beauty）和繁殖力（fertility）基因影响成功率
  - 生物具有繁殖本能，会主动寻找同类配偶
  - 初始生物数：40，最大种群数：64
- **生命周期** / **Lifespan**: 生物有独立的寿命（25-60秒），时间到了会自然死亡
- **捕食关系** / **Predation**: 不同物种之间可以互相捕食，较大的生物可以吞噬较小的（同种不可捕食）
- **趋利避害** / **Tropism**: 生物会主动寻找资源地形并回避危险地形，基于`curiosity`和`caution`基因，能够感知316像素范围内的地形并考虑多个候选目标（增强版）

### 基因特征 / Genetic Traits
- **外观** / **Appearance**: 身体半径、颜色（色相）
- **性格** / **Personality**: 好奇心、攻击性、谨慎度
- **能力** / **Abilities**: 最大速度、角速度
- **生命** / **Life**: 寿命（秒）
- **调节因子** / **Modifiers**: 
  - 新陈代谢（metabolism）：影响资源消耗时的成长幅度
  - 速度适应性（speed_adaptability）：影响资源带来的速度提升
  - 繁殖力（fertility）：影响资源带来的繁殖冷却减少
  - 抗性（resilience）：影响危险致死概率
  - 美丽度（beauty）：影响繁衍成功率

### 地形系统 / Terrain System

地形采用**六边形蜂巢网格**布局，完整覆盖整个屏幕，无缝衔接。

#### 基础地形 / Basic Terrain
- **冰面（Ice）**：低摩擦、低弹性，无资源，轻微危险（寒冷）
- **沙地（Sand）**：高摩擦、低弹性，少量资源，无危险
- **弹跳板（Bounce）**：中等摩擦、高弹性，极少资源，无危险
- **水域（Water）**：提供线性/角速度阻尼，较多资源，中等危险（溺水风险）

#### 效果地形 / Effect Terrain (传感器区域，生物进入时生效)
- **泥地（Mud）**：降低移动速度（×0.6），少量资源，中等危险（被困）
- **加速区（Boost）**：提升移动速度（×1.25），较多资源，无危险
- **毒区（Toxic）**：持续造成环境伤害，无资源，高危险
- **草地（Meadow）**：加快繁殖冷却，丰富资源，极低危险

#### 资源型地形 / Resource Terrain
- **水晶（Crystal）**：极高资源（0.35），中等危险，周期性能量爆发
- **森林（Forest）**：中高资源（0.22），低危险，安全稳定
- **地热（Geyser）**：高资源（0.28），中高危险，周期性喷发

#### 危险型地形 / Hazard Terrain
- **岩浆（Lava）**：低资源（0.08），极高危险（0.40），粘性减速，累积伤害
- **尖刺（Spike）**：无资源，高危险（0.30），速度越快伤害越高（物理碰撞）
- **风暴（Storm）**：无资源，中高危险（0.20），随机闪电打击

#### 混合动态地形系统 / Mixed Dynamic Terrain System

每种地形都具有动态变化机制，包含多个层次：

1. **基础属性** / **Base Properties**: 每种地形都有静态的资源值和危险级别
2. **周期波动** / **Periodic Waves**: 资源/危险值在±15-30%范围内波动，周期5-15秒
3. **特殊事件** / **Special Events**: 
   - 水晶：随机能量爆发（资源×2.0，持续2秒）
   - 地热：周期性喷发（资源×2.0，危险×2.0，持续1秒）
   - 岩浆：周期性激增（危险×1.5，持续2秒）
   - 风暴：随机闪电（危险×1.75，持续0.5秒）
   - 草地：昼夜循环（夜晚资源减少33%）
4. **季节变化** / **Seasonal Changes**: 全局季节循环（生长期/危险期/正常期），影响资源/危险水平
5. **生物反馈** / **Creature Feedback**: 
   - 资源消耗：生物聚集时消耗地形资源
   - 资源恢复：无生物时资源自动恢复
   - 污染积累：大量生物聚集时危险地形危险增加
6. **生态演替** / **Ecological Succession**: 低概率地形类型进化（如沙地→草地，毒区→森林）

- **独立随机化** / **Independent Randomization**: 每种地形类型都有独立的随机变化计时器（15-42秒），在不同时间重新生成布局

### 资源与危险 / Resources & Hazards

**资源和危险是地形的固有属性**，而非独立对象。生物站在地形上时会自动获得/承受效果：

- **资源效果** / **Resource Effects**: 
  - 持续成长（身体半径、质量增加）
  - 速度提升（基于`speed_adaptability`基因）
  - 繁殖冷却减少（基于`fertility`基因）
  
- **危险效果** / **Hazard Effects**: 
  - 概率性死亡（基于`resilience`基因）
  - 速度惩罚
  - 繁殖冷却增加

### 用户界面 / User Interface
- **双语支持** / **Bilingual Support**: 支持中文和英文，按 `L` 键切换
- **设置页面** / **Settings Page**: 可以实时调整地形参数（摩擦、弹性等）

## 安装 / Installation

```bash
# 创建虚拟环境 / Create virtual environment
python3 -m venv .venv

# 激活虚拟环境 / Activate virtual environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 升级 pip / Upgrade pip
pip install --upgrade pip

# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

## 运行 / Run

```bash
python main.py
```

## 操作控制 / Controls

### 主场景 / Main Scene
- **F1 / Tab**: 打开设置页面
- **G**: 切换调试绘制模式（显示物理形状）
- **L**: 切换语言（中文 ↔ 英文）
- **R**: 重建场景
- **Q / Esc**: 退出

### 设置页面 / Settings Page
- **↑/↓ 或 W/S**: 选择参数
- **←/→ 或 A/D**: 调整参数值
- **PageUp / PageDown**: 快速调整（×10）
- **Esc / Tab / F1**: 返回主场景
- **L**: 切换语言（中文 ↔ 英文）
- **Q**: 退出程序

## 技术架构 / Technical Architecture

```
engulf/
├── scene_topdown.py    # 主场景：物理模拟、生物更新、繁衍、地形动态系统
├── creatures.py         # 生物类：基因、行为、物理体、趋利避害
├── terrain.py          # 地形生成和管理（六边形网格、动态系统）
├── resources.py         # 资源生成（已弃用，资源现在是地形属性）
├── hazards.py           # 危险生成（已弃用，危险现在是地形属性）
├── scene_settings.py    # 设置页面UI
├── config.py           # 配置参数（地形属性等）
├── settings.py         # 全局常量（颜色、碰撞类型等）
├── i18n.py             # 国际化字符串
└── fonts.py            # 字体加载（支持CJK字符）
```

## 依赖 / Dependencies

- `pygame==2.6.1` - 渲染和事件处理
- `pymunk==6.6.0` - 2D 物理引擎

## 开发说明 / Development Notes

- **物理引擎** / **Physics**: 场景使用零重力物理，通过摩擦和弹性模拟地形效果
- **生物行为** / **Creature Behavior**: 生物使用力驱动移动，具有增强的趋向性和回避行为（可感知316像素范围）
- **地形系统** / **Terrain System**: 
  - 六边形网格布局，完整覆盖屏幕
  - 大部分地形使用传感器（sensor）实现，不产生物理碰撞响应
  - 尖刺地形（Spike）例外：具有物理碰撞，速度越快伤害越高
- **渲染系统** / **Rendering**: 自定义渲染确保生物始终可见（地形作为透明层绘制，生物绘制在上层）
- **性能优化** / **Performance**: 
  - 生物计数每0.5秒更新一次而非每帧
  - 地形距离检查优化（跳过>316像素的远距离地形）
  - 动态状态与静态属性同步更新

## 许可证 / License

本项目仅用于学习和演示目的。

This project is for educational and demonstration purposes only.