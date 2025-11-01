from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Strings:
    title: str
    scene_label: str
    hint_settings: str
    hint_toggle_draw: str
    hint_quit: str
    settings_title: str
    settings_back_hint: str
    settings_adjust_hint: str
    param_labels: Dict[str, str]


EN = Strings(
    title="Top-Down Physics Terrains (Pygame + Pymunk)",
    scene_label="Scene: Top-Down Terrains (static)",
    hint_settings="F1 / Tab: Settings  |  L: Language",
    hint_toggle_draw="G: Toggle debug draw",
    hint_quit="Q / Esc: Quit",
    settings_title="Settings (F1/Tab to return)  |  L: Language",
    settings_back_hint="Esc/Tab/F1: Back  Q: Quit",
    settings_adjust_hint="Up/Down: Select  Left/Right: Adjust  PgUp/PgDn: x10",
    param_labels={
        "wall_radius": "Wall Radius",
        "ice_friction": "Ice Friction",
        "ice_elasticity": "Ice Elasticity",
        "sand_friction": "Sand Friction",
        "sand_elasticity": "Sand Elasticity",
        "bounce_friction": "Bounce Friction",
        "bounce_elasticity": "Bounce Elasticity",
        "water_damping_linear": "Water Damping (Linear)",
        "water_damping_angular": "Water Damping (Angular)",
    },
)

ZH = Strings(
    title="俯视物理地形（Pygame + Pymunk）",
    scene_label="场景：俯视地形（带生物）",
    hint_settings="F1 / Tab：设置  |  L：语言",
    hint_toggle_draw="G：切换调试绘制",
    hint_quit="Q / Esc：退出",
    settings_title="设置（F1/Tab 返回）  |  L：语言",
    settings_back_hint="Esc/Tab/F1：返回  Q：退出",
    settings_adjust_hint="上/下：选择  左/右：调整  PgUp/PgDn：x10",
    param_labels={
        "wall_radius": "墙线半径",
        "ice_friction": "冰面摩擦",
        "ice_elasticity": "冰面弹性",
        "sand_friction": "沙地摩擦",
        "sand_elasticity": "沙地弹性",
        "bounce_friction": "弹板摩擦",
        "bounce_elasticity": "弹板弹性",
        "water_damping_linear": "水域阻尼（线性）",
        "water_damping_angular": "水域阻尼（角速度）",
    },
)


def get_strings(lang: str) -> Strings:
    return EN if lang == "en" else ZH
