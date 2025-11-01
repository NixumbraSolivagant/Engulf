"""Settings UI scene for adjusting simulation parameters."""

from __future__ import annotations

from typing import List, Tuple

import pygame

from .config import Config
from .settings import WINDOW_WIDTH, WINDOW_HEIGHT, BACKGROUND_COLOR, FPS
from .i18n import get_strings
from .fonts import load_font


class SettingsScene:
    """Scene for adjusting simulation parameters interactively."""
    def __init__(self, screen: pygame.Surface, cfg: Config):
        """Initialize the settings scene.

        Args:
            screen: Pygame surface for rendering
            cfg: Configuration object to modify
        """
        self.screen = screen
        self.cfg = cfg
        self.clock = pygame.time.Clock()
        self.font = load_font(18)
        self.strings = get_strings(self.cfg.language)
        self.index = 0  # selected item
        self.items: List[Tuple[str, str]] = [
            ("wall_radius", "float"),
            ("ice_friction", "float"),
            ("ice_elasticity", "float"),
            ("sand_friction", "float"),
            ("sand_elasticity", "float"),
            ("bounce_friction", "float"),
            ("bounce_elasticity", "float"),
            ("water_damping_linear", "float"),
            ("water_damping_angular", "float"),
        ]

    def _clamp(self, name: str, value: float) -> float:
        """Clamp value to valid range based on parameter name.

        Args:
            name: Parameter name
            value: Value to clamp

        Returns:
            Clamped value within valid range
        """
        if "friction" in name:
            return max(0.0, min(3.0, value))
        if "elasticity" in name:
            return max(0.0, min(2.0, value))
        if name == "wall_radius":
            return max(1.0, min(20.0, value))
        if "water_damping" in name:
            return max(0.5, min(0.99, value))
        return value

    def _adjust(self, delta: float) -> None:
        """Adjust the currently selected parameter.

        Args:
            delta: Adjustment amount (positive or negative)
        """
        name, typ = self.items[self.index]
        cur = getattr(self.cfg, name)
        step = 0.01 if typ == "float" else 1
        new_val = cur + delta * step
        setattr(self.cfg, name, self._clamp(name, new_val))

    def _handle_events(self) -> str | None:
        """Handle keyboard and window events.

        Returns:
            Route string ("quit", "play", "rebuild") or None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_TAB, pygame.K_F1):
                    return "play"
                if event.key in (pygame.K_q,):
                    return "quit"
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    self.index = (self.index + 1) % len(self.items)
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.index = (self.index - 1) % len(self.items)
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    self._adjust(+1)
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self._adjust(-1)
                if event.key == pygame.K_PAGEUP:
                    self._adjust(+10)
                if event.key == pygame.K_PAGEDOWN:
                    self._adjust(-10)
                if event.key == pygame.K_l:
                    # Toggle language in-place and refresh strings/font
                    self.cfg.language = "en" if self.cfg.language == "zh" else "zh"
                    self.strings = get_strings(self.cfg.language)
                    self.font = load_font(18)
        return None

    def _draw(self) -> None:
        """Draw the settings UI."""
        s = self.strings
        self.screen.fill(BACKGROUND_COLOR)
        title = self.font.render(s.settings_title, True, (235, 235, 235))
        self.screen.blit(title, (10, 10))
        y = 40
        for i, (name, _) in enumerate(self.items):
            value = getattr(self.cfg, name)
            label = s.param_labels.get(name, name)
            text = f"{'> ' if i == self.index else '  '}" f"{label}: {value:.3f}"
            color = (255, 235, 120) if i == self.index else (220, 220, 220)
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (20, y))
            y += surf.get_height() + 6

        hint_lines = [
            s.settings_adjust_hint,
            s.settings_back_hint,
        ]
        y += 10
        for line in hint_lines:
            surf = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (20, y))
            y += surf.get_height() + 2

    def run(self) -> str:
        """Run the settings scene main loop.

        Returns:
            Route string ("quit", "play", "rebuild")
        """
        while True:
            route = self._handle_events()
            if route:
                return route
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)
