#!/usr/bin/env python3
"""Main entry point for the Engulf simulation."""

import sys

import pygame

from engulf.settings import WINDOW_WIDTH, WINDOW_HEIGHT
from engulf.config import Config
from engulf.scene_topdown import TopDownScene
from engulf.scene_settings import SettingsScene


def main() -> int:
    """Run the main game loop.

    Returns:
        Exit code (0 for success)
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    cfg = Config()

    while True:
        top = TopDownScene(screen, cfg)
        route = top.run()
        if route == "quit":
            break
        if route == "settings":
            while True:
                settings = SettingsScene(screen, cfg)
                r2 = settings.run()
                if r2 == "quit":
                    pygame.display.set_caption("")
                    pygame.quit()
                    return 0
                if r2 == "play":
                    break
                if r2 == "rebuild":
                    break
            continue
        if route == "rebuild":
            continue

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
