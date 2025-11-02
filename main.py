#!/usr/bin/env python3
"""Main entry point for the Engulf simulation."""

import sys
import os

import pygame

from engulf.settings import WINDOW_WIDTH, WINDOW_HEIGHT
from engulf.config import Config
from engulf.scene_topdown import TopDownScene
from engulf.scene_settings import SettingsScene

try:
    from engulf.stats_reporter import generate_statistics_report
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("警告: matplotlib 未安装，无法生成统计图表")


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
        
        # Generate statistics report when scene ends
        if STATS_AVAILABLE:
            stats = top.get_statistics()
            if stats['time_steps']:
                output_file = "simulation_stats.png"
                try:
                    generate_statistics_report(stats, output_file)
                    print(f"\n✅ 统计报告已生成: {os.path.abspath(output_file)}")
                except Exception as e:
                    print(f"\n⚠️  生成统计报告时出错: {e}")
        
        # Generate RL training report if RL mode was enabled
        if hasattr(top, 'use_rl') and top.use_rl and hasattr(top, 'rl_visualizer') and top.rl_visualizer:
            try:
                rl_output_file = "rl_training_report.png"
                success = top.rl_visualizer.generate_report(rl_output_file)
                if success:
                    print(f"\n✅ RL训练报告已生成: {os.path.abspath(rl_output_file)}")
                else:
                    print(f"\n⚠️  RL训练数据不足，未生成报告")
            except Exception as e:
                print(f"\n⚠️  生成RL训练报告时出错: {e}")
        
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
