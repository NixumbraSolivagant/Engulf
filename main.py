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
        
        # Prepare output directory with timestamp
        ts = __import__('time').strftime('%Y%m%d_%H%M%S', __import__('time').localtime())
        base_dir = os.path.abspath(os.path.join(os.getcwd(), 'outputs'))
        out_dir = os.path.join(base_dir, ts)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            out_dir = os.getcwd()

        # Generate statistics report when scene ends
        if STATS_AVAILABLE:
            stats = top.get_statistics()
            if stats['time_steps']:
                output_file = os.path.join(out_dir, "simulation_stats.png")
                try:
                    generate_statistics_report(stats, output_file)
                    print(f"\n✅ 统计报告已保存到: {os.path.abspath(output_file)}")
                except Exception as e:
                    print(f"\n⚠️  生成统计报告时出错: {e}")
        
        # Generate RL training report if RL mode was enabled
        if hasattr(top, 'use_rl') and top.use_rl and hasattr(top, 'rl_visualizer') and top.rl_visualizer:
            try:
                rl_output_file = os.path.join(out_dir, "rl_training_report.png")
                success = top.rl_visualizer.generate_report(rl_output_file)
                if success:
                    print(f"\n✅ RL训练报告已保存到: {os.path.abspath(rl_output_file)}")
                else:
                    print(f"\n⚠️  RL训练数据不足，未生成报告")
            except Exception as e:
                print(f"\n⚠️  生成RL训练报告时出错: {e}")

        # Save only the best RL agent weights at the end of the simulation
        try:
            if hasattr(top, 'use_rl') and top.use_rl:
                weights_dir = os.path.join(out_dir, 'weights')
                os.makedirs(weights_dir, exist_ok=True)
                best_creature = None
                best_score = float('-inf')
                for c in getattr(top, 'creatures', []):
                    if hasattr(c, 'use_rl') and c.use_rl and hasattr(c, 'local_agent'):
                        # Prefer fitness total_reward, fallback to lifetime reward
                        score = None
                        try:
                            if hasattr(c, 'get_fitness_metrics'):
                                score = float(c.get_fitness_metrics().total_reward)
                        except Exception:
                            score = None
                        if score is None:
                            score = float(getattr(c, '_total_lifetime_reward', 0.0))
                        if score > best_score:
                            best_score = score
                            best_creature = c
                saved_path = None
                if best_creature is not None:
                    saved_path = os.path.join(weights_dir, 'best_agent.json')
                    try:
                        best_creature.local_agent.save(saved_path)
                    except Exception as e:
                        print(f"\n⚠️  保存最佳权重失败: {e}")
                        saved_path = None
                if saved_path:
                    print(f"\n✅ 已保存最佳权重 -> {os.path.abspath(saved_path)}  (score={best_score:.3f})")
                else:
                    print("\n⚠️  未能保存最佳权重（无可用代理或保存失败）")
        except Exception as e:
            print(f"\n⚠️  保存权重时出错: {e}")
        
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
