"""RL training visualization and monitoring tools."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from collections import deque
import time

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RLTrainingVisualizer:
    """Real-time and offline visualization for RL training."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize visualizer.
        
        Args:
            max_history: Maximum number of data points to keep in history
        """
        self.max_history = max_history
        
        # Training history
        self.history = {
            'update_step': [],
            'timestamp': [],
            'average_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
            'buffer_size': [],
            'episode_rewards': deque(maxlen=1000),  # Last 1000 episode rewards
            'episode_lengths': deque(maxlen=1000),  # Last 1000 episode lengths
        }
        
        # Running averages
        self.running_avg_reward = deque(maxlen=100)
        self.running_avg_loss = deque(maxlen=100)
    
    def record_training_step(self, stats: Dict, update_step: int):
        """Record a training step.
        
        Args:
            stats: Training statistics dictionary
            update_step: Current training step number
        """
        timestamp = time.time()
        
        self.history['update_step'].append(update_step)
        self.history['timestamp'].append(timestamp)
        
        # Record reward
        if 'average_reward' in stats:
            self.history['average_reward'].append(stats['average_reward'])
            self.running_avg_reward.append(stats['average_reward'])
        elif 'recent_avg_reward' in stats:
            self.history['average_reward'].append(stats['recent_avg_reward'])
            self.running_avg_reward.append(stats['recent_avg_reward'])
        
        # Record losses
        if 'policy_loss' in stats:
            self.history['policy_loss'].append(stats['policy_loss'])
        if 'value_loss' in stats:
            self.history['value_loss'].append(stats['value_loss'])
        if 'entropy' in stats:
            self.history['entropy'].append(stats['entropy'])
        if 'clip_fraction' in stats:
            self.history['clip_fraction'].append(stats['clip_fraction'])
        
        # Record buffer size
        if 'buffer_size' in stats:
            self.history['buffer_size'].append(stats['buffer_size'])
        
        # Trim history if too long
        if len(self.history['update_step']) > self.max_history:
            for key in self.history:
                if isinstance(self.history[key], list):
                    self.history[key] = self.history[key][-self.max_history:]
    
    def record_episode(self, reward: float, length: int):
        """Record an episode completion.
        
        Args:
            reward: Total episode reward
            length: Episode length in steps
        """
        self.history['episode_rewards'].append(reward)
        self.history['episode_lengths'].append(length)
    
    def get_recent_statistics(self) -> Dict:
        """Get recent training statistics.
        
        Returns:
            Dictionary of recent statistics
        """
        stats = {
            'avg_reward': 0.0,
            'avg_policy_loss': 0.0,
            'avg_value_loss': 0.0,
            'total_updates': len(self.history['update_step']),
            'recent_episode_reward': 0.0,
            'recent_episode_length': 0.0,
        }
        
        if self.running_avg_reward:
            stats['avg_reward'] = np.mean(self.running_avg_reward)
        
        if self.history['policy_loss']:
            stats['avg_policy_loss'] = np.mean(self.history['policy_loss'][-100:])
        if self.history['value_loss']:
            stats['avg_value_loss'] = np.mean(self.history['value_loss'][-100:])
        
        if self.history['episode_rewards']:
            stats['recent_episode_reward'] = np.mean(list(self.history['episode_rewards'])[-100:])
            stats['recent_episode_length'] = np.mean(list(self.history['episode_lengths'])[-100:])
        
        return stats
    
    def generate_report(self, output_path: str = "rl_training_report.png") -> bool:
        """Generate comprehensive training report.
        
        Args:
            output_path: Path to save the report image
            
        Returns:
            True if successful, False otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，无法生成报告")
            return False
        
        if not self.history['update_step']:
            print("没有训练数据可供可视化")
            return False
        
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle('RL Training Report', fontsize=16, fontweight='bold')
            
            update_steps = np.array(self.history['update_step'])
            
            # 1. Average reward over time（平滑+置信带）
            ax1 = plt.subplot(3, 3, 1)
            if self.history['average_reward']:
                rewards = np.array(self.history['average_reward'])
                steps_r = update_steps[:len(rewards)]
                ax1.plot(steps_r, rewards, color='#4C78A8', linewidth=1.2, alpha=0.6, label='Average Reward')
                # 滑动均值与±1σ置信带
                if len(rewards) >= 10:
                    window = max(10, min(100, len(rewards) // 5))
                    kernel = np.ones(window) / window
                    ma = np.convolve(rewards, kernel, mode='valid')
                    # 近似滑动标准差（Welford更精确，但代价高；这里用简化版）
                    sq = np.convolve(rewards**2, kernel, mode='valid')
                    std = np.sqrt(np.maximum(sq - ma**2, 1e-8))
                    s = steps_r[window-1:window-1+len(ma)]
                    ax1.plot(s, ma, color='#E45756', linewidth=2.0, label=f'MA({window})')
                    ax1.fill_between(s, ma-std, ma+std, color='#E45756', alpha=0.15, label='±1σ')
                ax1.set_xlabel('Update Step')
                ax1.set_ylabel('Average Reward')
                ax1.set_title('Average Reward (Smoothed with ±1σ)')
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right', framealpha=0.3)
            
            # 2. Policy loss
            ax2 = plt.subplot(3, 3, 2)
            if self.history['policy_loss']:
                losses = np.array(self.history['policy_loss'])
                ax2.plot(update_steps[:len(losses)], losses, 'g-', linewidth=1.5, label='Policy Loss')
                ax2.set_xlabel('Update Step')
                ax2.set_ylabel('Loss')
                ax2.set_title('Policy Loss')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            # 3. Value loss
            ax3 = plt.subplot(3, 3, 3)
            if self.history['value_loss']:
                losses = np.array(self.history['value_loss'])
                ax3.plot(update_steps[:len(losses)], losses, 'orange', linewidth=1.5, label='Value Loss')
                ax3.set_xlabel('Update Step')
                ax3.set_ylabel('Loss')
                ax3.set_title('Value Loss')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            # 4. Entropy
            ax4 = plt.subplot(3, 3, 4)
            if self.history['entropy']:
                entropies = np.array(self.history['entropy'])
                ax4.plot(update_steps[:len(entropies)], entropies, 'm-', linewidth=1.5, label='Entropy')
                ax4.set_xlabel('Update Step')
                ax4.set_ylabel('Entropy')
                ax4.set_title('Policy Entropy (Exploration)')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            
            # 5. Policy loss & Clip fraction（同图双轴）
            ax5 = plt.subplot(3, 3, 5)
            has_pol = bool(self.history['policy_loss'])
            has_clip = bool(self.history['clip_fraction'])
            if has_pol or has_clip:
                if has_pol:
                    losses = np.array(self.history['policy_loss'])
                    steps_l = update_steps[:len(losses)]
                    ax5.plot(steps_l, losses, color='#72B7B2', linewidth=1.5, label='Policy Loss')
                    ax5.set_ylabel('Policy Loss')
                ax5.set_xlabel('Update Step')
                ax5.set_title('Policy Loss & Clip Fraction')
                ax5.grid(True, alpha=0.3)
                ax5.legend(loc='upper left', framealpha=0.3)
                if has_clip:
                    clips = np.array(self.history['clip_fraction'])
                    steps_c = update_steps[:len(clips)]
                    ax5b = ax5.twinx()
                    ax5b.plot(steps_c, clips, color='#54A24B', linewidth=1.2, alpha=0.8, label='Clip Fraction')
                    ax5b.set_ylabel('Clip Fraction')
                    ax5b.legend(loc='upper right', framealpha=0.3)
            
            # 6. Buffer size
            ax6 = plt.subplot(3, 3, 6)
            if self.history['buffer_size']:
                sizes = np.array(self.history['buffer_size'])
                ax6.plot(update_steps[:len(sizes)], sizes, 'y-', linewidth=1.5, label='Buffer Size')
                ax6.set_xlabel('Update Step')
                ax6.set_ylabel('Size')
                ax6.set_title('Experience Buffer Size')
                ax6.grid(True, alpha=0.3)
                ax6.legend()
            
            # 7. Episode rewards distribution
            ax7 = plt.subplot(3, 3, 7)
            if self.history['episode_rewards']:
                episode_rewards = list(self.history['episode_rewards'])
                ax7.hist(episode_rewards, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
                ax7.set_xlabel('Episode Reward')
                ax7.set_ylabel('Frequency')
                ax7.set_title('Episode Reward Distribution')
                ax7.grid(True, alpha=0.3)
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards)
                    ax7.axvline(mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
                    ax7.legend()
            
            # 8. Episode lengths
            ax8 = plt.subplot(3, 3, 8)
            if self.history['episode_lengths']:
                episode_lengths = list(self.history['episode_lengths'])
                ax8.hist(episode_lengths, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
                ax8.set_xlabel('Episode Length (steps)')
                ax8.set_ylabel('Frequency')
                ax8.set_title('Episode Length Distribution')
                ax8.grid(True, alpha=0.3)
                if episode_lengths:
                    mean_length = np.mean(episode_lengths)
                    ax8.axvline(mean_length, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
                    ax8.legend()
            
            # 9. Training summary statistics
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            summary_text = self._generate_summary_text()
            ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', transform=ax9.transAxes)
            
            # 导出 CSV（便于对比分析）
            try:
                import csv
                csv_path = output_path.replace('.png', '_metrics.csv')
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = ['update_step','average_reward','policy_loss','value_loss','entropy','clip_fraction','buffer_size']
                    writer.writerow(headers)
                    n = len(self.history['update_step'])
                    for i in range(n):
                        row = [
                            self.history['update_step'][i],
                            self.history['average_reward'][i] if i < len(self.history['average_reward']) else '',
                            self.history['policy_loss'][i] if i < len(self.history['policy_loss']) else '',
                            self.history['value_loss'][i] if i < len(self.history['value_loss']) else '',
                            self.history['entropy'][i] if i < len(self.history['entropy']) else '',
                            self.history['clip_fraction'][i] if i < len(self.history['clip_fraction']) else '',
                            self.history['buffer_size'][i] if i < len(self.history['buffer_size']) else '',
                        ]
                        writer.writerow(row)
            except Exception:
                pass

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return False
    
    def _generate_summary_text(self) -> str:
        """Generate summary text for report.
        
        Returns:
            Summary text string
        """
        lines = []
        lines.append("Training Summary")
        lines.append("=" * 30)
        lines.append(f"Total Updates: {len(self.history['update_step'])}")
        
        if self.history['average_reward']:
            recent_rewards = self.history['average_reward'][-100:]
            lines.append(f"Recent Avg Reward: {np.mean(recent_rewards):.3f}")
            lines.append(f"Max Reward: {np.max(self.history['average_reward']):.3f}")
            lines.append(f"Min Reward: {np.min(self.history['average_reward']):.3f}")
        
        if self.history['policy_loss']:
            recent_loss = self.history['policy_loss'][-100:]
            lines.append(f"Recent Policy Loss: {np.mean(recent_loss):.3f}")
        
        if self.history['episode_rewards']:
            episode_rewards = list(self.history['episode_rewards'])
            lines.append(f"Episodes Recorded: {len(episode_rewards)}")
            lines.append(f"Avg Episode Reward: {np.mean(episode_rewards):.3f}")
            lines.append(f"Best Episode: {np.max(episode_rewards):.3f}")
        
        if self.history['episode_lengths']:
            episode_lengths = list(self.history['episode_lengths'])
            lines.append(f"Avg Episode Length: {np.mean(episode_lengths):.1f}")
        
        if self.history['buffer_size']:
            lines.append(f"Current Buffer Size: {self.history['buffer_size'][-1]}")
        
        return "\n".join(lines)
    
    def get_simple_stats_text(self) -> str:
        """Get simple statistics as text for in-game display.
        
        Returns:
            Statistics text string
        """
        stats = self.get_recent_statistics()
        
        lines = []
        lines.append(f"Updates: {stats['total_updates']}")
        lines.append(f"Avg Reward: {stats['avg_reward']:.2f}")
        
        if stats['avg_policy_loss'] > 0:
            lines.append(f"Policy Loss: {stats['avg_policy_loss']:.3f}")
        
        if stats['recent_episode_reward'] > 0:
            lines.append(f"Episode Reward: {stats['recent_episode_reward']:.2f}")
        
        return " | ".join(lines)


def create_simple_visualizer() -> RLTrainingVisualizer:
    """Create a simple visualizer instance.
    
    Returns:
        RLTrainingVisualizer instance
    """
    return RLTrainingVisualizer(max_history=5000)

