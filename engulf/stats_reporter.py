"""Statistics reporting and visualization for simulation results."""

from __future__ import annotations

from typing import Dict, List

# Set backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI required)

import matplotlib.pyplot as plt
import numpy as np


def generate_statistics_report(stats: Dict, output_path: str = "simulation_stats.png") -> None:
    """Generate comprehensive statistics report with charts.
    
    Args:
        stats: Statistics dictionary from TopDownScene
        output_path: Path to save the report image
    """
    if not stats['time_steps']:
        print("没有统计数据可供显示")
        return
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Simulation Statistics Report', fontsize=16, fontweight='bold')
    
    # Use default font (English labels for compatibility)
    plt.rcParams['axes.unicode_minus'] = False
    
    time_steps = stats['time_steps']
    
    # 1. Population over time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time_steps, stats['population'], 'b-', linewidth=1.5, label='Population')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Creature Count')
    ax1.set_title('Population Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Average speed over time
    if stats['avg_speed']:
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(time_steps[:len(stats['avg_speed'])], stats['avg_speed'], 'g-', linewidth=1.5, label='Avg Speed')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Speed (pixels/sec)')
        ax2.set_title('Average Speed')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Average age over time
    if stats['avg_age']:
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(time_steps[:len(stats['avg_age'])], stats['avg_age'], 'r-', linewidth=1.5, label='Avg Age')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Age (seconds)')
        ax3.set_title('Average Age')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Species count over time
    if stats['species_count']:
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(time_steps[:len(stats['species_count'])], stats['species_count'], 'm-', linewidth=1.5, label='Species')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Species Count')
        ax4.set_title('Species Diversity')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # 5. Births and Deaths summary
    ax5 = plt.subplot(3, 3, 5)
    events = ['Births', 'Deaths', 'Predations']
    counts = [stats['births_total'], stats['deaths_total'], stats['predations_total']]
    colors = ['#4CAF50', '#F44336', '#FF9800']
    bars = ax5.bar(events, counts, color=colors, alpha=0.7)
    ax5.set_ylabel('Count')
    ax5.set_title('Life Events')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 6. Population distribution by final count
    ax6 = plt.subplot(3, 3, 6)
    if stats['population']:
        final_pop = stats['population'][-1]
        initial_pop = stats['population'][0] if stats['population'] else 0
        pop_change = final_pop - initial_pop
        change_pct = (pop_change / initial_pop * 100) if initial_pop > 0 else 0
        
        labels = ['Initial', 'Final', 'Change']
        values = [initial_pop, final_pop, abs(pop_change)]
        colors_pie = ['#2196F3', '#4CAF50', '#FF9800']
        ax6.pie(values, labels=labels, autopct='%1.0f', colors=colors_pie, startangle=90)
        ax6.set_title(f'Population Change ({pop_change:+.0f}, {change_pct:+.1f}%)')
    
    # 7. Resource levels over time (top 5 terrain types)
    ax7 = plt.subplot(3, 3, 7)
    resource_data = {k: v for k, v in stats['terrain_resources'].items() if v}
    if resource_data:
        # Get top 5 by average resource value
        avg_resources = {k: np.mean(v) if v else 0.0 for k, v in resource_data.items()}
        top_5 = sorted(avg_resources.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for terrain_type, _ in top_5:
            if stats['terrain_resources'][terrain_type]:
                data_len = min(len(time_steps), len(stats['terrain_resources'][terrain_type]))
                ax7.plot(time_steps[:data_len], 
                        stats['terrain_resources'][terrain_type][:data_len],
                        label=terrain_type, linewidth=1.5, alpha=0.7)
        ax7.set_xlabel('Time (seconds)')
        ax7.set_ylabel('Resource Value')
        ax7.set_title('Top 5 Terrain Resources')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    
    # 8. Hazard levels over time (top 5 terrain types)
    ax8 = plt.subplot(3, 3, 8)
    hazard_data = {k: v for k, v in stats['terrain_hazards'].items() if v}
    if hazard_data:
        # Get top 5 by average hazard value
        avg_hazards = {k: np.mean(v) if v else 0.0 for k, v in hazard_data.items()}
        top_5_hazards = sorted(avg_hazards.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for terrain_type, _ in top_5_hazards:
            if stats['terrain_hazards'][terrain_type]:
                data_len = min(len(time_steps), len(stats['terrain_hazards'][terrain_type]))
                ax8.plot(time_steps[:data_len],
                        stats['terrain_hazards'][terrain_type][:data_len],
                        label=terrain_type, linewidth=1.5, alpha=0.7)
        ax8.set_xlabel('Time (seconds)')
        ax8.set_ylabel('Hazard Value')
        ax8.set_title('Top 5 Terrain Hazards')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = []
    summary_text.append("=== Simulation Summary ===\n")
    
    if stats['time_steps']:
        total_time = stats['time_steps'][-1]
        summary_text.append(f"Total Time: {total_time:.1f}s")
        summary_text.append(f"Final Population: {stats['population'][-1] if stats['population'] else 0}")
    
    summary_text.append(f"\nLife Events:")
    summary_text.append(f"  Births: {stats['births_total']}")
    summary_text.append(f"  Deaths: {stats['deaths_total']}")
    summary_text.append(f"  Predations: {stats['predations_total']}")
    
    if stats['avg_speed']:
        summary_text.append(f"\nSpeed Stats:")
        summary_text.append(f"  Avg: {np.mean(stats['avg_speed']):.1f} px/s")
        summary_text.append(f"  Max: {np.max(stats['avg_speed']):.1f} px/s")
    
    if stats['avg_age']:
        summary_text.append(f"\nAge Stats:")
        summary_text.append(f"  Avg: {np.mean(stats['avg_age']):.1f}s")
        summary_text.append(f"  Max: {np.max(stats['avg_age']):.1f}s")
    
    if stats['population']:
        summary_text.append(f"\nPopulation Stats:")
        summary_text.append(f"  Initial: {stats['population'][0]}")
        summary_text.append(f"  Final: {stats['population'][-1]}")
        summary_text.append(f"  Peak: {np.max(stats['population'])}")
        summary_text.append(f"  Min: {np.min(stats['population'])}")
    
    summary_text_str = "\n".join(summary_text)
    ax9.text(0.1, 0.95, summary_text_str, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"\n统计报告已保存到: {output_path}")
    
    # Also print summary to console (bilingual)
    print("\n" + "="*60)
    print("模拟统计摘要 / Simulation Summary")
    print("="*60)
    for line in summary_text:
        print(line)
    print("="*60)

