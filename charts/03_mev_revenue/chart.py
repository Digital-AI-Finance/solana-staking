"""
MEV Revenue Analysis - Jito Tips Distribution
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Simulated MEV tip distribution data (based on stake size)
np.random.seed(42)

# Stake tiers and corresponding MEV tips
stake_tiers = ['<100K', '100K-500K', '500K-1M', '1M-5M', '5M-10M', '>10M']
stake_midpoints = [50000, 300000, 750000, 3000000, 7500000, 15000000]

# MEV tips per day (SOL) - scales roughly with stake weight
# Assumes ~0.01 SOL average tip per slot, validator share ~3%
mev_per_day = np.array([0.5, 3.0, 7.5, 30, 75, 150])

# Annual MEV revenue (SOL)
mev_annual = mev_per_day * 365

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: MEV revenue by stake tier (bar chart)
colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(stake_tiers)))

bars = ax1.bar(stake_tiers, mev_annual, color=colors, edgecolor='white', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, mev_annual):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('Validator Stake Tier (SOL)', fontsize=13)
ax1.set_ylabel('Annual MEV Revenue (SOL)', fontsize=13)
ax1.set_title('MEV Revenue by Stake Size', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=30)

# Add annotation
ax1.annotate('MEV scales with\nstake weight',
             xy=(4, 27000), fontsize=10, color=COLORS['TEXT_SECONDARY'],
             ha='center')

# Right: Cumulative MEV distribution (Lorenz-like curve)
# Top validators capture most MEV
validator_percentile = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
mev_cumulative = np.array([45, 65, 78, 86, 91, 94, 96, 98, 99, 100])

ax2.fill_between(validator_percentile, mev_cumulative, alpha=0.3, color=COLORS['SOL_PURPLE'])
ax2.plot(validator_percentile, mev_cumulative, color=COLORS['SOL_PURPLE'],
         linewidth=2.5, marker='o', markersize=6)

# Perfect equality line
ax2.plot([0, 100], [0, 100], '--', color=COLORS['TEXT_SECONDARY'],
         linewidth=1.5, label='Perfect Equality')

ax2.set_xlabel('Validator Percentile (by stake)', fontsize=13)
ax2.set_ylabel('Cumulative MEV Share (%)', fontsize=13)
ax2.set_title('MEV Concentration Curve', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 105)
ax2.legend(loc='lower right', fontsize=11)

# Add annotation for concentration
ax2.annotate('Top 10% of validators\ncapture 45% of MEV',
             xy=(10, 45), xytext=(35, 30),
             fontsize=10, color=COLORS['SOL_PURPLE'],
             arrowprops=dict(arrowstyle='->', color=COLORS['SOL_PURPLE'], lw=1.5))

plt.suptitle('Jito MEV Revenue Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 03_mev_revenue/chart.pdf")
