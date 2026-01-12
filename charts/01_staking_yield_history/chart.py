"""
Solana Staking Yield History - Historical APY and Inflation Schedule
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Solana inflation schedule parameters
INITIAL_INFLATION = 0.08  # 8%
DISINFLATION_RATE = 0.15  # 15% annual decrease
TERMINAL_INFLATION = 0.015  # 1.5%
STAKED_RATIO = 0.68  # ~68% staked

# Generate data from Feb 2021 (epoch 150) to 2032
years = np.arange(0, 12, 0.1)  # 0 to 12 years
dates = 2021 + years

# Calculate inflation rate
inflation = np.maximum(
    INITIAL_INFLATION * ((1 - DISINFLATION_RATE) ** years),
    TERMINAL_INFLATION
)

# Approximate staking APY (inflation / staked_ratio, simplified)
# In reality, varies with staked percentage
staking_apy = inflation / STAKED_RATIO

# Historical staked ratio (approximation)
staked_ratio_history = np.clip(0.60 + 0.02 * years, 0.60, 0.72)
staking_apy_adjusted = inflation / staked_ratio_history

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot inflation rate
color1 = COLORS['SOL_PURPLE']
ax1.plot(dates, inflation * 100, color=color1, linewidth=2.5, label='Inflation Rate')
ax1.fill_between(dates, 0, inflation * 100, color=color1, alpha=0.15)
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Inflation Rate (%)', color=color1, fontsize=14)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 9)

# Secondary axis for staking APY
ax2 = ax1.twinx()
color2 = COLORS['SOL_GREEN']
ax2.plot(dates, staking_apy_adjusted * 100, color=color2, linewidth=2.5,
         linestyle='--', label='Staking APY')
ax2.set_ylabel('Staking APY (%)', color=color2, fontsize=14)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 15)

# Add key milestones
ax1.axvline(x=2025, color=COLORS['TEXT_SECONDARY'], linestyle=':', alpha=0.7)
ax1.annotate('Current\n(2025)', xy=(2025, 4.5), fontsize=10,
             ha='center', color=COLORS['TEXT_SECONDARY'])

ax1.axvline(x=2032, color=COLORS['TEXT_SECONDARY'], linestyle=':', alpha=0.7)
ax1.annotate('Terminal\nRate', xy=(2032, 2.0), fontsize=10,
             ha='center', color=COLORS['TEXT_SECONDARY'])

# Add current values annotation
current_idx = np.abs(dates - 2025).argmin()
ax1.annotate(f'{inflation[current_idx]*100:.1f}%',
             xy=(2025, inflation[current_idx]*100),
             xytext=(2023.5, inflation[current_idx]*100 + 1.5),
             fontsize=11, color=color1,
             arrowprops=dict(arrowstyle='->', color=color1, lw=1.5))

ax2.annotate(f'{staking_apy_adjusted[current_idx]*100:.1f}%',
             xy=(2025, staking_apy_adjusted[current_idx]*100),
             xytext=(2026.5, staking_apy_adjusted[current_idx]*100 + 2),
             fontsize=11, color=color2,
             arrowprops=dict(arrowstyle='->', color=color2, lw=1.5))

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

# Title and grid
ax1.set_title('Solana Inflation Schedule & Staking Yield', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2021, 2033)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 01_staking_yield_history/chart.pdf")
