"""
MicroStrategy BTC Accumulation vs SOL Strategy Comparison

Shows MSTR's Bitcoin accumulation approach applied to SOL.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Timeline (quarters from Q3 2020)
quarters = ['Q3\'20', 'Q4\'20', 'Q1\'21', 'Q2\'21', 'Q3\'21', 'Q4\'21',
            'Q1\'22', 'Q2\'22', 'Q3\'22', 'Q4\'22', 'Q1\'23', 'Q2\'23',
            'Q3\'23', 'Q4\'23', 'Q1\'24', 'Q2\'24', 'Q3\'24', 'Q4\'24', 'Q1\'25']
x = np.arange(len(quarters))

# MSTR BTC holdings (thousands)
mstr_btc = [21.5, 70.5, 91.1, 105.1, 114.0, 125.1, 129.2, 129.7, 130.0, 132.5,
            140.0, 152.8, 158.2, 189.2, 214.4, 252.2, 331.2, 447.5, 500.0]

# Hypothetical SOL accumulation (following similar pattern, thousands SOL)
sol_acc = [50, 180, 350, 520, 680, 850, 980, 1050, 1120, 1280,
           1450, 1680, 1850, 2200, 2650, 3200, 4000, 5200, 6500]

# Corresponding prices (for context)
btc_prices = [11500, 29000, 58000, 35000, 44000, 47000, 38000, 21000, 19500, 16500,
              24000, 30000, 27000, 42000, 68000, 61000, 58000, 95000, 100000]

sol_prices = [3, 4, 18, 35, 150, 180, 100, 40, 32, 13,
              22, 20, 18, 100, 170, 140, 145, 220, 200]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# MSTR BTC Accumulation
color_btc = '#F7931A'  # Bitcoin orange
ax1.bar(x, mstr_btc, color=color_btc, alpha=0.8, edgecolor='white', linewidth=1)
ax1.set_ylabel('BTC Holdings (thousands)', fontsize=12, color=color_btc)
ax1.tick_params(axis='y', labelcolor=color_btc)
ax1.set_title('MicroStrategy Bitcoin Accumulation (2020-2025)', fontsize=14, fontweight='bold')

# Add BTC price on secondary axis
ax1b = ax1.twinx()
ax1b.plot(x, np.array(btc_prices)/1000, color='gray', linewidth=2, linestyle='--', marker='o', markersize=4)
ax1b.set_ylabel('BTC Price ($K)', fontsize=12, color='gray')
ax1b.tick_params(axis='y', labelcolor='gray')

ax1.set_xticks(x[::2])
ax1.set_xticklabels(quarters[::2], rotation=45, ha='right', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Stats box for MSTR
mstr_stats = (f"Total BTC: {mstr_btc[-1]:.0f}K\n"
              f"Avg Cost: ~$35K/BTC\n"
              f"Total Invested: ~$17.5B")
ax1.text(0.02, 0.98, mstr_stats, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# SOL Accumulation (hypothetical)
ax2.bar(x, sol_acc, color=COLORS['SOL_PURPLE'], alpha=0.8, edgecolor='white', linewidth=1)
ax2.set_ylabel('SOL Holdings (thousands)', fontsize=12, color=COLORS['SOL_PURPLE'])
ax2.tick_params(axis='y', labelcolor=COLORS['SOL_PURPLE'])
ax2.set_title('Hypothetical SOL Accumulation (Same Capital, Same Strategy)', fontsize=14, fontweight='bold')

# Add SOL price on secondary axis
ax2b = ax2.twinx()
ax2b.plot(x, sol_prices, color='gray', linewidth=2, linestyle='--', marker='o', markersize=4)
ax2b.set_ylabel('SOL Price ($)', fontsize=12, color='gray')
ax2b.tick_params(axis='y', labelcolor='gray')

ax2.set_xticks(x[::2])
ax2.set_xticklabels(quarters[::2], rotation=45, ha='right', fontsize=9)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Stats box for SOL
sol_value = sol_acc[-1] * 1000 * sol_prices[-1] / 1e9
sol_stats = (f"Total SOL: {sol_acc[-1]/1000:.1f}M\n"
             f"Avg Cost: ~$40/SOL\n"
             f"Current Value: ~${sol_value:.1f}B")
ax2.text(0.02, 0.98, sol_stats, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 03a_mstr_btc_comparison/chart.pdf")
