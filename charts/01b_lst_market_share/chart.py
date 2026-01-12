"""
Solana Liquid Staking Token (LST) Market Share - 2025
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS, CHART_COLORS

plt.rcParams.update(CHART_RCPARAMS)

# LST Market Data (2025) - Verified from research
lst_data = {
    'Jito (JitoSOL)': 17.6,  # Million SOL
    'Binance (bnSOL)': 8.16,
    'Marinade (mSOL)': 5.28,
    'Jupiter (jupSOL)': 3.88,
    'Blaze (bSOL)': 1.44,
    'Other LSTs': 3.64,
}

names = list(lst_data.keys())
values = list(lst_data.values())
total = sum(values)

# Calculate percentages
percentages = [v / total * 100 for v in values]

# Colors
colors = [
    COLORS['SOL_PURPLE'],     # Jito - dominant
    COLORS['WARNING_ORANGE'], # Binance
    COLORS['SOL_GREEN'],      # Marinade
    COLORS['ACCENT_BLUE'],    # Jupiter
    COLORS['ERROR_RED'],      # Blaze
    COLORS['TEXT_SECONDARY'], # Other
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: Pie chart
wedges, texts, autotexts = ax1.pie(
    values,
    labels=None,
    autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
    colors=colors,
    startangle=90,
    explode=[0.05 if i == 0 else 0 for i in range(len(values))],  # Explode Jito
    pctdistance=0.75,
    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
)

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

ax1.set_title('LST Market Share', fontsize=14, fontweight='bold')

# Center text
ax1.text(0, 0, f'{total:.1f}M\nSOL', ha='center', va='center',
         fontsize=14, fontweight='bold')

# Right: Horizontal bar chart with TVL
y_pos = np.arange(len(names))
bars = ax2.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1.5)

# Add value labels
for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}M ({pct:.0f}%)',
             va='center', fontsize=11)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(names, fontsize=12)
ax2.set_xlabel('Total Value Locked (Million SOL)', fontsize=13)
ax2.set_title('LST TVL Breakdown', fontsize=14, fontweight='bold')
ax2.set_xlim(0, max(values) * 1.35)
ax2.invert_yaxis()  # Largest at top

# Add annotation for Jito dominance
ax2.annotate('Jito dominates with\n~44% market share',
             xy=(17.6, 0), xytext=(12, 2.5),
             fontsize=10, color=COLORS['SOL_PURPLE'],
             arrowprops=dict(arrowstyle='->', color=COLORS['SOL_PURPLE'], lw=1.5))

plt.suptitle('Solana Liquid Staking Landscape (2025)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 01b_lst_market_share/chart.pdf")
