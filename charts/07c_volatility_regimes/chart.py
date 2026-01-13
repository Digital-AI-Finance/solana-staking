"""
Volatility Regimes Analysis

Shows rolling volatility with regime classification and recommended position sizing.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

np.random.seed(42)

# Generate 18 months of synthetic price data
days = 540
dates = np.arange(days)

# Create price with varying volatility regimes
base_price = 150
returns = np.zeros(days)

# Different volatility regimes
returns[0:120] = np.random.normal(0.001, 0.025, 120)     # Low vol
returns[120:240] = np.random.normal(0.0005, 0.055, 120)  # High vol
returns[240:360] = np.random.normal(0.001, 0.03, 120)    # Medium vol
returns[360:420] = np.random.normal(-0.002, 0.07, 60)    # Crisis vol
returns[420:540] = np.random.normal(0.002, 0.035, 120)   # Recovery vol

prices = base_price * np.exp(np.cumsum(returns))

# Calculate 30-day rolling volatility (annualized)
window = 30
rolling_vol = np.zeros(days)
for i in range(window, days):
    rolling_vol[i] = np.std(returns[i-window:i]) * np.sqrt(365) * 100

rolling_vol[:window] = rolling_vol[window]

# Define regime thresholds
low_vol_threshold = 50
high_vol_threshold = 90

# Classify regimes
regime = np.zeros(days, dtype=int)  # 0=low, 1=medium, 2=high
regime[rolling_vol < low_vol_threshold] = 0
regime[(rolling_vol >= low_vol_threshold) & (rolling_vol < high_vol_threshold)] = 1
regime[rolling_vol >= high_vol_threshold] = 2

# Position sizing based on regime
position_size = np.where(regime == 0, 1.5, np.where(regime == 1, 1.0, 0.5))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8),
                                     gridspec_kw={'height_ratios': [2, 1, 1]})

# Price chart with regime coloring
ax1.plot(dates, prices, color=COLORS['SOL_PURPLE'], linewidth=2)
# Color background by regime
for i in range(len(dates) - 1):
    if regime[i] == 0:
        ax1.axvspan(dates[i], dates[i+1], alpha=0.15, color=COLORS['SOL_GREEN'])
    elif regime[i] == 2:
        ax1.axvspan(dates[i], dates[i+1], alpha=0.15, color=COLORS['ERROR_RED'])

ax1.set_ylabel('SOL Price ($)', fontsize=12)
ax1.set_title('SOL Price with Volatility Regime Classification', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, days)

# Legend for regimes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['SOL_GREEN'], alpha=0.3, label='Low Vol (<50%)'),
                   Patch(facecolor='white', alpha=0.3, label='Medium Vol (50-90%)'),
                   Patch(facecolor=COLORS['ERROR_RED'], alpha=0.3, label='High Vol (>90%)')]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Volatility chart
ax2.plot(dates, rolling_vol, color=COLORS['ACCENT_BLUE'], linewidth=2)
ax2.axhline(y=low_vol_threshold, color=COLORS['SOL_GREEN'], linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=high_vol_threshold, color=COLORS['ERROR_RED'], linestyle='--', linewidth=1.5, alpha=0.7)
ax2.fill_between(dates, 0, rolling_vol, where=(rolling_vol < low_vol_threshold),
                 alpha=0.3, color=COLORS['SOL_GREEN'])
ax2.fill_between(dates, 0, rolling_vol, where=(rolling_vol >= high_vol_threshold),
                 alpha=0.3, color=COLORS['ERROR_RED'])
ax2.set_ylabel('30-Day Volatility (%)', fontsize=12)
ax2.text(days * 0.98, low_vol_threshold - 5, 'Low Vol', fontsize=9, ha='right', color=COLORS['SOL_GREEN'])
ax2.text(days * 0.98, high_vol_threshold + 5, 'High Vol', fontsize=9, ha='right', color=COLORS['ERROR_RED'])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, days)
ax2.set_ylim(0, max(rolling_vol) * 1.1)

# Position sizing chart
colors_pos = [COLORS['SOL_GREEN'] if p == 1.5 else (COLORS['WARNING_ORANGE'] if p == 1.0 else COLORS['ERROR_RED'])
              for p in position_size]
ax3.bar(dates, position_size, color=colors_pos, width=1, alpha=0.7)
ax3.axhline(y=1.0, color=COLORS['TEXT_SECONDARY'], linestyle='-', linewidth=1)
ax3.set_ylabel('Position Size\n(vs Normal)', fontsize=12)
ax3.set_xlabel('Days (18-Month Period)', fontsize=12)
ax3.set_ylim(0, 2)
ax3.set_xlim(0, days)
ax3.grid(True, alpha=0.3)

# Position size legend
ax3.text(0.02, 0.85, 'Low Vol: 1.5x | Med Vol: 1.0x | High Vol: 0.5x',
        transform=ax3.transAxes, fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 07c_volatility_regimes/chart.pdf")
