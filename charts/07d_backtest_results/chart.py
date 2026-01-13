"""
Backtesting Results for Entry Signal Strategies

Shows performance metrics comparing different signal thresholds.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Simulated backtest results for different RSI thresholds
strategies = ['RSI < 25', 'RSI < 30', 'RSI < 35', 'RSI + MACD', 'RSI + Vol']
win_rates = [78, 72, 65, 82, 85]
avg_returns = [12.5, 8.2, 5.8, 10.8, 11.2]
num_signals = [8, 15, 28, 12, 10]
max_drawdowns = [18, 22, 28, 15, 14]

x = np.arange(len(strategies))
bar_width = 0.35

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

# Win Rate
bars1 = ax1.bar(x, win_rates, color=COLORS['SOL_GREEN'], alpha=0.8, edgecolor='white', linewidth=1.5)
ax1.axhline(y=50, color=COLORS['TEXT_SECONDARY'], linestyle='--', linewidth=1.5, alpha=0.5)
ax1.set_ylabel('Win Rate (%)', fontsize=12)
ax1.set_title('Win Rate by Strategy', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, rotation=15, ha='right', fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar, val in zip(bars1, win_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')

# Average Return
bars2 = ax2.bar(x, avg_returns, color=COLORS['ACCENT_BLUE'], alpha=0.8, edgecolor='white', linewidth=1.5)
ax2.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linestyle='-', linewidth=1)
ax2.set_ylabel('Avg 30-Day Return (%)', fontsize=12)
ax2.set_title('Average Return per Signal', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(strategies, rotation=15, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, avg_returns):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')

# Number of Signals
bars3 = ax3.bar(x, num_signals, color=COLORS['WARNING_ORANGE'], alpha=0.8, edgecolor='white', linewidth=1.5)
ax3.set_ylabel('Number of Signals\n(36-month period)', fontsize=12)
ax3.set_title('Signal Frequency', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(strategies, rotation=15, ha='right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, num_signals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}', ha='center', fontsize=10, fontweight='bold')

# Max Drawdown (lower is better)
bars4 = ax4.bar(x, max_drawdowns, color=COLORS['ERROR_RED'], alpha=0.8, edgecolor='white', linewidth=1.5)
ax4.set_ylabel('Max Drawdown (%)', fontsize=12)
ax4.set_title('Maximum Drawdown (Lower = Better)', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(strategies, rotation=15, ha='right', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars4, max_drawdowns):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')

# Highlight best strategy
best_idx = 4  # RSI + Vol
for ax, bars in [(ax1, bars1), (ax2, bars2), (ax4, bars4)]:
    bars[best_idx].set_edgecolor(COLORS['SOL_PURPLE'])
    bars[best_idx].set_linewidth(3)

# Add overall summary
fig.text(0.5, 0.02,
         'Best Strategy: RSI + Volatility Filter (85% win rate, 11.2% avg return, 14% max DD)',
         ha='center', fontsize=11, fontweight='bold', color=COLORS['SOL_PURPLE'],
         bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['SOL_PURPLE'], alpha=0.9))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 07d_backtest_results/chart.pdf")
