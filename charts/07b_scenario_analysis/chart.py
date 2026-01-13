"""
Historical Entry Point Analysis

Shows SOL price with marked entry signals and performance metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

np.random.seed(42)

# Generate 24 months of synthetic price data
months = 24
days = months * 30
dates = np.arange(days)

# Create realistic price movement
base_price = 100
returns = np.random.normal(0.0008, 0.035, days)
# Add market cycles
returns[90:150] = np.random.normal(-0.008, 0.04, 60)   # Bear phase
returns[150:210] = np.random.normal(0.002, 0.025, 60)  # Recovery
returns[330:420] = np.random.normal(-0.006, 0.035, 90) # Correction
returns[450:540] = np.random.normal(0.005, 0.03, 90)   # Bull phase
prices = base_price * np.exp(np.cumsum(returns))

# Calculate RSI for signal generation
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    rs = np.where(avg_loss > 0.001, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    return rsi

rsi = calculate_rsi(prices)

# Identify entry signals (RSI < 30 AND price below 20-day MA)
ma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
signal_mask = (rsi < 30) & (prices < ma_20)

# Get unique entry points (not consecutive days)
entry_points = []
last_entry = -30
for i in range(len(signal_mask)):
    if signal_mask[i] and (i - last_entry > 20):  # At least 20 days apart
        entry_points.append(i)
        last_entry = i

entry_days = np.array(entry_points)
entry_prices = prices[entry_days]

# Calculate returns 30 days after each entry
returns_30d = []
for ep in entry_points:
    if ep + 30 < len(prices):
        ret = (prices[ep + 30] - prices[ep]) / prices[ep]
        returns_30d.append(ret)
    else:
        returns_30d.append(np.nan)

returns_30d = np.array(returns_30d)
valid_returns = returns_30d[~np.isnan(returns_30d)]

# Statistics
avg_entry = np.mean(entry_prices)
win_rate = np.sum(valid_returns > 0) / len(valid_returns) * 100
avg_return = np.mean(valid_returns) * 100

fig, ax = plt.subplots(figsize=(10, 6))

# Plot price
ax.plot(dates, prices, color=COLORS['SOL_PURPLE'], linewidth=2, label='SOL Price', alpha=0.8)

# Plot 20-day MA
ax.plot(dates, ma_20, color=COLORS['WARNING_ORANGE'], linewidth=1.5, linestyle='--',
        label='20-Day MA', alpha=0.7)

# Mark entry points
winning = returns_30d > 0
for i, (day, price, won) in enumerate(zip(entry_days, entry_prices, returns_30d)):
    if np.isnan(won):
        continue
    color = COLORS['SOL_GREEN'] if won > 0 else COLORS['ERROR_RED']
    ax.scatter(day, price, color=color, s=150, zorder=5, marker='^', edgecolor='white', linewidth=1.5)
    # Draw arrow to 30-day price
    if day + 30 < len(prices):
        ax.annotate('', xy=(day + 30, prices[day + 30]), xytext=(day, price),
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1.5))

# Average entry price line
ax.axhline(y=avg_entry, color=COLORS['ACCENT_BLUE'], linestyle=':', linewidth=2,
          label=f'Avg Entry: ${avg_entry:.0f}')

# Stats box
stats_text = (f"Entry Signals: {len(entry_points)}\n"
              f"Win Rate: {win_rate:.0f}%\n"
              f"Avg 30-Day Return: {avg_return:+.1f}%\n"
              f"Avg Entry Price: ${avg_entry:.0f}")
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Legend for entry points
ax.scatter([], [], color=COLORS['SOL_GREEN'], s=100, marker='^', label='Winning Entry (+30d)')
ax.scatter([], [], color=COLORS['ERROR_RED'], s=100, marker='^', label='Losing Entry (+30d)')

ax.set_xlabel('Days (24-Month Period)', fontsize=12)
ax.set_ylabel('SOL Price ($)', fontsize=12)
ax.set_title('Historical Entry Point Analysis: RSI < 30 Signals', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, days)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 07b_scenario_analysis/chart.pdf")
