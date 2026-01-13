"""
Technical Indicators for SOL Entry Signals

Shows RSI (14) and MACD for identifying opportunistic entry points.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

np.random.seed(42)

# Generate synthetic SOL price data (180 days)
days = 180
dates = np.arange(days)

# Create realistic price movement with trends
base_price = 150
returns = np.random.normal(0.001, 0.03, days)
# Add some trend changes
returns[30:60] = np.random.normal(-0.015, 0.025, 30)  # Downtrend
returns[60:90] = np.random.normal(0.002, 0.02, 30)    # Recovery
returns[120:150] = np.random.normal(-0.012, 0.03, 30) # Another dip
prices = base_price * np.exp(np.cumsum(returns))

# Calculate RSI (14-period)
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))

    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Smooth averages
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Fill initial values
    return rsi

# Calculate MACD
def calculate_ema(data, period):
    ema = np.zeros(len(data))
    ema[0] = data[0]
    multiplier = 2 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

ema12 = calculate_ema(prices, 12)
ema26 = calculate_ema(prices, 26)
macd_line = ema12 - ema26
signal_line = calculate_ema(macd_line, 9)
macd_histogram = macd_line - signal_line

rsi = calculate_rsi(prices)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8),
                                     gridspec_kw={'height_ratios': [2, 1, 1]})

# Price chart
ax1.plot(dates, prices, color=COLORS['SOL_PURPLE'], linewidth=2, label='SOL Price')
ax1.fill_between(dates, prices.min() * 0.95, prices, alpha=0.1, color=COLORS['SOL_PURPLE'])

# Mark oversold entry points (RSI < 30)
oversold_mask = rsi < 30
ax1.scatter(dates[oversold_mask], prices[oversold_mask],
            color=COLORS['SOL_GREEN'], s=100, zorder=5, marker='^',
            label='RSI < 30 (Oversold)')

ax1.set_ylabel('SOL Price ($)', fontsize=12)
ax1.set_title('SOL Price with Entry Signals', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, days)

# RSI chart
ax2.plot(dates, rsi, color=COLORS['ACCENT_BLUE'], linewidth=2)
ax2.axhline(y=70, color=COLORS['ERROR_RED'], linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=30, color=COLORS['SOL_GREEN'], linestyle='--', linewidth=1.5, alpha=0.7)
ax2.fill_between(dates, 30, rsi, where=(rsi < 30), alpha=0.3, color=COLORS['SOL_GREEN'])
ax2.fill_between(dates, 70, rsi, where=(rsi > 70), alpha=0.3, color=COLORS['ERROR_RED'])
ax2.set_ylabel('RSI (14)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.text(days * 0.98, 72, 'Overbought', fontsize=9, ha='right', color=COLORS['ERROR_RED'])
ax2.text(days * 0.98, 28, 'Oversold', fontsize=9, ha='right', color=COLORS['SOL_GREEN'])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, days)

# MACD chart
colors_hist = [COLORS['SOL_GREEN'] if h >= 0 else COLORS['ERROR_RED'] for h in macd_histogram]
ax3.bar(dates, macd_histogram, color=colors_hist, alpha=0.7, width=1)
ax3.plot(dates, macd_line, color=COLORS['SOL_PURPLE'], linewidth=1.5, label='MACD')
ax3.plot(dates, signal_line, color=COLORS['WARNING_ORANGE'], linewidth=1.5, label='Signal')
ax3.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1, alpha=0.5)
ax3.set_ylabel('MACD', fontsize=12)
ax3.set_xlabel('Days', fontsize=12)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, days)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 07a_technical_indicators/chart.pdf")
