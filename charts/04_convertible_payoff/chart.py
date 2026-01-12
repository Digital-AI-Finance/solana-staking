"""
Convertible Bond Payoff Diagram - MicroStrategy Style for SOL
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Convertible bond parameters (MicroStrategy style)
FACE_VALUE = 1000  # Per $1000 face value
CONVERSION_PRICE = 200  # SOL conversion price
CONVERSION_RATIO = FACE_VALUE / CONVERSION_PRICE  # 5 SOL per $1000
CURRENT_SOL_PRICE = 150
CONVERSION_PREMIUM = (CONVERSION_PRICE - CURRENT_SOL_PRICE) / CURRENT_SOL_PRICE

# Price range for analysis
sol_prices = np.linspace(50, 400, 200)

# Payoff at maturity
face_value_line = FACE_VALUE * np.ones_like(sol_prices)
conversion_value = CONVERSION_RATIO * sol_prices
payoff_at_maturity = np.maximum(face_value_line, conversion_value)

# Bond floor (straight bond value at ~$750 for zero coupon 5yr at 5.5%)
bond_floor = 765 * np.ones_like(sol_prices)

# Theoretical convertible value (simplified - bond floor + option value)
# This curves above payoff before maturity due to time value
from scipy.stats import norm
T = 5.0  # 5 years to maturity
r = 0.045
sigma = 0.80

d1 = (np.log(sol_prices/CONVERSION_PRICE) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_value = sol_prices * norm.cdf(d1) - CONVERSION_PRICE * np.exp(-r*T) * norm.cdf(d2)
option_value = call_value * CONVERSION_RATIO

convertible_value = bond_floor + option_value

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(sol_prices, face_value_line, '--', color=COLORS['TEXT_SECONDARY'],
        linewidth=1.5, label='Face Value ($1,000)')
ax.plot(sol_prices, conversion_value, '--', color=COLORS['SOL_GREEN'],
        linewidth=2, label='Conversion Value')
ax.plot(sol_prices, bond_floor, '--', color=COLORS['WARNING_ORANGE'],
        linewidth=2, label='Bond Floor (~$765)')
ax.plot(sol_prices, payoff_at_maturity, '-', color=COLORS['ERROR_RED'],
        linewidth=2.5, label='Payoff at Maturity')
ax.plot(sol_prices, convertible_value, '-', color=COLORS['SOL_PURPLE'],
        linewidth=3, label='Convertible Value (Today)')

# Fill regions
ax.fill_between(sol_prices, bond_floor, convertible_value,
                where=(convertible_value > bond_floor),
                alpha=0.15, color=COLORS['SOL_PURPLE'], label='Option Premium')

# Mark key points
ax.axvline(x=CURRENT_SOL_PRICE, color=COLORS['ACCENT_BLUE'], linestyle=':',
           linewidth=1.5, alpha=0.7)
ax.axvline(x=CONVERSION_PRICE, color=COLORS['SOL_GREEN'], linestyle=':',
           linewidth=1.5, alpha=0.7)

# Annotations
ax.annotate(f'Current SOL\n${CURRENT_SOL_PRICE}',
            xy=(CURRENT_SOL_PRICE, 500), fontsize=10,
            ha='center', color=COLORS['ACCENT_BLUE'])

ax.annotate(f'Conversion Price\n${CONVERSION_PRICE}\n({CONVERSION_PREMIUM:.0%} premium)',
            xy=(CONVERSION_PRICE, 400), fontsize=10,
            ha='center', color=COLORS['SOL_GREEN'])

# Add conversion ratio annotation
ax.annotate(f'Conversion Ratio: {CONVERSION_RATIO:.1f} SOL per $1,000',
            xy=(300, 1600), fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('SOL Price at Maturity ($)', fontsize=14)
ax.set_ylabel('Convertible Value per $1,000 Face ($)', fontsize=14)
ax.set_title('Convertible Bond Payoff Diagram', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(50, 400)
ax.set_ylim(300, 2200)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 04_convertible_payoff/chart.pdf")
