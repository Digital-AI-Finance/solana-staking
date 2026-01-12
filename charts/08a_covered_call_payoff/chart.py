"""
Covered Call Strategy Payoff Diagram
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Strategy parameters
S0 = 150  # Current SOL price
K = 180  # Strike price (20% OTM)
premium = 15  # Premium received per SOL

# Price range
S_T = np.linspace(80, 280, 200)

# Payoff calculations
# Long stock payoff
stock_payoff = S_T - S0

# Short call payoff (from writer's perspective)
short_call_payoff = premium - np.maximum(S_T - K, 0)

# Covered call = Long stock + Short call
covered_call_payoff = stock_payoff + short_call_payoff

# Breakeven point
breakeven = S0 - premium

# Maximum profit
max_profit = K - S0 + premium

fig, ax = plt.subplots(figsize=(10, 6))

# Plot payoffs
ax.plot(S_T, stock_payoff, '--', color=COLORS['ACCENT_BLUE'],
        linewidth=2, label=f'Long SOL (Entry: ${S0})')
ax.plot(S_T, short_call_payoff, '--', color=COLORS['WARNING_ORANGE'],
        linewidth=2, label=f'Short Call (Strike: ${K}, Premium: ${premium})')
ax.plot(S_T, covered_call_payoff, '-', color=COLORS['SOL_PURPLE'],
        linewidth=3, label='Covered Call (Net Position)')

# Zero line
ax.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1, alpha=0.5)

# Key price levels
ax.axvline(x=S0, color=COLORS['ACCENT_BLUE'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=K, color=COLORS['WARNING_ORANGE'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=breakeven, color=COLORS['SOL_GREEN'], linestyle=':', linewidth=1.5, alpha=0.7)

# Fill profit/loss regions
ax.fill_between(S_T, covered_call_payoff, 0,
                where=(covered_call_payoff > 0),
                alpha=0.2, color=COLORS['SOL_GREEN'], label='Profit Zone')
ax.fill_between(S_T, covered_call_payoff, 0,
                where=(covered_call_payoff < 0),
                alpha=0.2, color=COLORS['ERROR_RED'], label='Loss Zone')

# Annotations
ax.annotate(f'Max Profit: ${max_profit}',
            xy=(K + 30, max_profit), fontsize=11, fontweight='bold',
            color=COLORS['SOL_PURPLE'])

ax.annotate(f'Breakeven: ${breakeven}',
            xy=(breakeven, 0), xytext=(breakeven - 25, -30),
            fontsize=11, color=COLORS['SOL_GREEN'],
            arrowprops=dict(arrowstyle='->', color=COLORS['SOL_GREEN']))

ax.annotate(f'Entry: ${S0}',
            xy=(S0, -50), fontsize=10, ha='center',
            color=COLORS['ACCENT_BLUE'])

ax.annotate(f'Strike: ${K}',
            xy=(K, -50), fontsize=10, ha='center',
            color=COLORS['WARNING_ORANGE'])

# Strategy summary box
summary = (f"Covered Call Strategy\n"
           f"-------------------\n"
           f"Entry: Buy SOL @ ${S0}\n"
           f"Sell Call: Strike ${K}\n"
           f"Premium: ${premium}/SOL\n"
           f"Max Profit: ${max_profit}\n"
           f"Breakeven: ${breakeven}")
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('SOL Price at Expiration ($)', fontsize=14)
ax.set_ylabel('Profit/Loss per SOL ($)', fontsize=14)
ax.set_title('Covered Call Payoff: Yield Enhancement Strategy', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(80, 280)
ax.set_ylim(-80, 80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 08a_covered_call_payoff/chart.pdf")
