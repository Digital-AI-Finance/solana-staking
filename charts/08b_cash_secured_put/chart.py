"""
Cash-Secured Put Strategy Payoff Diagram

Short put for SOL accumulation at lower prices.
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
K = 120   # Put strike (buy SOL at $120)
premium = 8  # Premium received per SOL

# Price range
S_T = np.linspace(80, 220, 200)

# Short put payoff
# If assigned: buy at K, receive premium -> effective cost = K - premium
# Payoff = premium if S_T > K, else premium - (K - S_T) = S_T - K + premium
short_put_payoff = np.where(S_T >= K, premium, premium - (K - S_T))

# Breakeven point
breakeven = K - premium

# Maximum profit (premium kept)
max_profit = premium

# Maximum loss (if SOL goes to 0)
max_loss = K - premium

fig, ax = plt.subplots(figsize=(10, 6))

# Plot payoff
ax.plot(S_T, short_put_payoff, '-', color=COLORS['SOL_PURPLE'],
        linewidth=3, label='Cash-Secured Put')

# Zero line
ax.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1, alpha=0.5)

# Key price levels
ax.axvline(x=S0, color=COLORS['ACCENT_BLUE'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=K, color=COLORS['WARNING_ORANGE'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=breakeven, color=COLORS['SOL_GREEN'], linestyle=':', linewidth=1.5, alpha=0.7)

# Fill profit/loss regions
ax.fill_between(S_T, short_put_payoff, 0,
                where=(short_put_payoff > 0),
                alpha=0.2, color=COLORS['SOL_GREEN'], label='Profit Zone')
ax.fill_between(S_T, short_put_payoff, 0,
                where=(short_put_payoff < 0),
                alpha=0.2, color=COLORS['ERROR_RED'], label='Loss Zone')

# Annotations
ax.annotate(f'Max Profit: ${max_profit}',
            xy=(K + 40, max_profit), fontsize=11, fontweight='bold',
            color=COLORS['SOL_PURPLE'])

ax.annotate(f'Breakeven: ${breakeven}',
            xy=(breakeven, 0), xytext=(breakeven - 15, 15),
            fontsize=11, color=COLORS['SOL_GREEN'],
            arrowprops=dict(arrowstyle='->', color=COLORS['SOL_GREEN']))

ax.annotate(f'Current: ${S0}',
            xy=(S0, -35), fontsize=10, ha='center',
            color=COLORS['ACCENT_BLUE'])

ax.annotate(f'Strike: ${K}',
            xy=(K, -35), fontsize=10, ha='center',
            color=COLORS['WARNING_ORANGE'])

# Assignment region
ax.annotate('Assigned\n(Buy SOL)', xy=(100, -20), fontsize=10, ha='center',
            color=COLORS['ERROR_RED'], fontweight='bold')
ax.annotate('Expires\nWorthless', xy=(180, max_profit/2), fontsize=10, ha='center',
            color=COLORS['SOL_GREEN'], fontweight='bold')

# Strategy summary box
effective_cost = K - premium
summary = (f"Cash-Secured Put\n"
           f"-------------------\n"
           f"Strike: ${K} ({((S0-K)/S0)*100:.0f}% below spot)\n"
           f"Premium: ${premium}/SOL\n"
           f"Cash Reserved: ${K}/SOL\n"
           f"Effective Cost: ${effective_cost}/SOL\n"
           f"Breakeven: ${breakeven}\n"
           f"Max Profit: ${max_profit}")
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('SOL Price at Expiration ($)', fontsize=14)
ax.set_ylabel('Profit/Loss per SOL ($)', fontsize=14)
ax.set_title('Cash-Secured Put: Accumulate SOL at Discount', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(80, 220)
ax.set_ylim(-60, 30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 08b_cash_secured_put/chart.pdf")
