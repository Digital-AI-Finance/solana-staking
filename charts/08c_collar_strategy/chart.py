"""
Collar Strategy Payoff Diagram

Long Stock + Long Put + Short Call
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
K_put = 130  # Put strike (downside protection)
K_call = 180  # Call strike (upside cap)
put_premium = 12  # Premium paid for put
call_premium = 15  # Premium received for call
net_credit = call_premium - put_premium  # Net premium (credit or debit)

# Price range
S_T = np.linspace(80, 250, 200)

# Payoff calculations
# Long stock payoff
stock_payoff = S_T - S0

# Long put payoff
long_put_payoff = np.maximum(K_put - S_T, 0) - put_premium

# Short call payoff
short_call_payoff = call_premium - np.maximum(S_T - K_call, 0)

# Collar = Long stock + Long put + Short call
collar_payoff = stock_payoff + long_put_payoff + short_call_payoff

# Key metrics
max_profit = K_call - S0 + net_credit
max_loss = S0 - K_put - net_credit
breakeven = S0 - net_credit

fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual components (dashed)
ax.plot(S_T, stock_payoff, '--', color=COLORS['ACCENT_BLUE'],
        linewidth=1.5, alpha=0.6, label=f'Long SOL (Entry: ${S0})')
ax.plot(S_T, long_put_payoff, '--', color=COLORS['SOL_GREEN'],
        linewidth=1.5, alpha=0.6, label=f'Long Put (K={K_put}, P=${put_premium})')
ax.plot(S_T, short_call_payoff, '--', color=COLORS['WARNING_ORANGE'],
        linewidth=1.5, alpha=0.6, label=f'Short Call (K={K_call}, P=${call_premium})')

# Plot collar (solid, bold)
ax.plot(S_T, collar_payoff, '-', color=COLORS['SOL_PURPLE'],
        linewidth=3, label='Collar (Net Position)')

# Zero line
ax.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1, alpha=0.5)

# Key price levels
ax.axvline(x=K_put, color=COLORS['SOL_GREEN'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=S0, color=COLORS['ACCENT_BLUE'], linestyle=':', linewidth=1.5, alpha=0.7)
ax.axvline(x=K_call, color=COLORS['WARNING_ORANGE'], linestyle=':', linewidth=1.5, alpha=0.7)

# Fill profit/loss regions
ax.fill_between(S_T, collar_payoff, 0,
                where=(collar_payoff > 0),
                alpha=0.2, color=COLORS['SOL_GREEN'])
ax.fill_between(S_T, collar_payoff, 0,
                where=(collar_payoff < 0),
                alpha=0.2, color=COLORS['ERROR_RED'])

# Annotations
ax.annotate(f'Max Profit: ${max_profit:.0f}',
            xy=(K_call + 20, max_profit), fontsize=11, fontweight='bold',
            color=COLORS['SOL_PURPLE'])

ax.annotate(f'Max Loss: ${max_loss:.0f}',
            xy=(K_put - 25, -max_loss), fontsize=11, fontweight='bold',
            color=COLORS['ERROR_RED'])

ax.annotate(f'Put Strike\n${K_put}',
            xy=(K_put, -55), fontsize=9, ha='center',
            color=COLORS['SOL_GREEN'])

ax.annotate(f'Entry\n${S0}',
            xy=(S0, -55), fontsize=9, ha='center',
            color=COLORS['ACCENT_BLUE'])

ax.annotate(f'Call Strike\n${K_call}',
            xy=(K_call, -55), fontsize=9, ha='center',
            color=COLORS['WARNING_ORANGE'])

# Strategy summary box
summary = (f"Collar Strategy\n"
           f"-------------------\n"
           f"Long SOL @ ${S0}\n"
           f"Long Put K=${K_put} (-${put_premium})\n"
           f"Short Call K=${K_call} (+${call_premium})\n"
           f"Net Premium: +${net_credit}\n"
           f"Max Profit: ${max_profit:.0f}\n"
           f"Max Loss: ${max_loss:.0f}\n"
           f"Protected Range: ${K_put}-${K_call}")
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('SOL Price at Expiration ($)', fontsize=14)
ax.set_ylabel('Profit/Loss per SOL ($)', fontsize=14)
ax.set_title('Collar Strategy: Downside Protection with Capped Upside', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(80, 250)
ax.set_ylim(-60, 60)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 08c_collar_strategy/chart.pdf")
