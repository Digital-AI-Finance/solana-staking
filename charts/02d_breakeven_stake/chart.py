"""
Validator Breakeven Analysis - Minimum Stake for Profitability
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Parameters
commission_rates = np.array([0.02, 0.05, 0.08, 0.10])  # 2%, 5%, 8%, 10%
base_apy = 0.075  # 7.5% base staking APY
sol_prices = np.array([100, 150, 200, 250])  # Different SOL prices

# Annual costs
annual_vote_cost_sol = 400  # ~1.1 SOL/day
monthly_infra_usd = 700  # Hardware, colo, bandwidth, monitoring
annual_infra_usd = monthly_infra_usd * 12

def breakeven_stake(commission, sol_price):
    """Calculate breakeven stake for given commission and SOL price"""
    # Revenue per SOL of stake = APY * commission
    revenue_per_sol = base_apy * commission

    # Total costs in SOL
    infra_cost_sol = annual_infra_usd / sol_price
    total_cost_sol = annual_vote_cost_sol + infra_cost_sol

    # Breakeven: revenue_per_sol * stake = total_cost_sol
    return total_cost_sol / revenue_per_sol

# Calculate breakeven for all combinations
breakeven_data = {}
for price in sol_prices:
    breakeven_data[price] = [breakeven_stake(c, price) for c in commission_rates]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: Breakeven stake by commission rate (for SOL=$150)
stake_levels = np.linspace(50000, 1000000, 100)
profit_by_commission = {}

for commission in commission_rates:
    annual_revenue = stake_levels * base_apy * commission
    total_cost = annual_vote_cost_sol + (annual_infra_usd / 150)
    profit = annual_revenue - total_cost
    profit_by_commission[commission] = profit

colors = [COLORS['SOL_PURPLE'], COLORS['SOL_GREEN'],
          COLORS['ACCENT_BLUE'], COLORS['WARNING_ORANGE']]

for (commission, profit), color in zip(profit_by_commission.items(), colors):
    ax1.plot(stake_levels / 1000, profit, color=color, linewidth=2.5,
             label=f'{commission:.0%} Commission')
    # Mark breakeven
    be_stake = breakeven_stake(commission, 150)
    ax1.scatter([be_stake/1000], [0], color=color, s=80, zorder=5, marker='x')

ax1.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1.5, linestyle='--')
ax1.fill_between(stake_levels/1000, 0, ax1.get_ylim()[1],
                 alpha=0.1, color=COLORS['SOL_GREEN'])
ax1.fill_between(stake_levels/1000, ax1.get_ylim()[0], 0,
                 alpha=0.1, color=COLORS['ERROR_RED'])

ax1.set_xlabel('Total Stake (Thousand SOL)', fontsize=13)
ax1.set_ylabel('Annual Profit (SOL)', fontsize=13)
ax1.set_title(f'Profitability vs Stake (SOL=${150})', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(50, 1000)
ax1.set_ylim(-500, 3000)
ax1.grid(True, alpha=0.3)

# Right: Breakeven stake heatmap by SOL price and commission
breakeven_matrix = np.array([[breakeven_stake(c, p) for c in commission_rates]
                              for p in sol_prices]) / 1000  # Convert to thousands

im = ax2.imshow(breakeven_matrix, cmap='RdYlGn_r', aspect='auto',
                vmin=50, vmax=800)

# Add text annotations
for i in range(len(sol_prices)):
    for j in range(len(commission_rates)):
        val = breakeven_matrix[i, j]
        color = 'white' if val > 400 else 'black'
        ax2.text(j, i, f'{val:.0f}K', ha='center', va='center',
                 fontsize=11, fontweight='bold', color=color)

ax2.set_xticks(range(len(commission_rates)))
ax2.set_xticklabels([f'{c:.0%}' for c in commission_rates], fontsize=11)
ax2.set_yticks(range(len(sol_prices)))
ax2.set_yticklabels([f'${p}' for p in sol_prices], fontsize=11)
ax2.set_xlabel('Commission Rate', fontsize=13)
ax2.set_ylabel('SOL Price', fontsize=13)
ax2.set_title('Breakeven Stake (Thousand SOL)', fontsize=14, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Breakeven Stake (K SOL)', fontsize=11)

plt.suptitle('Solana Validator Breakeven Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 02d_breakeven_stake/chart.pdf")
