"""
IRR Sensitivity Analysis - Returns Under Different Scenarios
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Scenario parameters
sol_price_changes = np.array([-50, -30, -10, 0, 10, 30, 50, 100])  # % change
staking_yields = np.array([5, 7.5, 10, 12.5])  # % APY
leverage_ratios = np.array([0, 1.0, 1.5, 2.0])  # D/E ratio

# Base case assumptions
base_sol_price = 150
initial_equity = 50_000_000
holding_period = 3  # years
cost_of_debt = 0.06

def calculate_irr(sol_change, staking_yield, leverage):
    """Calculate IRR for given parameters"""
    # Capital deployed
    debt = initial_equity * leverage
    total_capital = initial_equity + debt

    # SOL purchased
    sol_purchased = total_capital / base_sol_price

    # Annual cash flows
    annual_staking_income = sol_purchased * staking_yield / 100 * base_sol_price
    annual_interest = debt * cost_of_debt
    annual_net_cf = annual_staking_income - annual_interest

    # Final value
    final_sol_price = base_sol_price * (1 + sol_change / 100)
    final_portfolio_value = sol_purchased * (1 + staking_yield/100) ** holding_period * final_sol_price
    final_equity_value = final_portfolio_value - debt

    # Simple IRR approximation (using CAGR)
    total_return = (final_equity_value / initial_equity) ** (1/holding_period) - 1

    return total_return * 100  # Return as percentage

# Calculate IRR matrix
irr_matrix = np.zeros((len(staking_yields), len(sol_price_changes)))

for i, yield_rate in enumerate(staking_yields):
    for j, price_change in enumerate(sol_price_changes):
        irr_matrix[i, j] = calculate_irr(price_change, yield_rate, leverage=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: Heatmap of IRR by price change and staking yield
im = ax1.imshow(irr_matrix, cmap='RdYlGn', aspect='auto',
                vmin=-50, vmax=100)

# Add text annotations
for i in range(len(staking_yields)):
    for j in range(len(sol_price_changes)):
        val = irr_matrix[i, j]
        color = 'white' if abs(val) > 40 else 'black'
        ax1.text(j, i, f'{val:.0f}%', ha='center', va='center',
                 fontsize=10, fontweight='bold', color=color)

ax1.set_xticks(range(len(sol_price_changes)))
ax1.set_xticklabels([f'{x:+d}%' for x in sol_price_changes], fontsize=10)
ax1.set_yticks(range(len(staking_yields)))
ax1.set_yticklabels([f'{y}%' for y in staking_yields], fontsize=10)
ax1.set_xlabel('SOL Price Change', fontsize=13)
ax1.set_ylabel('Staking APY', fontsize=13)
ax1.set_title('IRR by Scenario (1.5x Leverage)', fontsize=14, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Annual IRR (%)', fontsize=11)

# Right: IRR by leverage level (for base case price)
price_scenarios = [-20, 0, 30, 75]  # % price changes
colors_line = [COLORS['ERROR_RED'], COLORS['WARNING_ORANGE'],
               COLORS['SOL_GREEN'], COLORS['SOL_PURPLE']]

for price_change, color in zip(price_scenarios, colors_line):
    irrs = [calculate_irr(price_change, 7.5, lev) for lev in leverage_ratios]
    ax2.plot(leverage_ratios, irrs, 'o-', color=color, linewidth=2.5,
             markersize=8, label=f'SOL {price_change:+d}%')

ax2.axhline(y=0, color=COLORS['TEXT_SECONDARY'], linewidth=1.5, linestyle='--')
ax2.fill_between(leverage_ratios, 0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 100,
                 alpha=0.1, color=COLORS['SOL_GREEN'])
ax2.fill_between(leverage_ratios, ax2.get_ylim()[0] if ax2.get_ylim()[0] < 0 else -50, 0,
                 alpha=0.1, color=COLORS['ERROR_RED'])

ax2.set_xlabel('Debt-to-Equity Ratio', fontsize=13)
ax2.set_ylabel('Annual IRR (%)', fontsize=13)
ax2.set_title('IRR vs Leverage (7.5% APY)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(-60, 120)

# Add risk annotation
ax2.annotate('Higher leverage\namplifies returns\n(both ways)',
             xy=(1.8, 80), fontsize=10, ha='center',
             color=COLORS['TEXT_SECONDARY'])

plt.suptitle('SOL Treasury IRR Sensitivity Analysis (3-Year Horizon)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 12_irr_sensitivity/chart.pdf")
