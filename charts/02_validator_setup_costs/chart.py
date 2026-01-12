"""
Solana Validator Setup and Operating Costs Breakdown
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Validator cost data (monthly USD)
cost_categories = {
    'Hardware/Server': 400,
    'Colocation': 300,
    'Bandwidth': 150,
    'Monitoring': 100,
    'Vote Costs (SOL)': 165,  # ~1.1 SOL/day * 30 * $5
}

# One-time setup costs
setup_costs = {
    'High-End Server': 8000,
    'Network Equipment': 1500,
    'Initial SOL Stake': 15000,  # ~100 SOL at $150
    'Setup & Config': 2000,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: Monthly operating costs (bar)
categories = list(cost_categories.keys())
costs = list(cost_categories.values())
total_monthly = sum(costs)

colors = [COLORS['SOL_PURPLE'], COLORS['ACCENT_BLUE'], COLORS['SOL_GREEN'],
          COLORS['WARNING_ORANGE'], COLORS['ERROR_RED']]

bars = ax1.barh(categories, costs, color=colors, edgecolor='white', linewidth=1.5)

# Add value labels
for bar, cost in zip(bars, costs):
    ax1.text(cost + 10, bar.get_y() + bar.get_height()/2,
             f'${cost:,}', va='center', fontsize=11)

ax1.set_xlabel('Monthly Cost (USD)', fontsize=13)
ax1.set_title('Monthly Operating Costs', fontsize=14, fontweight='bold')
ax1.set_xlim(0, max(costs) * 1.3)

# Add total annotation
ax1.text(0.95, 0.05, f'Total: ${total_monthly:,}/month\n(${total_monthly*12:,}/year)',
         transform=ax1.transAxes, fontsize=12, fontweight='bold',
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor=COLORS['SOL_PURPLE'], alpha=0.2))

# Right: One-time setup costs (stacked bar alternative: pie)
setup_labels = list(setup_costs.keys())
setup_values = list(setup_costs.values())
total_setup = sum(setup_values)

colors2 = [COLORS['ACCENT_BLUE'], COLORS['SOL_GREEN'],
           COLORS['WARNING_ORANGE'], COLORS['TEXT_SECONDARY']]

wedges, texts, autotexts = ax2.pie(
    setup_values,
    labels=None,
    autopct=lambda pct: f'${int(pct/100*total_setup):,}',
    colors=colors2,
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
)

for autotext in autotexts:
    autotext.set_fontsize(11)

# Legend
ax2.legend(wedges, setup_labels, loc='center left', bbox_to_anchor=(0.9, 0.5),
           fontsize=10)

ax2.set_title('Initial Setup Costs', fontsize=14, fontweight='bold')

# Center text
ax2.text(0, 0, f'${total_setup:,}\nTotal', ha='center', va='center',
         fontsize=13, fontweight='bold')

plt.suptitle('Solana Validator Cost Structure', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 02_validator_setup_costs/chart.pdf")
