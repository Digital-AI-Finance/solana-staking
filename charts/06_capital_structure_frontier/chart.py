"""
Capital Structure Optimization - Efficient Frontier
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# D/E ratios to analyze
d_e_ratios = np.linspace(0, 3.0, 100)

# Cost of capital functions
def cost_of_equity(d_e, r_u=0.20, r_d=0.06, tax=0.21):
    """MM Proposition II: r_e = r_u + (r_u - r_d) * (D/E) * (1 - tau)"""
    return r_u + (r_u - r_d) * d_e * (1 - tax)

def cost_of_debt(d_e, r_f=0.045, base_spread=0.015):
    """Cost of debt increases with leverage"""
    return r_f + base_spread + 0.02 * d_e ** 2

def wacc(d_e, tax=0.21):
    """Weighted Average Cost of Capital"""
    r_e = cost_of_equity(d_e)
    r_d = cost_of_debt(d_e)
    d_v = d_e / (1 + d_e)
    e_v = 1 / (1 + d_e)
    return e_v * r_e + d_v * r_d * (1 - tax)

# Calculate metrics
r_e_values = [cost_of_equity(de) for de in d_e_ratios]
r_d_values = [cost_of_debt(de) for de in d_e_ratios]
wacc_values = [wacc(de) for de in d_e_ratios]

# Find optimal D/E (minimum WACC)
optimal_idx = np.argmin(wacc_values)
optimal_de = d_e_ratios[optimal_idx]
optimal_wacc = wacc_values[optimal_idx]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot cost curves
ax.plot(d_e_ratios, np.array(r_e_values) * 100, color=COLORS['SOL_PURPLE'],
        linewidth=2.5, label='Cost of Equity ($r_e$)')
ax.plot(d_e_ratios, np.array(r_d_values) * 100, color=COLORS['SOL_GREEN'],
        linewidth=2.5, label='Cost of Debt ($r_d$)')
ax.plot(d_e_ratios, np.array(wacc_values) * 100, color=COLORS['ACCENT_BLUE'],
        linewidth=3, label='WACC')

# Mark optimal point
ax.scatter([optimal_de], [optimal_wacc * 100], color=COLORS['ERROR_RED'],
           s=150, zorder=5, edgecolor='white', linewidth=2)
ax.annotate(f'Optimal D/E = {optimal_de:.2f}\nWACC = {optimal_wacc:.1%}',
            xy=(optimal_de, optimal_wacc * 100),
            xytext=(optimal_de + 0.5, optimal_wacc * 100 - 3),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['ERROR_RED'], lw=2))

# Tax shield region
ax.fill_between(d_e_ratios[:optimal_idx+1],
                [cost_of_equity(0)] * (optimal_idx+1) * np.array([100]),
                np.array(wacc_values[:optimal_idx+1]) * 100,
                alpha=0.15, color=COLORS['SOL_GREEN'],
                label='Tax Shield Benefit')

# Distress cost region
ax.fill_between(d_e_ratios[optimal_idx:],
                np.array(wacc_values[optimal_idx:]) * 100,
                np.array([wacc_values[optimal_idx]] * len(d_e_ratios[optimal_idx:])) * 100,
                alpha=0.15, color=COLORS['ERROR_RED'],
                label='Distress Cost Premium')

# Reference lines
ax.axhline(y=cost_of_equity(0) * 100, color=COLORS['TEXT_SECONDARY'],
           linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.8, cost_of_equity(0) * 100 + 0.5, 'All-Equity ($r_u$)',
        fontsize=10, color=COLORS['TEXT_SECONDARY'])

ax.set_xlabel('Debt-to-Equity Ratio (D/E)', fontsize=14)
ax.set_ylabel('Cost of Capital (%)', fontsize=14)
ax.set_title('Capital Structure Optimization: WACC vs Leverage', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(0, 3.0)
ax.set_ylim(5, 35)
ax.grid(True, alpha=0.3)

# Add interpretation text
ax.text(0.05, 0.95, 'Modigliani-Miller with Taxes & Distress Costs',
        transform=ax.transAxes, fontsize=10, style='italic',
        verticalalignment='top', color=COLORS['TEXT_SECONDARY'])

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 06_capital_structure_frontier/chart.pdf")
