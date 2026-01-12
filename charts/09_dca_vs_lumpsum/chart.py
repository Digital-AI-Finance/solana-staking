"""
DCA vs Lump Sum vs Value Averaging - Accumulation Strategy Comparison
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

# Simulation parameters
np.random.seed(42)
S0 = 150
initial_capital = 100000  # $100K
T = 3.0  # 3 years
n_months = 36
n_simulations = 5000

# Simulate price paths (monthly)
mu = 0.15 / 12  # Monthly drift
sigma = 0.80 / np.sqrt(12)  # Monthly volatility

# Generate price paths
price_paths = np.zeros((n_simulations, n_months + 1))
price_paths[:, 0] = S0

for t in range(1, n_months + 1):
    Z = np.random.standard_normal(n_simulations)
    price_paths[:, t] = price_paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma * Z)

# Strategy 1: Lump Sum (buy all at t=0)
lump_sum_sol = initial_capital / S0
lump_sum_final_value = lump_sum_sol * price_paths[:, -1]

# Strategy 2: DCA (equal monthly purchases)
monthly_investment = initial_capital / n_months
dca_sol = np.zeros(n_simulations)
for t in range(n_months):
    dca_sol += monthly_investment / price_paths[:, t]
dca_final_value = dca_sol * price_paths[:, -1]

# Strategy 3: Value Averaging (buy more when down, less when up)
target_growth = (1 + 0.05/12)  # 5% annual target growth
va_sol = np.zeros(n_simulations)
va_invested = np.zeros(n_simulations)
base_investment = initial_capital / n_months

for t in range(n_months):
    target_value = base_investment * (t + 1) * target_growth ** t
    current_value = va_sol * price_paths[:, t]
    needed = np.clip(target_value - current_value, 0, base_investment * 2)
    va_sol += needed / price_paths[:, t]
    va_invested += needed

va_final_value = va_sol * price_paths[:, -1]

# Calculate statistics
strategies = {
    'Lump Sum': lump_sum_final_value,
    'DCA': dca_final_value,
    'Value Averaging': va_final_value,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# Left: Box plot comparison
data = [lump_sum_final_value / 1000, dca_final_value / 1000, va_final_value / 1000]
positions = [1, 2, 3]
colors_box = [COLORS['SOL_PURPLE'], COLORS['SOL_GREEN'], COLORS['ACCENT_BLUE']]

bp = ax1.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                  showfliers=False, medianprops=dict(color='white', linewidth=2))

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticklabels(['Lump Sum', 'DCA', 'Value\nAveraging'], fontsize=12)
ax1.set_ylabel('Final Portfolio Value ($K)', fontsize=13)
ax1.set_title('Distribution of Outcomes', fontsize=14, fontweight='bold')
ax1.axhline(y=100, color=COLORS['TEXT_SECONDARY'], linestyle='--',
            linewidth=1.5, label='Initial Capital')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add median labels
for i, d in enumerate(data):
    median = np.median(d)
    ax1.annotate(f'${median:.0f}K', xy=(positions[i], median),
                 xytext=(positions[i] + 0.3, median),
                 fontsize=10, fontweight='bold', color=colors_box[i])

# Right: Probability metrics
metrics = {
    'P(2x)': [np.mean(v >= 2*initial_capital) for v in strategies.values()],
    'P(Loss)': [np.mean(v < initial_capital) for v in strategies.values()],
    'Median Return': [np.median(v/initial_capital - 1) for v in strategies.values()],
}

x = np.arange(3)
width = 0.25

bars1 = ax2.bar(x - width, [m*100 for m in metrics['P(2x)']], width,
                label='P(2x) %', color=COLORS['SOL_GREEN'], alpha=0.8)
bars2 = ax2.bar(x, [m*100 for m in metrics['P(Loss)']], width,
                label='P(Loss) %', color=COLORS['ERROR_RED'], alpha=0.8)
bars3 = ax2.bar(x + width, [m*100 for m in metrics['Median Return']], width,
                label='Median Return %', color=COLORS['SOL_PURPLE'], alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(['Lump Sum', 'DCA', 'Value Avg'], fontsize=12)
ax2.set_ylabel('Percentage (%)', fontsize=13)
ax2.set_title('Strategy Metrics Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}%',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)

plt.suptitle('SOL Accumulation Strategies: $100K over 3 Years', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 09_dca_vs_lumpsum/chart.pdf")
