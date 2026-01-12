"""
Monte Carlo Simulation Fan Chart - SOL Price Paths
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
S0 = 150  # Initial SOL price
mu = 0.15  # Annual drift (includes staking yield)
sigma = 0.80  # Annual volatility
T = 3.0  # 3 years
n_steps = 252 * 3  # Daily steps
n_paths = 5000

# Jump parameters (Merton model)
lambda_j = 2.0  # 2 jumps per year on average
mu_j = -0.15  # Mean jump size (negative = crashes)
sigma_j = 0.10  # Jump volatility

dt = T / n_steps
kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1  # Expected jump compensation

# Simulate paths
paths = np.zeros((n_paths, n_steps + 1))
paths[:, 0] = S0

for t in range(1, n_steps + 1):
    # Brownian motion
    Z = np.random.standard_normal(n_paths)

    # Jump process
    n_jumps = np.random.poisson(lambda_j * dt, n_paths)
    jump_sum = np.zeros(n_paths)
    for i in range(n_paths):
        if n_jumps[i] > 0:
            jumps = np.random.normal(mu_j, sigma_j, n_jumps[i])
            jump_sum[i] = np.sum(jumps)

    # Price update
    drift = (mu - 0.5 * sigma**2 - lambda_j * kappa) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jump_sum)

# Calculate percentiles
percentiles = [5, 10, 25, 50, 75, 90, 95]
percentile_paths = {p: np.percentile(paths, p, axis=0) for p in percentiles}

# Time axis (in years)
time_years = np.linspace(0, T, n_steps + 1)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot fan chart (confidence bands)
colors_fan = plt.cm.Purples(np.linspace(0.2, 0.8, 4))

# 90% band (5-95)
ax.fill_between(time_years, percentile_paths[5], percentile_paths[95],
                color=colors_fan[0], alpha=0.6, label='90% Confidence')

# 80% band (10-90)
ax.fill_between(time_years, percentile_paths[10], percentile_paths[90],
                color=colors_fan[1], alpha=0.6, label='80% Confidence')

# 50% band (25-75)
ax.fill_between(time_years, percentile_paths[25], percentile_paths[75],
                color=colors_fan[2], alpha=0.6, label='50% Confidence')

# Median line
ax.plot(time_years, percentile_paths[50], color=COLORS['SOL_PURPLE'],
        linewidth=3, label='Median Path')

# Starting point
ax.scatter([0], [S0], color=COLORS['SOL_GREEN'], s=100, zorder=5,
           edgecolor='white', linewidth=2)

# Add sample paths for visualization
for i in range(5):
    ax.plot(time_years, paths[i*1000], color=COLORS['TEXT_SECONDARY'],
            linewidth=0.5, alpha=0.3)

# Final distribution stats
final_prices = paths[:, -1]
mean_final = np.mean(final_prices)
median_final = np.median(final_prices)

# Annotations
ax.annotate(f'Median: ${median_final:.0f}',
            xy=(T, median_final), xytext=(T - 0.5, median_final + 150),
            fontsize=12, fontweight='bold', color=COLORS['SOL_PURPLE'],
            arrowprops=dict(arrowstyle='->', color=COLORS['SOL_PURPLE']))

ax.annotate(f'95th: ${percentile_paths[95][-1]:.0f}',
            xy=(T, percentile_paths[95][-1]),
            xytext=(T - 0.3, percentile_paths[95][-1] + 100),
            fontsize=10, color=COLORS['TEXT_SECONDARY'])

ax.annotate(f'5th: ${percentile_paths[5][-1]:.0f}',
            xy=(T, percentile_paths[5][-1]),
            xytext=(T - 0.3, percentile_paths[5][-1] - 50),
            fontsize=10, color=COLORS['TEXT_SECONDARY'])

# Add model parameters box
param_text = f'Parameters:\n$S_0$ = ${S0}\n$\\mu$ = {mu:.0%}\n$\\sigma$ = {sigma:.0%}\nT = {T:.0f} years\nn = {n_paths:,} paths'
ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Time (Years)', fontsize=14)
ax.set_ylabel('SOL Price ($)', fontsize=14)
ax.set_title('Monte Carlo Simulation: SOL Price Paths (GBM with Jumps)', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(0.15, 0.65), fontsize=11)
ax.set_xlim(0, T)
ax.set_ylim(0, 1000)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 07_monte_carlo_fan/chart.pdf")
