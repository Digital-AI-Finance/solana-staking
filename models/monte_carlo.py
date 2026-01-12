"""
Monte Carlo Simulation Engine for SOL Price Paths.

Implements Geometric Brownian Motion with Jumps (Merton Model).
"""
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import stats
from utils import (
    RISK_FREE_RATE, SOL_VOLATILITY, CURRENT_STAKING_APY,
    value_at_risk, conditional_var, sharpe_ratio, max_drawdown
)


class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion Model for SOL price simulation.

    dS_t = (mu - lambda*kappa) * S_t * dt + sigma * S_t * dW_t + S_t * dJ_t

    Where:
        - mu: drift rate (includes staking yield)
        - sigma: volatility
        - lambda_j: jump intensity (avg jumps per year)
        - mu_j: mean jump size (log)
        - sigma_j: jump size volatility (log)
        - W_t: standard Brownian motion
        - J_t: compound Poisson process
    """

    def __init__(
        self,
        S0: float = 150.0,
        mu: float = 0.15,
        sigma: float = SOL_VOLATILITY,
        lambda_j: float = 2.0,
        mu_j: float = -0.15,
        sigma_j: float = 0.10,
        staking_yield: float = CURRENT_STAKING_APY,
        seed: Optional[int] = None
    ):
        """
        Initialize Merton Jump-Diffusion model.

        Parameters:
            S0: Initial price
            mu: Annual drift rate
            sigma: Annual volatility
            lambda_j: Jump intensity (jumps per year)
            mu_j: Mean log jump size
            sigma_j: Std dev of log jump size
            staking_yield: Annual staking yield (added to returns)
            seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.staking_yield = staking_yield

        if seed is not None:
            np.random.seed(seed)

        # Expected jump size compensation
        self.kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def simulate_paths(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 10000
    ) -> np.ndarray:
        """
        Simulate price paths using exact discretization.

        S_{t+dt} = S_t * exp[(mu - 0.5*sigma^2 - lambda*kappa)*dt
                             + sigma*sqrt(dt)*Z + sum(J)]

        Parameters:
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths

        Returns:
            Array of shape (n_paths, n_steps + 1) with price paths
        """
        dt = T / n_steps

        # Drift term (adjusted for jumps)
        drift = (self.mu - 0.5 * self.sigma**2 - self.lambda_j * self.kappa) * dt

        # Diffusion term
        diffusion = self.sigma * np.sqrt(dt)

        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        for t in range(1, n_steps + 1):
            # Brownian motion
            Z = np.random.standard_normal(n_paths)

            # Jump process: number of jumps in dt
            n_jumps = np.random.poisson(self.lambda_j * dt, n_paths)

            # Jump sizes (sum of log-normal jumps)
            jump_sum = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps = np.random.normal(self.mu_j, self.sigma_j, n_jumps[i])
                    jump_sum[i] = np.sum(jumps)

            # Update prices
            log_return = drift + diffusion * Z + jump_sum
            paths[:, t] = paths[:, t-1] * np.exp(log_return)

        return paths

    def simulate_with_staking(
        self,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 10000,
        compound_frequency: int = 365
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price paths with staking rewards compounded.

        Parameters:
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            compound_frequency: How often staking rewards compound per year

        Returns:
            Tuple of (price_paths, staked_sol_paths)
            - price_paths: SOL price at each step
            - staked_sol_paths: Total SOL holdings (initial + staking rewards)
        """
        price_paths = self.simulate_paths(T, n_steps, n_paths)

        # Calculate staking rewards
        dt = T / n_steps
        staking_rate_per_step = self.staking_yield * dt

        staked_paths = np.ones((n_paths, n_steps + 1))
        for t in range(1, n_steps + 1):
            # Compound staking rewards
            staked_paths[:, t] = staked_paths[:, t-1] * (1 + staking_rate_per_step)

        return price_paths, staked_paths


class AccumulationSimulator:
    """
    Simulate different SOL accumulation strategies.
    """

    def __init__(
        self,
        initial_capital: float,
        model: MertonJumpDiffusion,
        seed: Optional[int] = None
    ):
        """
        Initialize accumulation simulator.

        Parameters:
            initial_capital: Starting capital in USD
            model: Price simulation model
            seed: Random seed
        """
        self.initial_capital = initial_capital
        self.model = model
        if seed is not None:
            np.random.seed(seed)

    def lump_sum(
        self,
        T: float = 3.0,
        n_steps: int = 756,
        n_paths: int = 10000
    ) -> Dict:
        """
        Simulate lump sum investment strategy.

        All capital deployed at time 0.

        Returns:
            Dictionary with simulation results
        """
        price_paths, staked_paths = self.model.simulate_with_staking(T, n_steps, n_paths)

        # Buy all SOL at t=0
        sol_purchased = self.initial_capital / self.model.S0

        # Final value = SOL * staking_multiplier * final_price
        final_sol = sol_purchased * staked_paths[:, -1]
        final_value = final_sol * price_paths[:, -1]

        return self._compute_metrics(final_value, final_sol, price_paths, "Lump Sum")

    def dca(
        self,
        T: float = 3.0,
        n_steps: int = 756,
        n_paths: int = 10000,
        n_purchases: int = 36
    ) -> Dict:
        """
        Simulate Dollar Cost Averaging strategy.

        Capital deployed evenly over n_purchases periods.

        Parameters:
            n_purchases: Number of equal purchases (e.g., 36 for monthly over 3 years)

        Returns:
            Dictionary with simulation results
        """
        price_paths, staked_paths = self.model.simulate_with_staking(T, n_steps, n_paths)

        amount_per_purchase = self.initial_capital / n_purchases
        purchase_interval = n_steps // n_purchases

        # Track SOL accumulated over time
        sol_holdings = np.zeros(n_paths)
        staking_multiplier = np.ones(n_paths)

        for i in range(n_purchases):
            purchase_step = i * purchase_interval
            price_at_purchase = price_paths[:, purchase_step]
            sol_bought = amount_per_purchase / price_at_purchase

            # Apply staking from purchase time to end
            remaining_steps = n_steps - purchase_step
            staking_factor = staked_paths[:, n_steps] / staked_paths[:, purchase_step]

            sol_holdings += sol_bought * staking_factor

        final_value = sol_holdings * price_paths[:, -1]

        return self._compute_metrics(final_value, sol_holdings, price_paths, "DCA")

    def value_averaging(
        self,
        T: float = 3.0,
        n_steps: int = 756,
        n_paths: int = 10000,
        n_periods: int = 36,
        target_growth_rate: float = 0.05
    ) -> Dict:
        """
        Simulate Value Averaging strategy.

        Adjust purchases to maintain target portfolio growth.

        Parameters:
            n_periods: Number of periods
            target_growth_rate: Target growth rate per period

        Returns:
            Dictionary with simulation results
        """
        price_paths, staked_paths = self.model.simulate_with_staking(T, n_steps, n_paths)
        purchase_interval = n_steps // n_periods

        # Initial target value
        target_value = np.full(n_paths, self.initial_capital / n_periods)
        sol_holdings = np.zeros(n_paths)
        total_invested = np.zeros(n_paths)

        for i in range(n_periods):
            step = i * purchase_interval
            current_price = price_paths[:, step]

            # Current portfolio value
            current_value = sol_holdings * current_price

            # Target value increases each period
            period_target = (self.initial_capital / n_periods) * (1 + target_growth_rate) ** i

            # Amount needed to reach target
            amount_needed = period_target - current_value
            amount_needed = np.clip(amount_needed, 0, self.initial_capital / n_periods * 2)

            # Buy SOL
            sol_bought = amount_needed / current_price
            total_invested += amount_needed

            # Apply staking factor to newly bought SOL (from purchase time to end)
            staking_factor = staked_paths[:, n_steps] / staked_paths[:, step]
            sol_holdings += sol_bought * staking_factor

        final_value = sol_holdings * price_paths[:, -1]

        return self._compute_metrics(final_value, sol_holdings, price_paths, "Value Averaging")

    def _compute_metrics(
        self,
        final_values: np.ndarray,
        final_sol: np.ndarray,
        price_paths: np.ndarray,
        strategy_name: str
    ) -> Dict:
        """
        Compute performance metrics for simulation results.
        """
        returns = (final_values - self.initial_capital) / self.initial_capital

        return {
            'strategy': strategy_name,
            'initial_capital': self.initial_capital,
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'var_95': value_at_risk(returns, 0.05),
            'cvar_95': conditional_var(returns, 0.05),
            'prob_2x': np.mean(final_values >= 2 * self.initial_capital),
            'prob_5x': np.mean(final_values >= 5 * self.initial_capital),
            'prob_10x': np.mean(final_values >= 10 * self.initial_capital),
            'prob_loss': np.mean(final_values < self.initial_capital),
            'mean_sol_accumulated': np.mean(final_sol),
            'median_sol_accumulated': np.median(final_sol),
            'final_values': final_values,
            'final_sol': final_sol,
        }


def compute_percentile_paths(
    paths: np.ndarray,
    percentiles: list = [5, 25, 50, 75, 95]
) -> Dict[int, np.ndarray]:
    """
    Compute percentile paths for fan chart visualization.

    Parameters:
        paths: Array of shape (n_paths, n_steps)
        percentiles: List of percentiles to compute

    Returns:
        Dictionary mapping percentile to path
    """
    return {p: np.percentile(paths, p, axis=0) for p in percentiles}


if __name__ == "__main__":
    # Test simulations
    print("=== Monte Carlo Simulation Test ===")

    model = MertonJumpDiffusion(
        S0=150,
        mu=0.15,
        sigma=0.80,
        lambda_j=2.0,
        mu_j=-0.15,
        sigma_j=0.10,
        staking_yield=0.075,
        seed=42
    )

    print("\n--- Price Simulation ---")
    paths = model.simulate_paths(T=3.0, n_steps=756, n_paths=10000)
    print(f"Initial price: ${model.S0}")
    print(f"Mean final price: ${np.mean(paths[:, -1]):.2f}")
    print(f"Median final price: ${np.median(paths[:, -1]):.2f}")
    print(f"5th percentile: ${np.percentile(paths[:, -1], 5):.2f}")
    print(f"95th percentile: ${np.percentile(paths[:, -1], 95):.2f}")

    print("\n--- Accumulation Strategies ---")
    sim = AccumulationSimulator(initial_capital=100000, model=model, seed=42)

    for strategy in [sim.lump_sum, sim.dca, sim.value_averaging]:
        results = strategy(T=3.0, n_steps=756, n_paths=10000)
        print(f"\n{results['strategy']}:")
        print(f"  Mean final value: ${results['mean_final_value']:,.0f}")
        print(f"  Median final value: ${results['median_final_value']:,.0f}")
        print(f"  VaR (95%): {results['var_95']:.2%}")
        print(f"  P(2x): {results['prob_2x']:.1%}")
        print(f"  P(loss): {results['prob_loss']:.1%}")
