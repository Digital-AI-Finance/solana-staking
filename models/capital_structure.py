"""
Capital Structure Optimization Model.

Implements Modigliani-Miller framework extended for crypto treasury.

Key concepts:
- Optimal leverage for SOL acquisition
- WACC minimization
- Risk-adjusted returns
- Distress cost modeling
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy import optimize
from utils import (
    RISK_FREE_RATE, SOL_VOLATILITY, CORPORATE_TAX_RATE,
    CURRENT_SOL_PRICE, CURRENT_STAKING_APY
)


@dataclass
class CapitalStructureParams:
    """Parameters for capital structure optimization."""

    # Initial capital
    equity_capital: float = 50_000_000  # $50M equity

    # Cost of capital
    cost_of_equity_base: float = 0.20  # 20% base cost of equity
    cost_of_debt_base: float = 0.06  # 6% base cost of debt
    risk_free_rate: float = RISK_FREE_RATE

    # Tax and distress
    tax_rate: float = CORPORATE_TAX_RATE
    distress_cost_factor: float = 0.25  # 25% of firm value lost in distress

    # SOL parameters
    sol_price: float = CURRENT_SOL_PRICE
    sol_expected_return: float = 0.15  # 15% expected SOL return
    sol_volatility: float = SOL_VOLATILITY
    staking_yield: float = CURRENT_STAKING_APY

    # Leverage constraints
    max_debt_to_equity: float = 3.0  # Max 3:1 D/E ratio
    min_interest_coverage: float = 1.5  # Min interest coverage ratio


class CapitalStructureOptimizer:
    """
    Optimize capital structure for SOL accumulation strategy.

    Model Framework:
        V_L = V_U + PV(Tax Shield) - PV(Distress Costs)

    Where:
        V_U = Unlevered firm value
        PV(Tax Shield) = tau_c * D (perpetuity assumption)
        PV(Distress Costs) = p(default) * alpha * V_U

    Cost of Capital:
        WACC = (E/V) * r_e + (D/V) * r_d * (1 - tau)
    """

    def __init__(self, params: CapitalStructureParams):
        self.params = params

    def cost_of_equity(self, debt_to_equity: float) -> float:
        """
        Calculate cost of equity using Modigliani-Miller Proposition II.

        r_e = r_u + (r_u - r_d) * (D/E) * (1 - tau)

        As leverage increases, cost of equity increases.

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            Cost of equity
        """
        p = self.params

        # Unlevered cost of equity (approximate as base cost)
        r_u = p.cost_of_equity_base

        # Levered cost of equity
        r_e = r_u + (r_u - p.cost_of_debt_base) * debt_to_equity * (1 - p.tax_rate)

        return r_e

    def cost_of_debt(self, debt_to_equity: float) -> float:
        """
        Calculate cost of debt with credit spread adjustment.

        Cost increases with leverage due to higher default risk.

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            Cost of debt
        """
        p = self.params

        # Credit spread increases with leverage
        # Simple model: spread = base_spread + k * (D/E)^2
        base_spread = p.cost_of_debt_base - p.risk_free_rate
        leverage_spread = 0.02 * debt_to_equity ** 2  # Quadratic increase

        return p.risk_free_rate + base_spread + leverage_spread

    def wacc(self, debt_to_equity: float) -> float:
        """
        Calculate Weighted Average Cost of Capital.

        WACC = (E/V) * r_e + (D/V) * r_d * (1 - tau)

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            WACC
        """
        p = self.params

        if debt_to_equity <= 0:
            return self.cost_of_equity(0)

        # Weight calculations
        d_v = debt_to_equity / (1 + debt_to_equity)  # D/V
        e_v = 1 / (1 + debt_to_equity)  # E/V

        r_e = self.cost_of_equity(debt_to_equity)
        r_d = self.cost_of_debt(debt_to_equity)

        return e_v * r_e + d_v * r_d * (1 - p.tax_rate)

    def probability_of_distress(self, debt_to_equity: float) -> float:
        """
        Estimate probability of financial distress.

        Simple model based on distance to default (Merton-style).

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            Probability of distress (0-1)
        """
        p = self.params

        if debt_to_equity <= 0:
            return 0.0

        # Distance to default (simplified)
        # Higher leverage -> higher distress probability
        # Using logistic function for smooth transition
        dd = 3.0 / (1 + debt_to_equity) - 1.5  # Distance to default proxy

        # Convert to probability via normal CDF
        from scipy import stats
        prob = stats.norm.cdf(-dd)

        return min(prob, 0.5)  # Cap at 50%

    def firm_value(self, debt_to_equity: float) -> Dict[str, float]:
        """
        Calculate levered firm value.

        V_L = V_U + Tax_Shield - Distress_Costs

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            Dictionary with value components
        """
        p = self.params

        # Total capital
        debt = p.equity_capital * debt_to_equity
        total_capital = p.equity_capital + debt

        # Unlevered value (equity capital invested at expected return)
        # Simplified: V_U = E * (1 + expected_return)
        v_unlevered = p.equity_capital

        # Tax shield value
        # Perpetuity: PV = tau * r_d * D / r_d = tau * D
        tax_shield = p.tax_rate * debt

        # Distress costs
        prob_distress = self.probability_of_distress(debt_to_equity)
        distress_cost = prob_distress * p.distress_cost_factor * (v_unlevered + tax_shield)

        # Levered value
        v_levered = v_unlevered + tax_shield - distress_cost

        return {
            'debt': debt,
            'equity': p.equity_capital,
            'total_capital': total_capital,
            'v_unlevered': v_unlevered,
            'tax_shield': tax_shield,
            'prob_distress': prob_distress,
            'distress_cost': distress_cost,
            'v_levered': v_levered,
            'value_added': v_levered - v_unlevered,
        }

    def sol_acquisition_metrics(self, debt_to_equity: float) -> Dict[str, float]:
        """
        Calculate SOL acquisition metrics for given leverage.

        Parameters:
            debt_to_equity: D/E ratio

        Returns:
            Dictionary with SOL metrics
        """
        p = self.params

        debt = p.equity_capital * debt_to_equity
        total_capital = p.equity_capital + debt

        # SOL acquirable
        sol_purchased = total_capital / p.sol_price

        # Interest expense
        r_d = self.cost_of_debt(debt_to_equity)
        annual_interest = debt * r_d

        # Staking income
        staking_income_sol = sol_purchased * p.staking_yield
        staking_income_usd = staking_income_sol * p.sol_price

        # Interest coverage ratio
        interest_coverage = staking_income_usd / annual_interest if annual_interest > 0 else np.inf

        # Net income (simplified)
        net_income = staking_income_usd - annual_interest * (1 - p.tax_rate)

        # ROE
        roe = net_income / p.equity_capital if p.equity_capital > 0 else 0

        return {
            'debt': debt,
            'total_capital': total_capital,
            'sol_purchased': sol_purchased,
            'sol_per_equity_dollar': sol_purchased / p.equity_capital,
            'annual_interest': annual_interest,
            'staking_income_sol': staking_income_sol,
            'staking_income_usd': staking_income_usd,
            'interest_coverage': interest_coverage,
            'net_income': net_income,
            'roe': roe,
        }

    def optimal_leverage(self) -> Tuple[float, Dict]:
        """
        Find optimal debt-to-equity ratio.

        Objective: Maximize firm value subject to constraints.

        Returns:
            Tuple of (optimal_d_e_ratio, metrics)
        """
        p = self.params

        def objective(d_e):
            # Negative because we minimize
            return -self.firm_value(d_e[0])['v_levered']

        def interest_coverage_constraint(d_e):
            metrics = self.sol_acquisition_metrics(d_e[0])
            return metrics['interest_coverage'] - p.min_interest_coverage

        constraints = [
            {'type': 'ineq', 'fun': interest_coverage_constraint},
        ]

        bounds = [(0, p.max_debt_to_equity)]

        result = optimize.minimize(
            objective,
            x0=[1.0],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_de = result.x[0]

        return optimal_de, {
            'optimal_d_e': optimal_de,
            'firm_value': self.firm_value(optimal_de),
            'sol_metrics': self.sol_acquisition_metrics(optimal_de),
            'wacc': self.wacc(optimal_de),
            'cost_of_equity': self.cost_of_equity(optimal_de),
            'cost_of_debt': self.cost_of_debt(optimal_de),
        }

    def leverage_analysis(
        self,
        d_e_range: Tuple[float, float] = (0, 3.0),
        n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Analyze metrics across leverage range.

        Returns:
            Dictionary with arrays for each metric
        """
        d_e_ratios = np.linspace(d_e_range[0], d_e_range[1], n_points)

        results = {
            'd_e_ratio': d_e_ratios,
            'wacc': [],
            'cost_of_equity': [],
            'cost_of_debt': [],
            'v_levered': [],
            'tax_shield': [],
            'distress_cost': [],
            'sol_purchased': [],
            'interest_coverage': [],
            'roe': [],
        }

        for d_e in d_e_ratios:
            results['wacc'].append(self.wacc(d_e))
            results['cost_of_equity'].append(self.cost_of_equity(d_e))
            results['cost_of_debt'].append(self.cost_of_debt(d_e))

            fv = self.firm_value(d_e)
            results['v_levered'].append(fv['v_levered'])
            results['tax_shield'].append(fv['tax_shield'])
            results['distress_cost'].append(fv['distress_cost'])

            sm = self.sol_acquisition_metrics(d_e)
            results['sol_purchased'].append(sm['sol_purchased'])
            results['interest_coverage'].append(min(sm['interest_coverage'], 20))  # Cap for plotting
            results['roe'].append(sm['roe'])

        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])

        return results

    def scenario_analysis(
        self,
        sol_prices: List[float] = [75, 100, 150, 200, 300],
        d_e_ratio: float = 1.5
    ) -> List[Dict]:
        """
        Analyze outcomes under different SOL price scenarios.

        Parameters:
            sol_prices: List of SOL prices to analyze
            d_e_ratio: Debt-to-equity ratio to use

        Returns:
            List of dictionaries with scenario outcomes
        """
        results = []

        for price in sol_prices:
            # Update params with new price
            scenario_params = CapitalStructureParams(
                equity_capital=self.params.equity_capital,
                cost_of_equity_base=self.params.cost_of_equity_base,
                cost_of_debt_base=self.params.cost_of_debt_base,
                tax_rate=self.params.tax_rate,
                sol_price=price,
                sol_expected_return=self.params.sol_expected_return,
                sol_volatility=self.params.sol_volatility,
                staking_yield=self.params.staking_yield,
            )

            scenario_optimizer = CapitalStructureOptimizer(scenario_params)
            metrics = scenario_optimizer.sol_acquisition_metrics(d_e_ratio)

            # Portfolio value calculation
            portfolio_value = metrics['sol_purchased'] * price

            # Return on equity capital
            return_on_equity = (portfolio_value - self.params.equity_capital) / self.params.equity_capital

            results.append({
                'sol_price': price,
                'sol_purchased': metrics['sol_purchased'],
                'portfolio_value': portfolio_value,
                'return_on_equity': return_on_equity,
                'interest_coverage': metrics['interest_coverage'],
            })

        return results


def compare_funding_strategies(
    equity_capital: float,
    strategies: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Compare different funding strategies for SOL acquisition.

    Parameters:
        equity_capital: Base equity capital
        strategies: Dict mapping strategy name to D/E ratio

    Returns:
        Comparison results
    """
    results = {}

    for name, d_e in strategies.items():
        params = CapitalStructureParams(equity_capital=equity_capital)
        optimizer = CapitalStructureOptimizer(params)

        metrics = optimizer.sol_acquisition_metrics(d_e)
        firm_val = optimizer.firm_value(d_e)

        results[name] = {
            'd_e_ratio': d_e,
            'total_capital': metrics['total_capital'],
            'sol_purchased': metrics['sol_purchased'],
            'wacc': optimizer.wacc(d_e),
            'interest_coverage': metrics['interest_coverage'],
            'roe': metrics['roe'],
            'prob_distress': firm_val['prob_distress'],
        }

    return results


if __name__ == "__main__":
    print("=== Capital Structure Optimization ===\n")

    params = CapitalStructureParams(
        equity_capital=50_000_000,  # $50M
        cost_of_equity_base=0.20,
        cost_of_debt_base=0.06,
        tax_rate=0.21,
        sol_price=150,
        staking_yield=0.075,
    )

    optimizer = CapitalStructureOptimizer(params)

    print("--- Cost of Capital at Various Leverage Levels ---")
    for d_e in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"D/E = {d_e:.1f}: WACC = {optimizer.wacc(d_e):.2%}, "
              f"r_e = {optimizer.cost_of_equity(d_e):.2%}, "
              f"r_d = {optimizer.cost_of_debt(d_e):.2%}")

    print("\n--- Optimal Leverage ---")
    optimal_de, metrics = optimizer.optimal_leverage()
    print(f"Optimal D/E Ratio: {optimal_de:.2f}")
    print(f"WACC at Optimal: {metrics['wacc']:.2%}")
    print(f"SOL Purchased: {metrics['sol_metrics']['sol_purchased']:,.0f}")
    print(f"Interest Coverage: {metrics['sol_metrics']['interest_coverage']:.2f}x")
    print(f"ROE: {metrics['sol_metrics']['roe']:.2%}")

    print("\n--- Funding Strategy Comparison ---")
    strategies = {
        'All Equity': 0.0,
        'Conservative (0.5x)': 0.5,
        'Moderate (1.0x)': 1.0,
        'Aggressive (2.0x)': 2.0,
        'Maximum (3.0x)': 3.0,
    }

    comparison = compare_funding_strategies(50_000_000, strategies)
    for name, data in comparison.items():
        print(f"\n{name}:")
        print(f"  Total Capital: ${data['total_capital']:,.0f}")
        print(f"  SOL Purchased: {data['sol_purchased']:,.0f}")
        print(f"  WACC: {data['wacc']:.2%}")
        print(f"  Interest Coverage: {data['interest_coverage']:.2f}x")
        print(f"  P(Distress): {data['prob_distress']:.2%}")
