"""
Solana Validator Economics Model.

Comprehensive model for validator profitability analysis including:
- Staking rewards
- MEV revenue (Jito)
- Vote costs
- Infrastructure costs
- Commission optimization
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from utils import (
    CURRENT_STAKING_APY, CURRENT_SOL_PRICE, DAILY_VOTE_COST_SOL,
    ANNUAL_VOTE_COST_SOL, JITO_VALIDATOR_SHARE, AVERAGE_MEV_TIPS_PER_SLOT,
    sol_to_usd, format_currency
)


@dataclass
class ValidatorConfig:
    """Configuration for a Solana validator."""

    # Stake parameters
    total_stake: float = 100_000  # Total SOL delegated
    self_stake: float = 10_000  # Validator's own stake
    commission_rate: float = 0.05  # 5% commission

    # Network parameters
    base_apy: float = CURRENT_STAKING_APY  # Base staking APY
    sol_price: float = CURRENT_SOL_PRICE  # SOL price in USD

    # MEV parameters
    jito_enabled: bool = True
    jito_validator_share: float = JITO_VALIDATOR_SHARE
    avg_mev_tips_per_slot: float = AVERAGE_MEV_TIPS_PER_SLOT

    # Infrastructure costs (monthly USD)
    hardware_cost: float = 300  # Server/hardware
    bandwidth_cost: float = 100  # Network
    colocation_cost: float = 200  # Data center
    maintenance_cost: float = 100  # Monitoring, updates

    # Vote costs
    daily_vote_cost_sol: float = DAILY_VOTE_COST_SOL

    @property
    def delegated_stake(self) -> float:
        """Stake from delegators (not self-stake)."""
        return self.total_stake - self.self_stake

    @property
    def monthly_infra_cost_usd(self) -> float:
        """Total monthly infrastructure cost in USD."""
        return (self.hardware_cost + self.bandwidth_cost +
                self.colocation_cost + self.maintenance_cost)

    @property
    def annual_infra_cost_usd(self) -> float:
        """Total annual infrastructure cost in USD."""
        return self.monthly_infra_cost_usd * 12


class ValidatorEconomics:
    """
    Calculate validator economics and profitability.

    Revenue Model:
        Revenue = Staking_Rewards + MEV_Tips - Vote_Costs - Infra_Costs

    Staking Rewards:
        - Validator earns commission on delegated stake rewards
        - Validator earns full rewards on self-stake
        - Total = self_stake * APY + delegated_stake * APY * commission

    MEV Revenue (Jito):
        - Validators running Jito earn tips from searchers
        - Tip share depends on stake weight and Jito configuration
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

    def annual_staking_revenue(self) -> Dict[str, float]:
        """
        Calculate annual staking revenue breakdown.

        Returns:
            Dictionary with revenue components in SOL
        """
        cfg = self.config

        # Self-stake rewards (validator keeps 100%)
        self_stake_rewards = cfg.self_stake * cfg.base_apy

        # Delegated stake rewards
        total_delegated_rewards = cfg.delegated_stake * cfg.base_apy

        # Commission from delegated rewards
        commission_revenue = total_delegated_rewards * cfg.commission_rate

        # Rewards passed to delegators
        delegator_rewards = total_delegated_rewards * (1 - cfg.commission_rate)

        return {
            'self_stake_rewards': self_stake_rewards,
            'total_delegated_rewards': total_delegated_rewards,
            'commission_revenue': commission_revenue,
            'delegator_rewards': delegator_rewards,
            'total_validator_staking_revenue': self_stake_rewards + commission_revenue,
        }

    def annual_mev_revenue(self) -> Dict[str, float]:
        """
        Calculate annual MEV revenue from Jito.

        Simplified model based on stake weight and average tips.

        Returns:
            Dictionary with MEV revenue components in SOL
        """
        cfg = self.config

        if not cfg.jito_enabled:
            return {
                'jito_enabled': False,
                'annual_mev_tips': 0,
                'validator_mev_share': 0,
            }

        # Slots per year (approximate)
        slots_per_year = 365 * 24 * 60 * 60 / 0.4  # ~400ms per slot

        # Total MEV tips on network (estimate)
        total_network_tips = slots_per_year * cfg.avg_mev_tips_per_slot

        # Stake weight (simplified - assumes uniform distribution)
        # In reality, depends on leader selection probability
        stake_weight = cfg.total_stake / 400_000_000  # vs total staked SOL

        # Validator's share of tips (when they're leader)
        validator_tips = total_network_tips * stake_weight * cfg.jito_validator_share

        return {
            'jito_enabled': True,
            'slots_per_year': slots_per_year,
            'total_network_tips_estimate': total_network_tips,
            'stake_weight': stake_weight,
            'annual_mev_tips': validator_tips,
        }

    def annual_costs(self) -> Dict[str, float]:
        """
        Calculate annual operating costs.

        Returns:
            Dictionary with cost components
        """
        cfg = self.config

        # Vote costs in SOL
        vote_costs_sol = cfg.daily_vote_cost_sol * 365

        # Infrastructure costs (convert to SOL)
        infra_costs_sol = cfg.annual_infra_cost_usd / cfg.sol_price

        return {
            'vote_costs_sol': vote_costs_sol,
            'infra_costs_usd': cfg.annual_infra_cost_usd,
            'infra_costs_sol': infra_costs_sol,
            'total_costs_sol': vote_costs_sol + infra_costs_sol,
            'total_costs_usd': (vote_costs_sol * cfg.sol_price) + cfg.annual_infra_cost_usd,
        }

    def annual_profit(self) -> Dict[str, float]:
        """
        Calculate annual profit/loss.

        Returns:
            Comprehensive P&L breakdown
        """
        cfg = self.config
        staking = self.annual_staking_revenue()
        mev = self.annual_mev_revenue()
        costs = self.annual_costs()

        total_revenue_sol = (
            staking['total_validator_staking_revenue'] +
            mev.get('annual_mev_tips', 0)
        )

        net_profit_sol = total_revenue_sol - costs['total_costs_sol']
        net_profit_usd = net_profit_sol * cfg.sol_price

        # ROI on self-stake
        roi = net_profit_sol / cfg.self_stake if cfg.self_stake > 0 else 0

        return {
            'staking_revenue_sol': staking['total_validator_staking_revenue'],
            'mev_revenue_sol': mev.get('annual_mev_tips', 0),
            'total_revenue_sol': total_revenue_sol,
            'total_revenue_usd': total_revenue_sol * cfg.sol_price,
            'total_costs_sol': costs['total_costs_sol'],
            'total_costs_usd': costs['total_costs_usd'],
            'net_profit_sol': net_profit_sol,
            'net_profit_usd': net_profit_usd,
            'roi_on_self_stake': roi,
            'monthly_profit_sol': net_profit_sol / 12,
            'monthly_profit_usd': net_profit_usd / 12,
        }

    def breakeven_stake(self) -> float:
        """
        Calculate minimum total stake needed to break even.

        Solve: Stake * APY * Commission + Self_Stake * APY + MEV = Costs

        Returns:
            Minimum total stake in SOL
        """
        cfg = self.config
        costs = self.annual_costs()

        # Revenue per SOL of total stake (simplified)
        # Assumes commission applies to all stake for simplicity
        revenue_per_sol = cfg.base_apy * cfg.commission_rate

        # Add MEV factor (rough estimate)
        if cfg.jito_enabled:
            mev_factor = 0.002  # ~0.2% additional yield from MEV
            revenue_per_sol += mev_factor

        # Breakeven: revenue_per_sol * stake = costs
        breakeven = costs['total_costs_sol'] / revenue_per_sol

        return max(0, breakeven)

    def optimal_commission_rate(
        self,
        min_rate: float = 0.0,
        max_rate: float = 0.10,
        n_points: int = 100
    ) -> Tuple[float, float]:
        """
        Find optimal commission rate that maximizes delegations
        while maintaining profitability.

        Simple model: Higher commission -> fewer delegations

        Delegation model: D(c) = D_max * exp(-k * c)

        Returns:
            Tuple of (optimal_rate, expected_profit)
        """
        cfg = self.config

        # Delegation sensitivity to commission
        k = 20  # Elasticity parameter

        # Maximum potential delegations at 0% commission
        d_max = cfg.total_stake * 2

        rates = np.linspace(min_rate, max_rate, n_points)
        profits = []

        for rate in rates:
            # Delegations at this rate
            delegations = d_max * np.exp(-k * rate)

            # Create temporary config
            temp_config = ValidatorConfig(
                total_stake=delegations + cfg.self_stake,
                self_stake=cfg.self_stake,
                commission_rate=rate,
                base_apy=cfg.base_apy,
                sol_price=cfg.sol_price,
                jito_enabled=cfg.jito_enabled,
                hardware_cost=cfg.hardware_cost,
                bandwidth_cost=cfg.bandwidth_cost,
                colocation_cost=cfg.colocation_cost,
                maintenance_cost=cfg.maintenance_cost,
            )

            temp_econ = ValidatorEconomics(temp_config)
            profit = temp_econ.annual_profit()['net_profit_sol']
            profits.append(profit)

        optimal_idx = np.argmax(profits)
        return rates[optimal_idx], profits[optimal_idx]

    def summary(self) -> str:
        """Generate human-readable summary."""
        cfg = self.config
        profit = self.annual_profit()
        breakeven = self.breakeven_stake()

        lines = [
            "=" * 50,
            "VALIDATOR ECONOMICS SUMMARY",
            "=" * 50,
            "",
            "CONFIGURATION:",
            f"  Total Stake: {format_currency(cfg.total_stake, 'SOL')}",
            f"  Self Stake: {format_currency(cfg.self_stake, 'SOL')}",
            f"  Commission Rate: {cfg.commission_rate:.1%}",
            f"  Base APY: {cfg.base_apy:.2%}",
            f"  SOL Price: ${cfg.sol_price:,.2f}",
            f"  Jito Enabled: {cfg.jito_enabled}",
            "",
            "ANNUAL REVENUE:",
            f"  Staking Revenue: {format_currency(profit['staking_revenue_sol'], 'SOL')}",
            f"  MEV Revenue: {format_currency(profit['mev_revenue_sol'], 'SOL')}",
            f"  Total Revenue: {format_currency(profit['total_revenue_sol'], 'SOL')}",
            f"               = {format_currency(profit['total_revenue_usd'])}",
            "",
            "ANNUAL COSTS:",
            f"  Total Costs: {format_currency(profit['total_costs_sol'], 'SOL')}",
            f"            = {format_currency(profit['total_costs_usd'])}",
            "",
            "PROFITABILITY:",
            f"  Net Profit: {format_currency(profit['net_profit_sol'], 'SOL')}",
            f"           = {format_currency(profit['net_profit_usd'])}",
            f"  Monthly: {format_currency(profit['monthly_profit_usd'])}/month",
            f"  ROI on Self-Stake: {profit['roi_on_self_stake']:.1%}",
            "",
            f"BREAKEVEN STAKE: {format_currency(breakeven, 'SOL')}",
            "=" * 50,
        ]

        return "\n".join(lines)


def compare_commission_rates(
    base_config: ValidatorConfig,
    rates: list = [0.0, 0.02, 0.05, 0.08, 0.10]
) -> Dict[float, Dict]:
    """
    Compare profitability at different commission rates.

    Returns:
        Dictionary mapping rate to profit metrics
    """
    results = {}

    for rate in rates:
        config = ValidatorConfig(
            total_stake=base_config.total_stake,
            self_stake=base_config.self_stake,
            commission_rate=rate,
            base_apy=base_config.base_apy,
            sol_price=base_config.sol_price,
            jito_enabled=base_config.jito_enabled,
            hardware_cost=base_config.hardware_cost,
            bandwidth_cost=base_config.bandwidth_cost,
            colocation_cost=base_config.colocation_cost,
            maintenance_cost=base_config.maintenance_cost,
        )

        econ = ValidatorEconomics(config)
        results[rate] = econ.annual_profit()

    return results


if __name__ == "__main__":
    # Test validator economics
    config = ValidatorConfig(
        total_stake=500_000,
        self_stake=50_000,
        commission_rate=0.05,
        base_apy=0.075,
        sol_price=150,
        jito_enabled=True,
        hardware_cost=400,
        bandwidth_cost=150,
        colocation_cost=300,
        maintenance_cost=150,
    )

    econ = ValidatorEconomics(config)
    print(econ.summary())

    print("\n--- Commission Rate Comparison ---")
    comparison = compare_commission_rates(config)
    for rate, metrics in comparison.items():
        print(f"Commission {rate:.0%}: "
              f"Profit = {format_currency(metrics['net_profit_sol'], 'SOL')} "
              f"({format_currency(metrics['net_profit_usd'])})")

    print("\n--- Optimal Commission ---")
    opt_rate, opt_profit = econ.optimal_commission_rate()
    print(f"Optimal Rate: {opt_rate:.2%}")
    print(f"Expected Profit: {format_currency(opt_profit, 'SOL')}")
