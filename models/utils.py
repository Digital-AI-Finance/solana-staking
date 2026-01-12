"""
Utility functions and constants for SOL Staking Capital Structure models.
"""
import numpy as np
from scipy import stats

# =============================================================================
# SOLANA NETWORK CONSTANTS (Verified 2025-2026)
# =============================================================================

# Staking Parameters
CURRENT_STAKING_APY = 0.075  # 7.5% base APY
STAKED_SUPPLY_RATIO = 0.6857  # 68.57% of supply staked
TOTAL_STAKED_SOL = 387_000_000  # 387M SOL staked
CURRENT_SOL_PRICE = 150.0  # USD

# Inflation Schedule
INITIAL_INFLATION_RATE = 0.08  # 8%
DISINFLATION_RATE = 0.15  # 15% annual decrease
TERMINAL_INFLATION_RATE = 0.015  # 1.5%
CURRENT_INFLATION_RATE = 0.04071  # 4.071% as of 2025

# Validator Economics
SLOTS_PER_EPOCH = 432_000
EPOCHS_PER_YEAR = 365 / 2.5  # ~146 epochs
VOTE_TX_FEE = 0.000005  # SOL per vote transaction
DAILY_VOTE_COST_SOL = 1.1  # ~1.1 SOL/day
ANNUAL_VOTE_COST_SOL = 400  # ~400 SOL/year

# LST Market Share (2025)
LST_MARKET = {
    'Jito (JitoSOL)': {'share': 0.76, 'tvl_sol': 17_600_000},
    'Marinade (mSOL)': {'share': 0.22, 'tvl_sol': 5_280_000},
    'Jupiter (jupSOL)': {'share': 0.05, 'tvl_sol': 3_880_000},
    'Binance (bnSOL)': {'share': 0.10, 'tvl_sol': 8_160_000},
}

# MEV Parameters (Jito)
JITO_VALIDATOR_SHARE = 0.03  # 3% of tips go to validators
AVERAGE_MEV_TIPS_PER_SLOT = 0.01  # SOL estimate

# =============================================================================
# COLOR PALETTE (Solana-Themed)
# =============================================================================

COLORS = {
    'SOL_PURPLE': '#9945FF',
    'SOL_GREEN': '#14F195',
    'DARK_BG': '#0D1117',
    'CARD_BG': '#161B22',
    'ACCENT_BLUE': '#58A6FF',
    'WARNING_ORANGE': '#F0883E',
    'ERROR_RED': '#F85149',
    'SUCCESS_GREEN': '#3FB950',
    'TEXT_PRIMARY': '#C9D1D9',
    'TEXT_SECONDARY': '#8B949E',
    'BORDER': '#30363D',
}

# Chart colors for multiple series
CHART_COLORS = [
    '#9945FF',  # SOL Purple
    '#14F195',  # SOL Green
    '#58A6FF',  # Accent Blue
    '#F0883E',  # Warning Orange
    '#F85149',  # Error Red
    '#3FB950',  # Success Green
    '#A371F7',  # Light Purple
    '#79C0FF',  # Light Blue
]

# =============================================================================
# MATPLOTLIB RCPARAMS (Scaled for Beamer slides)
# =============================================================================

CHART_RCPARAMS = {
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
}

# =============================================================================
# FINANCIAL CONSTANTS
# =============================================================================

# Risk-Free Rate (US Treasury)
RISK_FREE_RATE = 0.045  # 4.5%

# SOL Volatility (Historical annualized)
SOL_VOLATILITY = 0.80  # 80% annualized

# Corporate Tax Rate
CORPORATE_TAX_RATE = 0.21  # 21% US

# Typical PIPE Discount
PIPE_DISCOUNT = 0.10  # 10%

# MicroStrategy-style Convertible Parameters
MSTR_CONVERSION_PREMIUM = 0.35  # 35%
MSTR_COUPON_RATE = 0.0  # 0% coupon

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_inflation_rate(years_from_start: float) -> float:
    """
    Calculate Solana inflation rate at a given point in time.

    Parameters:
        years_from_start: Years from inflation activation (Feb 2021)

    Returns:
        Current inflation rate as decimal
    """
    rate = INITIAL_INFLATION_RATE * ((1 - DISINFLATION_RATE) ** years_from_start)
    return max(rate, TERMINAL_INFLATION_RATE)


def calculate_staking_apy(inflation_rate: float, staked_ratio: float) -> float:
    """
    Calculate staking APY based on inflation and staked ratio.

    Parameters:
        inflation_rate: Current inflation rate
        staked_ratio: Percentage of supply staked

    Returns:
        Staking APY as decimal
    """
    # Staking rewards = inflation / staked_ratio (simplified)
    return inflation_rate / staked_ratio


def sol_to_usd(sol_amount: float, sol_price: float = CURRENT_SOL_PRICE) -> float:
    """Convert SOL to USD."""
    return sol_amount * sol_price


def usd_to_sol(usd_amount: float, sol_price: float = CURRENT_SOL_PRICE) -> float:
    """Convert USD to SOL."""
    return usd_amount / sol_price


def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency string."""
    if currency == 'USD':
        if abs(amount) >= 1e9:
            return f"${amount/1e9:.2f}B"
        elif abs(amount) >= 1e6:
            return f"${amount/1e6:.2f}M"
        elif abs(amount) >= 1e3:
            return f"${amount/1e3:.1f}K"
        else:
            return f"${amount:,.2f}"
    elif currency == 'SOL':
        if abs(amount) >= 1e6:
            return f"{amount/1e6:.2f}M SOL"
        elif abs(amount) >= 1e3:
            return f"{amount/1e3:.1f}K SOL"
        else:
            return f"{amount:,.2f} SOL"
    return str(amount)


def annualize_return(total_return: float, periods: int, periods_per_year: int = 12) -> float:
    """
    Convert total return to annualized return.

    Parameters:
        total_return: Total return as decimal (e.g., 0.5 for 50%)
        periods: Number of periods
        periods_per_year: Periods per year (12 for monthly, 252 for daily)

    Returns:
        Annualized return as decimal
    """
    years = periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calculate Sharpe ratio from returns array.

    Parameters:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown from price series.

    Parameters:
        prices: Array of prices

    Returns:
        Maximum drawdown as decimal (negative)
    """
    running_max = np.maximum.accumulate(prices)
    drawdowns = (prices - running_max) / running_max
    return np.min(drawdowns)


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) at given confidence level.

    Parameters:
        returns: Array of returns
        alpha: Confidence level (0.05 for 95% VaR)

    Returns:
        VaR as positive number (loss)
    """
    return -np.percentile(returns, alpha * 100)


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    Parameters:
        returns: Array of returns
        alpha: Confidence level

    Returns:
        CVaR as positive number (expected loss beyond VaR)
    """
    var = value_at_risk(returns, alpha)
    return -np.mean(returns[returns <= -var])


if __name__ == "__main__":
    # Test functions
    print("=== Solana Network Constants ===")
    print(f"Current Staking APY: {CURRENT_STAKING_APY:.2%}")
    print(f"Staked Supply: {STAKED_SUPPLY_RATIO:.2%}")
    print(f"Current Inflation: {CURRENT_INFLATION_RATE:.3%}")

    print("\n=== Inflation Schedule ===")
    for year in range(0, 15):
        rate = calculate_inflation_rate(year)
        print(f"Year {year}: {rate:.3%}")

    print("\n=== Sample Calculations ===")
    print(f"100 SOL = {format_currency(sol_to_usd(100))}")
    print(f"$10,000 = {format_currency(usd_to_sol(10000), 'SOL')}")
