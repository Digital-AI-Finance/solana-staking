"""
Black-Scholes Options Pricing Model with Greeks.

Full mathematical derivations for convertible bond and options analysis.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from utils import RISK_FREE_RATE, SOL_VOLATILITY


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 parameter for Black-Scholes.

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))

    Parameters:
        S: Current spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility (annualized)

    Returns:
        d1 value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 parameter for Black-Scholes.

    d2 = d1 - sigma * sqrt(T)

    Parameters:
        S: Current spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility (annualized)

    Returns:
        d2 value
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
               sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate European call option price using Black-Scholes.

    C = S * N(d1) - K * exp(-r*T) * N(d2)

    Parameters:
        S: Current spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    return S * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(d2_val)


def put_price(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
              sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate European put option price using Black-Scholes.

    P = K * exp(-r*T) * N(-d2) - S * N(-d1)

    Parameters:
        S: Current spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    return K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S * stats.norm.cdf(-d1_val)


# =============================================================================
# GREEKS
# =============================================================================

def delta_call(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
               sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate delta for call option.

    Delta_call = N(d1)

    Interpretation: Change in option price per $1 change in underlying.
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    return stats.norm.cdf(d1(S, K, T, r, sigma))


def delta_put(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
              sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate delta for put option.

    Delta_put = N(d1) - 1 = -N(-d1)

    Interpretation: Change in option price per $1 change in underlying.
    """
    if T <= 0:
        return -1.0 if S < K else 0.0
    return stats.norm.cdf(d1(S, K, T, r, sigma)) - 1


def gamma(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
          sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate gamma (same for calls and puts).

    Gamma = N'(d1) / (S * sigma * sqrt(T))

    Interpretation: Rate of change of delta per $1 change in underlying.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    return stats.norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
         sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate vega (same for calls and puts).

    Vega = S * sqrt(T) * N'(d1)

    Interpretation: Change in option price per 1% change in volatility.
    Returns value per 1% (0.01) change in volatility.
    """
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    # Divide by 100 to get per 1% change
    return S * np.sqrt(T) * stats.norm.pdf(d1_val) / 100


def theta_call(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
               sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate theta for call option (per day).

    Theta_call = -[S * N'(d1) * sigma] / [2 * sqrt(T)] - r * K * exp(-r*T) * N(d2)

    Interpretation: Change in option price per day (time decay).
    """
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    term1 = -(S * stats.norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2_val)

    # Convert to per-day (divide by 365)
    return (term1 + term2) / 365


def theta_put(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
              sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate theta for put option (per day).

    Interpretation: Change in option price per day (time decay).
    """
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    term1 = -(S * stats.norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2_val)

    return (term1 + term2) / 365


def rho_call(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
             sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate rho for call option.

    Rho_call = K * T * exp(-r*T) * N(d2)

    Interpretation: Change in option price per 1% change in interest rate.
    """
    if T <= 0:
        return 0.0
    d2_val = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * stats.norm.cdf(d2_val) / 100


def rho_put(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
            sigma: float = SOL_VOLATILITY) -> float:
    """
    Calculate rho for put option.

    Rho_put = -K * T * exp(-r*T) * N(-d2)

    Interpretation: Change in option price per 1% change in interest rate.
    """
    if T <= 0:
        return 0.0
    d2_val = d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2_val) / 100


def all_greeks_call(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
                    sigma: float = SOL_VOLATILITY) -> Dict[str, float]:
    """
    Calculate all Greeks for a call option.

    Returns:
        Dictionary with delta, gamma, vega, theta, rho
    """
    return {
        'price': call_price(S, K, T, r, sigma),
        'delta': delta_call(S, K, T, r, sigma),
        'gamma': gamma(S, K, T, r, sigma),
        'vega': vega(S, K, T, r, sigma),
        'theta': theta_call(S, K, T, r, sigma),
        'rho': rho_call(S, K, T, r, sigma),
    }


def all_greeks_put(S: float, K: float, T: float, r: float = RISK_FREE_RATE,
                   sigma: float = SOL_VOLATILITY) -> Dict[str, float]:
    """
    Calculate all Greeks for a put option.

    Returns:
        Dictionary with delta, gamma, vega, theta, rho
    """
    return {
        'price': put_price(S, K, T, r, sigma),
        'delta': delta_put(S, K, T, r, sigma),
        'gamma': gamma(S, K, T, r, sigma),
        'vega': vega(S, K, T, r, sigma),
        'theta': theta_put(S, K, T, r, sigma),
        'rho': rho_put(S, K, T, r, sigma),
    }


# =============================================================================
# IMPLIED VOLATILITY
# =============================================================================

def implied_volatility_call(market_price: float, S: float, K: float, T: float,
                            r: float = RISK_FREE_RATE, tol: float = 1e-6,
                            max_iter: int = 100) -> float:
    """
    Calculate implied volatility from market call price using Newton-Raphson.

    Parameters:
        market_price: Observed market price of call
        S: Current spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Implied volatility
    """
    if T <= 0:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * market_price / S

    for _ in range(max_iter):
        price = call_price(S, K, T, r, sigma)
        vega_val = vega(S, K, T, r, sigma) * 100  # Undo the /100 in vega function

        if vega_val < 1e-10:
            break

        diff = market_price - price
        if abs(diff) < tol:
            return sigma

        sigma = sigma + diff / vega_val
        sigma = max(0.001, min(sigma, 5.0))  # Bound sigma

    return sigma


def implied_volatility_put(market_price: float, S: float, K: float, T: float,
                           r: float = RISK_FREE_RATE, tol: float = 1e-6,
                           max_iter: int = 100) -> float:
    """
    Calculate implied volatility from market put price using Newton-Raphson.
    """
    if T <= 0:
        return 0.0

    sigma = np.sqrt(2 * np.pi / T) * market_price / S

    for _ in range(max_iter):
        price = put_price(S, K, T, r, sigma)
        vega_val = vega(S, K, T, r, sigma) * 100

        if vega_val < 1e-10:
            break

        diff = market_price - price
        if abs(diff) < tol:
            return sigma

        sigma = sigma + diff / vega_val
        sigma = max(0.001, min(sigma, 5.0))

    return sigma


# =============================================================================
# OPTIONS STRATEGY PAYOFFS
# =============================================================================

def covered_call_payoff(S_T: np.ndarray, S_0: float, K: float,
                        premium: float) -> np.ndarray:
    """
    Calculate covered call payoff at expiration.

    Payoff = min(K, S_T) - S_0 + premium

    Parameters:
        S_T: Array of prices at expiration
        S_0: Initial stock price
        K: Strike price
        premium: Call premium received

    Returns:
        Array of payoffs
    """
    return np.minimum(K, S_T) - S_0 + premium


def cash_secured_put_payoff(S_T: np.ndarray, K: float, premium: float) -> np.ndarray:
    """
    Calculate cash-secured put payoff at expiration.

    Payoff = premium - max(K - S_T, 0)

    Parameters:
        S_T: Array of prices at expiration
        K: Strike price
        premium: Put premium received

    Returns:
        Array of payoffs (relative to cash secured)
    """
    return premium - np.maximum(K - S_T, 0)


def collar_payoff(S_T: np.ndarray, S_0: float, K_put: float, K_call: float,
                  put_premium: float, call_premium: float) -> np.ndarray:
    """
    Calculate collar strategy payoff at expiration.

    Long stock + Long put + Short call

    Payoff = S_T - S_0 + max(K_put - S_T, 0) - max(S_T - K_call, 0)
             - put_premium + call_premium

    Parameters:
        S_T: Array of prices at expiration
        S_0: Initial stock price
        K_put: Put strike (downside protection)
        K_call: Call strike (upside cap)
        put_premium: Put premium paid
        call_premium: Call premium received

    Returns:
        Array of payoffs
    """
    stock_pnl = S_T - S_0
    put_pnl = np.maximum(K_put - S_T, 0) - put_premium
    call_pnl = call_premium - np.maximum(S_T - K_call, 0)

    return stock_pnl + put_pnl + call_pnl


def straddle_payoff(S_T: np.ndarray, K: float, call_premium: float,
                    put_premium: float) -> np.ndarray:
    """
    Calculate long straddle payoff at expiration.

    Long call + Long put at same strike

    Payoff = max(S_T - K, 0) + max(K - S_T, 0) - call_premium - put_premium
           = |S_T - K| - total_premium

    Parameters:
        S_T: Array of prices at expiration
        K: Strike price (same for call and put)
        call_premium: Call premium paid
        put_premium: Put premium paid

    Returns:
        Array of payoffs
    """
    return np.abs(S_T - K) - call_premium - put_premium


if __name__ == "__main__":
    # Test calculations
    S = 150  # SOL price
    K = 180  # Strike
    T = 0.5  # 6 months
    r = 0.045
    sigma = 0.80

    print("=== Black-Scholes Test ===")
    print(f"Spot: ${S}, Strike: ${K}, Time: {T}y, r: {r:.1%}, sigma: {sigma:.0%}")
    print()

    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    print(f"Call Price: ${call:.2f}")
    print(f"Put Price: ${put:.2f}")
    print()

    print("=== Call Greeks ===")
    greeks = all_greeks_call(S, K, T, r, sigma)
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    print()

    print("=== Put-Call Parity Check ===")
    pcp_call = put + S - K * np.exp(-r * T)
    print(f"Call from PCP: ${pcp_call:.2f} (actual: ${call:.2f})")
    print(f"Difference: ${abs(pcp_call - call):.6f}")
