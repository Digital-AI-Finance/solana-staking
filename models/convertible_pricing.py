"""
Convertible Bond Pricing Model.

Implements decomposition approach:
    V_convertible = V_bond + V_call_option

Based on MicroStrategy convertible structure for SOL application.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import optimize
from black_scholes import call_price, all_greeks_call, implied_volatility_call
from utils import RISK_FREE_RATE, SOL_VOLATILITY, CURRENT_SOL_PRICE


@dataclass
class ConvertibleBondTerms:
    """Terms of a convertible bond issuance."""

    face_value: float = 100_000_000  # $100M face value
    conversion_price: float = 200  # Conversion price per SOL
    maturity_years: float = 5.0  # Years to maturity
    coupon_rate: float = 0.0  # 0% coupon (MicroStrategy style)
    coupon_frequency: int = 2  # Semi-annual (if any)

    # Market conditions
    current_sol_price: float = CURRENT_SOL_PRICE
    risk_free_rate: float = RISK_FREE_RATE
    sol_volatility: float = SOL_VOLATILITY

    # Credit spread for bond floor
    credit_spread: float = 0.03  # 3% credit spread

    @property
    def conversion_ratio(self) -> float:
        """Number of SOL per $1000 bond."""
        return 1000 / self.conversion_price

    @property
    def conversion_premium(self) -> float:
        """Premium over current stock price."""
        return (self.conversion_price - self.current_sol_price) / self.current_sol_price

    @property
    def parity_value(self) -> float:
        """Conversion value per $1000 bond."""
        return self.conversion_ratio * self.current_sol_price

    @property
    def total_sol_if_converted(self) -> float:
        """Total SOL received if entire issue converts."""
        return self.face_value / self.conversion_price


class ConvertibleBondPricer:
    """
    Price convertible bonds using bond + option decomposition.

    Decomposition:
        V_cb = V_straight_bond + V_embedded_call

    Where:
        V_straight_bond = PV of coupons + PV of principal (discounted at r + spread)
        V_embedded_call = Black-Scholes call on SOL * conversion_ratio
    """

    def __init__(self, terms: ConvertibleBondTerms):
        self.terms = terms

    def bond_floor(self, per_1000: bool = True) -> float:
        """
        Calculate straight bond floor (investment value).

        For zero-coupon:
            V_bond = Face * exp(-(r + spread) * T)

        For coupon bond:
            V_bond = sum(C * exp(-(r+spread)*t)) + Face * exp(-(r+spread)*T)

        Parameters:
            per_1000: If True, return value per $1000 face value

        Returns:
            Bond floor value
        """
        t = self.terms
        discount_rate = t.risk_free_rate + t.credit_spread

        if t.coupon_rate == 0:
            # Zero-coupon bond
            bond_value = np.exp(-discount_rate * t.maturity_years)
        else:
            # Coupon bond
            n_coupons = int(t.maturity_years * t.coupon_frequency)
            coupon_payment = t.coupon_rate / t.coupon_frequency

            bond_value = 0
            for i in range(1, n_coupons + 1):
                time_to_coupon = i / t.coupon_frequency
                bond_value += coupon_payment * np.exp(-discount_rate * time_to_coupon)

            # Add principal
            bond_value += np.exp(-discount_rate * t.maturity_years)

        if per_1000:
            return bond_value * 1000
        else:
            return bond_value * t.face_value

    def embedded_option_value(self, per_1000: bool = True) -> float:
        """
        Calculate value of embedded call option.

        Uses Black-Scholes with:
            S = current SOL price
            K = conversion price
            T = maturity
            sigma = SOL volatility

        Value per $1000 bond = call_price * conversion_ratio

        Returns:
            Option value
        """
        t = self.terms

        # Call price per SOL
        call = call_price(
            S=t.current_sol_price,
            K=t.conversion_price,
            T=t.maturity_years,
            r=t.risk_free_rate,
            sigma=t.sol_volatility
        )

        # Option value = call * conversion_ratio
        option_value = call * t.conversion_ratio

        if per_1000:
            return option_value
        else:
            return option_value * t.face_value / 1000

    def convertible_value(self, per_1000: bool = True) -> Dict[str, float]:
        """
        Calculate total convertible bond value.

        V_cb = V_bond + V_option

        Returns:
            Dictionary with value breakdown
        """
        bond_floor = self.bond_floor(per_1000)
        option_value = self.embedded_option_value(per_1000)
        parity = self.terms.parity_value if per_1000 else self.terms.parity_value * self.terms.face_value / 1000

        total_value = bond_floor + option_value

        # Investment premium (over bond floor)
        investment_premium = (total_value - bond_floor) / bond_floor

        # Conversion premium (over parity)
        conversion_premium_actual = (total_value - parity) / parity if parity > 0 else np.inf

        return {
            'bond_floor': bond_floor,
            'option_value': option_value,
            'parity_value': parity,
            'total_value': total_value,
            'investment_premium': investment_premium,
            'conversion_premium_actual': conversion_premium_actual,
            'conversion_premium_stated': self.terms.conversion_premium,
        }

    def greeks(self) -> Dict[str, float]:
        """
        Calculate Greeks for the convertible (via embedded option).

        Returns:
            Dictionary with Greeks (scaled by conversion ratio)
        """
        t = self.terms

        option_greeks = all_greeks_call(
            S=t.current_sol_price,
            K=t.conversion_price,
            T=t.maturity_years,
            r=t.risk_free_rate,
            sigma=t.sol_volatility
        )

        # Scale by conversion ratio
        return {
            'delta': option_greeks['delta'] * t.conversion_ratio,
            'gamma': option_greeks['gamma'] * t.conversion_ratio,
            'vega': option_greeks['vega'] * t.conversion_ratio,
            'theta': option_greeks['theta'] * t.conversion_ratio,
            'rho': option_greeks['rho'] * t.conversion_ratio,
        }

    def breakeven_price(self) -> float:
        """
        Calculate SOL price at which conversion equals bond floor value.

        At breakeven:
            conversion_ratio * S_breakeven = bond_floor
            S_breakeven = bond_floor / conversion_ratio

        This is the price below which the bond floor provides more value
        than conversion, making the convertible trade like a straight bond.
        """
        t = self.terms
        bond_floor_value = self.bond_floor(per_1000=True)
        conversion_ratio = 1000 / t.conversion_price  # SOL per $1000 face

        return bond_floor_value / conversion_ratio

    def implied_volatility_from_market(self, market_price: float) -> float:
        """
        Back out implied volatility from market price of convertible.

        Parameters:
            market_price: Market price per $1000 face value

        Returns:
            Implied volatility
        """
        t = self.terms

        # Subtract bond floor to get option value
        option_value_implied = market_price - self.bond_floor(per_1000=True)

        if option_value_implied <= 0:
            return 0.0

        # Option value per SOL
        option_per_sol = option_value_implied / t.conversion_ratio

        # Implied vol
        return implied_volatility_call(
            market_price=option_per_sol,
            S=t.current_sol_price,
            K=t.conversion_price,
            T=t.maturity_years,
            r=t.risk_free_rate
        )

    def payoff_at_maturity(self, sol_prices: np.ndarray) -> np.ndarray:
        """
        Calculate convertible payoff at maturity for range of SOL prices.

        At maturity:
            Payoff = max(Face_Value, Conversion_Value)
                   = max(1000, conversion_ratio * S_T)  per $1000

        Parameters:
            sol_prices: Array of potential SOL prices at maturity

        Returns:
            Array of payoffs per $1000 face value
        """
        conversion_value = self.terms.conversion_ratio * sol_prices
        face_value = 1000 * np.ones_like(sol_prices)

        return np.maximum(face_value, conversion_value)

    def sensitivity_analysis(
        self,
        price_range: Tuple[float, float] = (50, 400),
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Analyze convertible value sensitivity to SOL price.

        Returns:
            Dictionary with price array and corresponding values
        """
        prices = np.linspace(price_range[0], price_range[1], n_points)
        values = []
        deltas = []
        parities = []

        for price in prices:
            temp_terms = ConvertibleBondTerms(
                face_value=self.terms.face_value,
                conversion_price=self.terms.conversion_price,
                maturity_years=self.terms.maturity_years,
                coupon_rate=self.terms.coupon_rate,
                current_sol_price=price,
                risk_free_rate=self.terms.risk_free_rate,
                sol_volatility=self.terms.sol_volatility,
                credit_spread=self.terms.credit_spread,
            )

            temp_pricer = ConvertibleBondPricer(temp_terms)
            val = temp_pricer.convertible_value()
            greeks = temp_pricer.greeks()

            values.append(val['total_value'])
            deltas.append(greeks['delta'])
            parities.append(temp_terms.parity_value)

        return {
            'sol_prices': prices,
            'convertible_values': np.array(values),
            'deltas': np.array(deltas),
            'parity_values': np.array(parities),
            'bond_floor': self.bond_floor(per_1000=True) * np.ones(n_points),
        }


def mstr_style_issuance(
    capital_to_raise: float,
    current_sol_price: float,
    conversion_premium: float = 0.35,
    maturity_years: float = 5.0,
    coupon_rate: float = 0.0,
    volatility: float = SOL_VOLATILITY
) -> Dict:
    """
    Model a MicroStrategy-style convertible issuance for SOL.

    Parameters:
        capital_to_raise: USD amount to raise
        current_sol_price: Current SOL price
        conversion_premium: Premium over current price
        maturity_years: Bond maturity
        coupon_rate: Annual coupon (typically 0%)
        volatility: SOL volatility

    Returns:
        Dictionary with issuance details and value analysis
    """
    conversion_price = current_sol_price * (1 + conversion_premium)

    terms = ConvertibleBondTerms(
        face_value=capital_to_raise,
        conversion_price=conversion_price,
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        current_sol_price=current_sol_price,
        sol_volatility=volatility,
    )

    pricer = ConvertibleBondPricer(terms)
    valuation = pricer.convertible_value(per_1000=True)
    greeks = pricer.greeks()

    # SOL that can be purchased with proceeds
    sol_purchasable = capital_to_raise / current_sol_price

    # SOL issued if converted
    sol_if_converted = terms.total_sol_if_converted

    # Effective cost of capital
    # If SOL appreciates to conversion price, investor converts
    # Issuer pays face value in SOL at conversion price
    effective_dilution = sol_if_converted / sol_purchasable - 1

    return {
        'capital_raised': capital_to_raise,
        'conversion_price': conversion_price,
        'conversion_premium': conversion_premium,
        'sol_purchasable_now': sol_purchasable,
        'sol_if_converted': sol_if_converted,
        'effective_dilution_if_convert': effective_dilution,
        'bond_floor_per_1000': valuation['bond_floor'],
        'option_value_per_1000': valuation['option_value'],
        'total_value_per_1000': valuation['total_value'],
        'delta_per_1000': greeks['delta'],
        'terms': terms,
    }


if __name__ == "__main__":
    print("=== Convertible Bond Pricing Test ===\n")

    # MicroStrategy-style terms
    terms = ConvertibleBondTerms(
        face_value=100_000_000,  # $100M
        conversion_price=200,  # $200 per SOL
        maturity_years=5.0,
        coupon_rate=0.0,  # Zero coupon
        current_sol_price=150,
        risk_free_rate=0.045,
        sol_volatility=0.80,
        credit_spread=0.03,
    )

    pricer = ConvertibleBondPricer(terms)

    print("--- Bond Terms ---")
    print(f"Face Value: ${terms.face_value:,.0f}")
    print(f"Conversion Price: ${terms.conversion_price}")
    print(f"Conversion Ratio: {terms.conversion_ratio:.4f} SOL per $1000")
    print(f"Conversion Premium: {terms.conversion_premium:.1%}")
    print(f"Maturity: {terms.maturity_years} years")
    print(f"Coupon: {terms.coupon_rate:.1%}")
    print()

    print("--- Valuation (per $1000 face) ---")
    val = pricer.convertible_value()
    print(f"Bond Floor: ${val['bond_floor']:.2f}")
    print(f"Option Value: ${val['option_value']:.2f}")
    print(f"Parity Value: ${val['parity_value']:.2f}")
    print(f"Total CB Value: ${val['total_value']:.2f}")
    print(f"Investment Premium: {val['investment_premium']:.1%}")
    print()

    print("--- Greeks (per $1000 face) ---")
    greeks = pricer.greeks()
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    print()

    print("--- MicroStrategy-Style Issuance ---")
    issuance = mstr_style_issuance(
        capital_to_raise=500_000_000,  # $500M
        current_sol_price=150,
        conversion_premium=0.35,
    )
    print(f"Capital Raised: ${issuance['capital_raised']:,.0f}")
    print(f"Conversion Price: ${issuance['conversion_price']:.2f}")
    print(f"SOL Purchasable Now: {issuance['sol_purchasable_now']:,.0f}")
    print(f"SOL if Converted: {issuance['sol_if_converted']:,.0f}")
    print(f"Effective Dilution if Convert: {issuance['effective_dilution_if_convert']:.1%}")
