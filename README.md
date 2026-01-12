# SOL Staking & Capital Structure Optimization

A comprehensive framework for permanent SOL accumulation through staking, convertible bonds, and strategic capital deployment.

## Live Demo

**GitHub Pages**: [https://digital-ai-finance.github.io/solana-staking/](https://digital-ai-finance.github.io/solana-staking/)

## Features

### Interactive Calculators (Client-Side JS)
- **Validator Economics**: Calculate profitability by stake level, commission rate, and infrastructure costs
- **Convertible Bond Analyzer**: Price convertible bonds with Black-Scholes, compute Greeks
- **Options Strategy Builder**: Design covered calls, puts, and collars for yield enhancement
- **Monte Carlo Simulator**: Simulate SOL price paths with Merton Jump-Diffusion model
- **Capital Structure Optimizer**: Find optimal leverage ratio and minimize WACC

### Python Models
- `monte_carlo.py` - Price path simulation with jump-diffusion
- `black_scholes.py` - Full options pricing with all Greeks
- `validator_economics.py` - Validator P&L model
- `convertible_pricing.py` - Bond + embedded option valuation
- `capital_structure.py` - WACC optimization, Modigliani-Miller framework

### Beamer Slides (40 pages)
Eight modules covering:
1. Solana Staking Fundamentals
2. Validator Business Model
3. Corporate Finance - SOL Accumulation
4. Capital Structure Optimization
5. Options Strategies for Yield Enhancement
6. Monte Carlo Simulations
7. Opportunistic Acquisition Framework
8. Hybrid Model: Validator + Corp Finance

## Current Solana Metrics (January 2026)

| Metric | Value |
|--------|-------|
| Staking APY | 7.5% |
| Staked Supply | 68.6% |
| Inflation Rate | 4.07% |
| LST TVL | $10.7B |
| Jito Market Share | 76% |

## Mathematical Framework

### Stochastic Process (Merton Jump-Diffusion)
```
dS = (mu - lambda*kappa)*S*dt + sigma*S*dW + S*dJ
```

### Convertible Valuation
```
V_CB = V_bond + V_call_option
```

### Capital Structure (Modigliani-Miller)
```
V_L = V_U + Tax_Shield - Distress_Cost
```

### WACC
```
WACC = (E/V)*r_e + (D/V)*r_d*(1-tau)
```

## Project Structure

```
Solana-Staking/
├── docs/                     # GitHub Pages site
│   ├── index.html           # Landing page
│   ├── calculators/         # 5 interactive tools
│   ├── css/                 # Solana-themed styling
│   ├── js/                  # Black-Scholes, utilities
│   └── slides/              # PDF presentation
├── models/                   # Python simulation code
├── charts/                   # Generated visualizations
└── slides/                   # LaTeX source
```

## Local Development

### Run Python models
```bash
cd models
python monte_carlo.py
python validator_economics.py
```

### Generate charts
```bash
cd charts/01_staking_yield_history
python chart.py
```

### Compile slides
```bash
cd slides
pdflatex 20260112_1720_solana_staking.tex
```

## License

MIT License

## References

- Solana Documentation: https://docs.solana.com
- Jito Network: https://jito.network
- MicroStrategy Investor Relations (convertible bond structure)
- Hull, J.C. - Options, Futures, and Other Derivatives
