/**
 * Utility functions for SOL Staking Calculators
 */

// Format number as currency
function formatCurrency(amount, currency = 'USD') {
    if (currency === 'USD') {
        if (Math.abs(amount) >= 1e9) {
            return '$' + (amount / 1e9).toFixed(2) + 'B';
        } else if (Math.abs(amount) >= 1e6) {
            return '$' + (amount / 1e6).toFixed(2) + 'M';
        } else if (Math.abs(amount) >= 1e3) {
            return '$' + (amount / 1e3).toFixed(1) + 'K';
        } else {
            return '$' + amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
    } else if (currency === 'SOL') {
        if (Math.abs(amount) >= 1e6) {
            return (amount / 1e6).toFixed(2) + 'M SOL';
        } else if (Math.abs(amount) >= 1e3) {
            return (amount / 1e3).toFixed(1) + 'K SOL';
        } else {
            return amount.toFixed(2) + ' SOL';
        }
    }
    return amount.toString();
}

// Format percentage
function formatPercent(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

// Format number with commas
function formatNumber(num, decimals = 0) {
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

// Standard Normal CDF (for Black-Scholes)
function normalCDF(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Standard Normal PDF
function normalPDF(x) {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// Black-Scholes Call Price
function blackScholesCall(S, K, T, r, sigma) {
    if (T <= 0) return Math.max(S - K, 0);

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
}

// Black-Scholes Put Price
function blackScholesPut(S, K, T, r, sigma) {
    if (T <= 0) return Math.max(K - S, 0);

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    return K * Math.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
}

// Delta (Call)
function deltaCall(S, K, T, r, sigma) {
    if (T <= 0) return S > K ? 1 : 0;

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    return normalCDF(d1);
}

// Gamma
function gamma(S, K, T, r, sigma) {
    if (T <= 0 || sigma <= 0) return 0;

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    return normalPDF(d1) / (S * sigma * Math.sqrt(T));
}

// Vega (per 1% change)
function vega(S, K, T, r, sigma) {
    if (T <= 0) return 0;

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    return S * Math.sqrt(T) * normalPDF(d1) / 100;
}

// Theta (Call, per day)
function thetaCall(S, K, T, r, sigma) {
    if (T <= 0) return 0;

    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    const term1 = -(S * normalPDF(d1) * sigma) / (2 * Math.sqrt(T));
    const term2 = -r * K * Math.exp(-r * T) * normalCDF(d2);

    return (term1 + term2) / 365;
}

// Random number from normal distribution (Box-Muller)
function randomNormal(mean = 0, std = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * std + mean;
}

// Poisson random number
function randomPoisson(lambda) {
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;

    do {
        k++;
        p *= Math.random();
    } while (p > L);

    return k - 1;
}

// Percentile calculation
function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);

    if (lower === upper) return sorted[lower];
    return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
}

// Mean
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Standard deviation
function std(arr) {
    const avg = mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
}

// Debounce function for input handlers
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Update slider value display
function updateSliderValue(sliderId, displayId, formatter = (v) => v) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);

    if (slider && display) {
        display.textContent = formatter(parseFloat(slider.value));
    }
}

// Chart.js default config
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                color: '#C9D1D9',
                font: { size: 12 }
            }
        }
    },
    scales: {
        x: {
            grid: { color: 'rgba(48, 54, 61, 0.5)' },
            ticks: { color: '#8B949E' }
        },
        y: {
            grid: { color: 'rgba(48, 54, 61, 0.5)' },
            ticks: { color: '#8B949E' }
        }
    }
};

// Solana color palette
const solanaColors = {
    purple: '#9945FF',
    green: '#14F195',
    blue: '#58A6FF',
    orange: '#F0883E',
    red: '#F85149',
    gray: '#8B949E'
};
