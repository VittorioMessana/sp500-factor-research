# S&P 500 Factor Research Lab

An empirical study of six equity factor premia across 50 S&P 500 stocks 
from 2010 to 2025. I wanted to know which factors actually work, which ones 
have decayed, and whether any results hold up statistically.

## The six factors

- **Momentum** — stocks going up tend to keep going up
- **Value** — cheap stocks outperform expensive ones
- **Quality** — consistent stable returns beat erratic ones
- **Size** — smaller stocks outperform larger ones
- **Low-vol** — less volatile stocks outperform
- **Profitability** — profitable companies do better

All factors use price-based proxies since I am working with free data only.
Value and size in particular are approximations — real implementations use 
accounting data like P/E ratios and market capitalisation.

## Long/short portfolio results

Each factor is tested as a long/short portfolio — buy top 20% of stocks 
by factor score, short bottom 20%, rebalance monthly.

| Factor | Ann. Return | Sharpe |
|---|---|---|
| Profitability | 10.65% | 0.53 |
| Momentum | 7.65% | 0.40 |
| Quality | 6.47% | 0.37 |
| Low-vol | -4.60% | -0.22 |
| Value | -4.70% | -0.24 |
| Size | -5.13% | -0.46 |

Profitability, momentum, and quality show positive returns.
Value, size, and low-vol are negative over this period.

## Statistical significance

I ran t-tests on each factor's returns against zero, with Bonferroni 
correction for multiple testing across six factors.

| Factor | T-stat | P-value | Significant |
|---|---|---|---|
| Profitability | 1.52 | 0.1280 | NO |
| Momentum | 1.15 | 0.2521 | NO |
| Quality | 1.06 | 0.2883 | NO |
| Size | -1.33 | 0.1840 | NO |
| Value | -0.68 | 0.4936 | NO |
| Low-vol | -0.65 | 0.5179 | NO |

None of the factors are statistically significant at the 5% level.
After Bonferroni correction the threshold tightens to p < 0.0083 — 
still nothing passes.

This does not mean the factors do not work. It means with 50 stocks 
and monthly rebalancing the sample is too small to prove significance 
beyond reasonable doubt. Larger universes and daily rebalancing would 
give more statistical power.

## Factor behaviour over time

The most interesting finding is how much factor performance varies 
by time period:

| Factor | 2010-2014 | 2015-2019 | 2020-2025 |
|---|---|---|---|
| Momentum | 0.02 | -0.02 | 0.58 |
| Value | 0.62 | -0.10 | -0.42 |
| Quality | -1.05 | 0.74 | 0.35 |
| Size | -1.51 | -0.74 | -0.28 |
| Low-vol | -0.91 | -0.54 | -0.07 |
| Profitability | -0.56 | -0.05 | 0.86 |

Value worked in 2010-2014 then died completely.
Profitability was flat for a decade then became the best factor after 2020.
Momentum was near zero for ten years then suddenly worked.

This matches what academic researchers have published about factor decay 
and regime dependence. Factor premia are not stable — they come and go.

## Out-of-sample results

| Factor | In-sample Sharpe | Out-of-sample Sharpe |
|---|---|---|
| Momentum | 0.10 | 0.60 |
| Profitability | 0.04 | 0.82 |
| Quality | 0.51 | 0.43 |
| Value | -0.12 | -0.40 |
| Size | -1.17 | -0.15 |
| Low-vol | -0.44 | -0.06 |

Momentum and profitability actually got stronger out of sample.
Value and size got worse. Quality was roughly consistent.

## Honest limitations

- Value and size use price proxies not accounting data. This weakens 
  both factors significantly compared to proper implementations.
- 50 stocks is a small universe. Academic factor studies typically use 
  thousands of stocks. My sample size limits statistical power.
- No transaction costs modelled. Monthly rebalancing of a long/short 
  book would incur meaningful costs in practice.
- Survivorship bias is present — all 50 stocks survived to 2025. 
  Companies that went bankrupt are excluded, which flatters returns.
- Factor decay is real. Results from 2010-2025 may not reflect 
  what happens going forward.

## Files

- `src/data_collection.py` — downloads price data for 50 stocks
- `src/factor_analysis.py` — builds factors and long/short portfolios
- `src/significance_tests.py` — t-tests and Bonferroni correction
- `src/robustness.py` — sub-period and percentile sensitivity analysis

## Libraries

yfinance, pandas, numpy, scipy, matplotlib, seaborn

## Author

Vittorio Messana, 2026
