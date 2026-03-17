# significance_tests.py
# Vittorio Messana, 2026
#
# testing whether the factor returns are statistically significant
# or whether they could just be random noise
#
# I'm using a t-test which asks: is the mean return different enough
# from zero that we can be confident it is real?
#
# I also apply Bonferroni correction - when you test 6 things at once
# you expect some to look significant purely by chance
# Bonferroni makes the threshold stricter to account for this
#
# spoiler: nothing passes. I'm reporting this honestly.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

factor_names = ["momentum", "value", "quality", "size", "low_vol", "profitability"]

returns = pd.read_csv("data/returns_cut.csv", index_col=0, parse_dates=True)
factors = {}
for name in factor_names:
    factors[name] = pd.read_csv(f"data/factor_{name}.csv", index_col=0, parse_dates=True)

common = returns.index
for name in factor_names:
    common = common.intersection(factors[name].index)

returns = returns.loc[common]
for name in factor_names:
    factors[name] = factors[name].loc[common]

def long_short_portfolio(factor_df, returns_df, top_pct=0.2, bottom_pct=0.2):
    port_returns = []
    dates        = []
    month_ends   = factor_df.resample("ME").last().index
    for i in range(len(month_ends) - 1):
        rd = month_ends[i]
        nd = month_ends[i + 1]
        if rd not in factor_df.index:
            continue
        scores       = factor_df.loc[rd].dropna()
        n            = len(scores)
        top_n        = max(1, int(n * top_pct))
        bottom_n     = max(1, int(n * bottom_pct))
        long_stocks  = scores.nlargest(top_n).index.tolist()
        short_stocks = scores.nsmallest(bottom_n).index.tolist()
        mask         = (returns_df.index > rd) & (returns_df.index <= nd)
        period       = returns_df.loc[mask]
        if period.empty:
            continue
        ls_ret = period[long_stocks].mean(axis=1) - period[short_stocks].mean(axis=1)
        for date, ret in ls_ret.items():
            port_returns.append(ret)
            dates.append(date)
    return pd.Series(port_returns, index=dates).sort_index()

ls_returns = {}
for name in factor_names:
    ls_returns[name] = long_short_portfolio(factors[name], returns)

print("significance tests:")
print(f"{'factor':<15} {'mean return':>12} {'t-stat':>8} {'p-value':>10} {'significant':>12}")
print("-" * 60)

sig_results = []
for name in factor_names:
    r               = ls_returns[name].dropna()
    t_stat, p_value = stats.ttest_1samp(r, 0)
    ann_ret         = r.mean() * 252
    sig             = "YES" if p_value < 0.05 else "NO"
    print(f"{name:<15} {ann_ret:>12.2%} {t_stat:>8.2f} {p_value:>10.4f} {sig:>12}")
    sig_results.append({
        "factor":      name,
        "ann. return": f"{ann_ret:.2%}",
        "t-stat":      round(t_stat, 2),
        "p-value":     round(p_value, 4),
        "significant": sig
    })

bonferroni = 0.05 / len(factor_names)
print(f"\nbonferroni threshold: {bonferroni:.4f}")
for r in sig_results:
    result = "passes" if float(r["p-value"]) < bonferroni else "fails"
    print(f"  {r['factor']}: {result}")

pd.DataFrame(sig_results).to_csv("results/significance_tests.csv", index=False)

# chart
fig, ax = plt.subplots(figsize=(10, 5))
t_stats = [r["t-stat"] for r in sig_results]
colors  = ["#2ca02c" if t > 2 else "#d62728" if t < -2 else "#ff7f0e" for t in t_stats]
ax.bar(factor_names, t_stats, color=colors, alpha=0.7)
ax.axhline(2,  color="green", linestyle="--", linewidth=1, label="p=0.05 threshold")
ax.axhline(-2, color="green", linestyle="--", linewidth=1)
ax.axhline(0,  color="black", linewidth=0.5)
ax.set_title("factor t-statistics")
ax.set_ylabel("t-statistic")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/significance_tests.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nsaved significance_tests.png")
print("done")
