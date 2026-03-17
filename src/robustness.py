# robustness.py
# Vittorio Messana, 2026
#
# testing whether the factor results hold up under different assumptions
# I'm varying three things:
# 1. the top/bottom percentile cutoff (10%, 20%, 30%)
# 2. in-sample vs out-of-sample split at 2019
# 3. performance broken down by time period (2010-2014, 2015-2019, 2020-2025)
#
# the decade analysis was the most interesting finding -
# factors behave completely differently across time periods
# value worked in 2010-2014 then died
# profitability was flat for a decade then suddenly became the best factor

import pandas as pd
import numpy as np
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

def sharpe(r):
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

# test 1 - percentile cutoff
print("test 1: percentile cutoff...")
pct_results = []
for pct in [0.1, 0.2, 0.3]:
    row = {"percentile": f"{int(pct*100)}%"}
    for name in factor_names:
        s      = long_short_portfolio(factors[name], returns, top_pct=pct, bottom_pct=pct)
        row[name] = round(sharpe(s), 2)
    pct_results.append(row)
    print(f"  {int(pct*100)}%: {row}")

pd.DataFrame(pct_results).to_csv("results/robustness_percentile.csv", index=False)

# test 2 - in/out of sample
print("\ntest 2: in-sample vs out-of-sample...")
split = "2019-01-01"
for label, before in [("in-sample (2010-2018)", True), ("out-of-sample (2019-2025)", False)]:
    mask = common < split if before else common >= split
    print(f"\n  {label}")
    for name in factor_names:
        s = long_short_portfolio(factors[name].loc[mask], returns.loc[mask])
        print(f"    {name}: sharpe={sharpe(s):.2f}, return={s.mean()*252:.2%}")

# test 3 - decade analysis
print("\ndecade analysis...")
decade_results = []
for label, start, end in [
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2025", "2020-01-01", "2025-12-31")
]:
    row  = {"period": label}
    mask = (common >= start) & (common <= end)
    print(f"\n  {label}")
    for name in factor_names:
        s         = long_short_portfolio(factors[name].loc[mask], returns.loc[mask])
        sh        = round(sharpe(s), 2)
        row[name] = sh
        print(f"    {name}: sharpe={sh:.2f}")
    decade_results.append(row)

pd.DataFrame(decade_results).to_csv("results/decade_analysis.csv", index=False)

# chart - decade sharpes
fig, ax = plt.subplots(figsize=(12, 6))
x      = np.arange(len(factor_names))
width  = 0.25
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, row in enumerate(decade_results):
    sharpes = [row[name] for name in factor_names]
    ax.bar(x + i * width, sharpes, width, label=row["period"], color=colors[i], alpha=0.7)

ax.set_xticks(x + width)
ax.set_xticklabels(factor_names, rotation=15)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("factor sharpe ratios by time period")
ax.set_ylabel("sharpe ratio")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/factor_decade_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nsaved factor_decade_analysis.png")
print("done")
