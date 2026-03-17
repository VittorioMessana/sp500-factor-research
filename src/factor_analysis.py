# factor_analysis.py
# Vittorio Messana, 2026
#
# building the six factors and running long/short portfolios
# a long/short portfolio buys the top stocks and shorts the bottom stocks
# this isolates the pure factor return independent of market direction
#
# I'm using top/bottom 20% cutoff for the long and short legs
# rebalancing monthly

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

print("loading data...")
returns = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)
prices  = pd.read_csv("data/stock_prices.csv",  index_col=0, parse_dates=True)
spy     = pd.read_csv("data/spy_returns.csv",    index_col=0, parse_dates=True).squeeze()

# building factors
print("building factors...")

momentum      = ((1 + returns).rolling(252).apply(np.prod, raw=True) / (1 + returns).rolling(21).apply(np.prod, raw=True)) - 1
high_52w      = prices.rolling(252).max()
value         = (1 - (prices / high_52w)).loc[returns.index]
quality       = returns.rolling(63).mean() / (returns.rolling(63).std() + 1e-8)
size          = prices.rank(axis=1, ascending=True).loc[returns.index]
low_vol       = 1 / (returns.rolling(63).std() + 1e-8)
profitability = (1 + returns).rolling(252).apply(np.prod, raw=True) - 1

warmup = 252
factors = {
    "momentum":      momentum.iloc[warmup:],
    "value":         value.iloc[warmup:],
    "quality":       quality.iloc[warmup:],
    "size":          size.iloc[warmup:],
    "low_vol":       low_vol.iloc[warmup:],
    "profitability": profitability.iloc[warmup:]
}
returns_cut = returns.iloc[warmup:]

for name, df in factors.items():
    df.to_csv(f"data/factor_{name}.csv")
returns_cut.to_csv("data/returns_cut.csv")

# aligning dates
common = returns_cut.index
for name in factors:
    common = common.intersection(factors[name].index)
common = common.intersection(spy.index)

returns_cut = returns_cut.loc[common]
spy         = spy.loc[common]
for name in factors:
    factors[name] = factors[name].loc[common]

# long short portfolio function
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

print("computing long/short portfolios...")
factor_names = ["momentum", "value", "quality", "size", "low_vol", "profitability"]
ls_returns   = {}
for name in factor_names:
    ls_returns[name] = long_short_portfolio(factors[name], returns_cut)
    print(f"  {name} done")

# performance metrics
def metrics(r, label):
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "factor":          label,
        "ann. return":     f"{ann_ret:.2%}",
        "ann. volatility": f"{ann_vol:.2%}",
        "sharpe":          f"{sharpe:.2f}",
        "max drawdown":    f"{max_dd:.2%}"
    }

print("\nresults:")
results = []
for name in factor_names:
    m = metrics(ls_returns[name], name)
    results.append(m)
    print(f"  {name}: return={m['ann. return']}, sharpe={m['sharpe']}")

pd.DataFrame(results).to_csv("results/factor_performance.csv", index=False)

# chart 1 - cumulative returns
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
fig, ax = plt.subplots(figsize=(14, 7))
for i, name in enumerate(factor_names):
    cum = (1 + ls_returns[name]).cumprod()
    ax.plot(cum.index, cum.values, label=name, color=colors[i], linewidth=1.2)
ax.set_title("long/short factor cumulative returns")
ax.set_ylabel("cumulative return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/factor_cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.close()

# chart 2 - rolling sharpe
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
for i, name in enumerate(factor_names):
    r              = ls_returns[name]
    rolling_sharpe = (r.rolling(252).mean() * 252) / (r.rolling(252).std() * np.sqrt(252))
    axes[i].plot(rolling_sharpe.index, rolling_sharpe.values, color=colors[i], linewidth=1)
    axes[i].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[i].set_title(f"{name} rolling sharpe")
    axes[i].grid(True, alpha=0.3)
plt.suptitle("rolling 1-year sharpe by factor", fontsize=13)
plt.tight_layout()
plt.savefig("results/rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()

# chart 3 - correlation matrix
ls_df   = pd.DataFrame(ls_returns)
corr    = ls_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax, square=True, linewidths=0.5)
ax.set_title("factor return correlation matrix")
plt.tight_layout()
plt.savefig("results/factor_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nsaved all charts")
print("done")
