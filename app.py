import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Interpolation",
    page_icon=None,
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F4FF; }
    .block-container { padding-top: 2rem; }
    h1 { color: #0D1B3E; font-size: 2rem !important; }
    h3 { color: #0D1B3E; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #2C7BE5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.8rem; color: #64748B; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #0D1B3E; }
    .best-badge {
        background: #DCFCE7; color: #166534;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        display: inline-block; margin-top: 0.3rem;
    }
    .section-header {
        background: #0D1B3E;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Interpolation functions (for-loop only)
# ─────────────────────────────────────────────
def linear_interp(x_pts, y_pts, x):
    for i in range(len(x_pts) - 1):
        if x_pts[i] <= x <= x_pts[i + 1]:
            h = x_pts[i + 1] - x_pts[i]
            return y_pts[i] + ((y_pts[i + 1] - y_pts[i]) / h) * (x - x_pts[i])
    return None

def lagrange_interp(x_pts, y_pts, x):
    n = len(x_pts)
    result = 0.0
    for i in range(n):
        term = float(y_pts[i])
        for j in range(n):
            if j != i:
                term *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        result += term
    return result

def newton_divided_diff(x_pts, y_pts):
    n = len(x_pts)
    coef = [float(v) for v in y_pts]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_pts[i] - x_pts[i - j])
    return coef

def newton_interp(coef, x_pts, x):
    n = len(coef)
    result = coef[0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_pts[i - 1])
        result += coef[i] * prod
    return result

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# Stock Interpolation")
st.markdown(
    "Estimating stock prices on weekends when the market is closed, "
    "using three methods: Linear, Lagrange, and Newton interpolation."
)
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="US: AAPL, TSLA, GOOGL  |  Thai: PTT.BK, CPALL.BK",
    ).upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-10-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-31"))

    window = st.slider(
        "Window Size",
        min_value=3, max_value=10, value=5,
        help="Number of nearby data points used for Lagrange and Newton",
    )

    run = st.button("Calculate", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Suggested tickers**")
    st.markdown("US:  `AAPL`  `TSLA`  `GOOGL`  `MSFT`")
    st.markdown("TH:  `PTT.BK`  `CPALL.BK`  `AOT.BK`")

# ─────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────
if not run:
    st.info("Configure the settings on the left panel, then click Calculate.")
    st.stop()

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with st.spinner(f"Downloading {ticker} price data..."):
    try:
        df = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            st.error(f"No data found for '{ticker}'. Please check the ticker symbol.")
            st.stop()
        df = df[["Close"]].copy()
        df.columns = ["Price"]
        df = df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# ─────────────────────────────────────────────
# Prepare index mapping
# ─────────────────────────────────────────────
all_dates   = pd.date_range(df.index[0], df.index[-1])
trading_set = set(df.index.normalize())
trading_x   = [i for i, d in enumerate(all_dates) if d in trading_set]
trading_y   = df["Price"].values.tolist()
weekend_x   = [i for i, d in enumerate(all_dates) if d.weekday() >= 5]

# Newton uses local window per point (avoids Runge's phenomenon)

# ─────────────────────────────────────────────
# Estimate weekend prices
# ─────────────────────────────────────────────
rows = []
for wx in weekend_x:
    left   = [i for i in trading_x if i <  wx][-window:]
    right  = [i for i in trading_x if i >= wx][:window]
    near   = left + right
    near_y = [trading_y[trading_x.index(i)] for i in near]
    y_lin  = linear_interp(trading_x, trading_y, wx)
    y_lag  = lagrange_interp(near, near_y, wx)
    coef_local = newton_divided_diff(near, near_y)
    y_nwt  = newton_interp(coef_local, near, wx)
    rows.append({
        "Date":     all_dates[wx].date(),
        "Day":      all_dates[wx].strftime("%A"),
        "Linear":   round(y_lin,  4) if y_lin else None,
        "Lagrange": round(y_lag,  4),
        "Newton":   round(y_nwt,  4),
    })

# ─────────────────────────────────────────────
# Calculate MAE
# ─────────────────────────────────────────────
# MAE: Leave-One-Out on actual trading days
# Remove one Friday or Monday, predict it, compare to real price
test_rows = []
for i in range(2, len(trading_x) - 2):
    # Only test on Fridays (last day before weekend)
    actual_date = all_dates[trading_x[i]]
    if actual_date.weekday() != 4:   # 4 = Friday
        continue

    target_x = trading_x[i]
    y_true   = trading_y[i]

    # Build window WITHOUT the target point
    left_pool  = trading_x[:i]
    right_pool = trading_x[i+1:]
    left   = left_pool[-window:]
    right  = right_pool[:window]
    near   = left + right
    near_y = [trading_y[trading_x.index(j)] for j in near]

    y_lin = linear_interp(near, near_y, target_x)
    y_lag = lagrange_interp(near, near_y, target_x)
    coef_local = newton_divided_diff(near, near_y)
    y_nwt = newton_interp(coef_local, near, target_x)

    if y_lin:
        test_rows.append({
            "AE_Linear":   abs(y_lin - y_true),
            "AE_Lagrange": abs(y_lag - y_true),
            "AE_Newton":   abs(y_nwt - y_true),
        })

test_df  = pd.DataFrame(test_rows)
mae_lin  = test_df["AE_Linear"].mean()
mae_lag  = test_df["AE_Lagrange"].mean()
mae_nwt  = test_df["AE_Newton"].mean()
best_mae = min(
    [("Linear", mae_lin), ("Lagrange", mae_lag), ("Newton", mae_nwt)],
    key=lambda t: t[1],
)

# ─────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────
st.markdown(f"### Results for {ticker}  ({start_date} to {end_date})")

c1, c2, c3, c4 = st.columns(4)

def summary_card(col, label, value, sub=None):
    sub_html = f'<div style="font-size:0.78rem;color:#64748B;margin-top:0.2rem;">{sub}</div>' if sub else ""
    col.markdown(f"""
    <div style="background:white;border-radius:10px;padding:1.2rem 1.4rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.08);min-height:90px;">
        <div style="font-size:0.78rem;color:#64748B;margin-bottom:0.3rem;">{label}</div>
        <div style="font-size:1.7rem;font-weight:700;color:#0D1B3E;">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

summary_card(c1, "Trading Days",        f"{len(trading_x)}")
summary_card(c2, "Weekends Estimated",  f"{len(weekend_x)}")
summary_card(c3, "Latest Closing Price",f"{trading_y[-1]:.2f}")
summary_card(c4, "Best Method",         best_mae[0], sub=f"MAE = {best_mae[1]:.4f}")

st.markdown("---")

# ─────────────────────────────────────────────
# MAE cards
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Mean Absolute Error (MAE) — lower is better</div>',
    unsafe_allow_html=True,
)

mc1, mc2, mc3 = st.columns(3)
mae_data = [
    ("Linear",   mae_lin, "#2C7BE5"),
    ("Lagrange", mae_lag, "#F59E0B"),
    ("Newton",   mae_nwt, "#10B981"),
]
for col, (name, val, color) in zip([mc1, mc2, mc3], mae_data):
    badge = '<span class="best-badge">Best</span>' if name == best_mae[0] else ""
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">{name} Interpolation</div>
        <div class="metric-value">{val:.4f}</div>
        {badge}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Graph 1: Weekly Mon-Sun grid
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Weekly Price View — Mon to Sun (actual + estimated)</div>',
    unsafe_allow_html=True,
)

import matplotlib.lines as mlines

mondays = [i for i, d in enumerate(all_dates) if d.weekday() == 0]
n_weeks = len(mondays)
cols_per_row = 4
n_rows = (n_weeks + cols_per_row - 1) // cols_per_row

fig1, axes1 = plt.subplots(
    n_rows, cols_per_row,
    figsize=(cols_per_row * 3.2, n_rows * 3.2),
    sharey=False,
)
fig1.patch.set_facecolor("white")

ax_flat = axes1.flatten() if hasattr(axes1, "flatten") else [axes1]

for wi, mon_idx in enumerate(mondays):
    ax = ax_flat[wi]
    ax.set_facecolor("#F8FAFC")

    wk_days = list(range(mon_idx, min(mon_idx + 7, len(all_dates))))
    wk_dates_local = [all_dates[i] for i in wk_days]

    real_xpos, real_y = [], []
    est_by_pos = {}

    for pos, xi in enumerate(wk_days):
        d = all_dates[xi]
        if xi in trading_x:
            real_xpos.append(pos)
            real_y.append(trading_y[trading_x.index(xi)])
        elif d.weekday() >= 5:
            left   = [j for j in trading_x if j <  xi][-window:]
            right  = [j for j in trading_x if j >= xi][:window]
            near   = left + right
            near_y = [trading_y[trading_x.index(j)] for j in near]
            y_lin  = linear_interp(trading_x, trading_y, xi)
            y_lag  = lagrange_interp(near, near_y, xi)
            coef_local = newton_divided_diff(near, near_y)
            y_nwt  = newton_interp(coef_local, near, xi)
            est_by_pos[pos] = (y_lin, y_lag, y_nwt)

    ax.axvspan(4.5, 6.5, alpha=0.10, color="#94A3B8", zorder=0)

    if real_xpos:
        ax.plot(real_xpos, real_y, color="#0D1B3E", lw=2, zorder=4,
                marker="o", markersize=6,
                markerfacecolor="#0D1B3E", markeredgecolor="white", markeredgewidth=1)

    for pos, (y_lin, y_lag, y_nwt) in est_by_pos.items():
        if y_lin is not None:
            ax.scatter(pos, y_lin,  color="#2C7BE5", s=80, zorder=5,
                       marker="D", edgecolors="white", lw=1)
        ax.scatter(pos, y_lag, color="#F59E0B", s=80, zorder=5,
                   marker="s", edgecolors="white", lw=1)
        ax.scatter(pos, y_nwt, color="#10B981", s=80, zorder=5,
                   marker="^", edgecolors="white", lw=1)

    ax.set_xticks(range(len(wk_days)))
    ax.set_xticklabels([d.strftime("%a") for d in wk_dates_local],
                       fontsize=7.5, color="#64748B")
    ax.set_title(wk_dates_local[0].strftime("%d %b"),
                 fontsize=9, fontweight="bold", color="#0D1B3E", pad=4)
    ax.grid(True, color="#E2E8F0", lw=0.7, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_color("#D1D5DB")
    ax.tick_params(colors="#64748B", labelsize=7)

for wi in range(n_weeks, len(ax_flat)):
    ax_flat[wi].set_visible(False)

h1 = mlines.Line2D([], [], color="#0D1B3E", marker="o", markersize=7,
                    markerfacecolor="#0D1B3E", markeredgecolor="white",
                    lw=2, label="Actual (Mon-Fri)")
h2 = mlines.Line2D([], [], color="#2C7BE5", marker="D", markersize=7,
                    lw=0, markeredgecolor="white", label=f"Linear (MAE={mae_lin:.4f})")
h3 = mlines.Line2D([], [], color="#F59E0B", marker="s", markersize=7,
                    lw=0, markeredgecolor="white", label=f"Lagrange (MAE={mae_lag:.4f})")
h4 = mlines.Line2D([], [], color="#10B981", marker="^", markersize=7,
                    lw=0, markeredgecolor="white", label=f"Newton (MAE={mae_nwt:.4f})")
fig1.legend(handles=[h1, h2, h3, h4], loc="lower center",
            ncol=4, fontsize=9, facecolor="white",
            edgecolor="#D1D5DB", bbox_to_anchor=(0.5, -0.02))
fig1.suptitle(
    f"{ticker} — Mon to Fri: actual price  |  Sat to Sun: estimated",
    fontsize=12, fontweight="bold", color="#0D1B3E", y=1.01,
)
plt.tight_layout()
st.pyplot(fig1)

# ─────────────────────────────────────────────
# Graph 2: MAE comparison
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-header">MAE Comparison</div>',
    unsafe_allow_html=True,
)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig2.patch.set_facecolor("white")

# Bar: overall MAE
ax1.set_facecolor("#F8FAFC")
methods = ["Linear", "Lagrange", "Newton"]
maes    = [mae_lin, mae_lag, mae_nwt]
colors  = ["#2C7BE5", "#F59E0B", "#10B981"]
bars = ax1.bar(methods, maes, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, maes):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(maes) * 0.02,
        f"{val:.4f}",
        ha="center", fontsize=11, fontweight="bold", color="#0D1B3E",
    )
ax1.set_ylabel("MAE", fontsize=10, color="#64748B")
ax1.set_title("Overall MAE per Method", fontsize=11, fontweight="bold", color="#0D1B3E")
ax1.set_ylim(0, max(maes) * 1.3)
ax1.grid(axis="y", color="#E2E8F0", lw=0.8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(colors="#64748B")

# Line: per-day AE
ax2.set_facecolor("#F8FAFC")
ax2.plot(test_df["AE_Linear"].values,   color="#2C7BE5", lw=1.8, label="Linear")
ax2.plot(test_df["AE_Lagrange"].values, color="#F59E0B", lw=1.8, label="Lagrange")
ax2.plot(test_df["AE_Newton"].values,   color="#10B981", lw=1.8, label="Newton")
ax2.set_ylabel("Absolute Error", fontsize=10, color="#64748B")
ax2.set_title("Error per Weekend Day", fontsize=11, fontweight="bold", color="#0D1B3E")
ax2.legend(fontsize=9, facecolor="white", edgecolor="#D1D5DB")
ax2.grid(True, color="#E2E8F0", lw=0.8, alpha=0.8)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(colors="#64748B")

plt.suptitle(
    f"MAE Analysis — {ticker}",
    fontsize=13, fontweight="bold", color="#0D1B3E",
)
plt.tight_layout()
st.pyplot(fig2)

# ─────────────────────────────────────────────
# Data table
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="section-header">Estimated Prices on Weekends</div>',
    unsafe_allow_html=True,
)
result_df = pd.DataFrame(rows)
st.dataframe(result_df, use_container_width=True, height=300)