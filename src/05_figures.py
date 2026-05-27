"""
05_figures.py
==============
Generates all publication-quality figures for the paper.

Figures produced:
  Fig 1 — ONI time series 1950–2026 with event annotations
  Fig 2 — Seasonal composite anomalies by ENSO phase (bar chart)
  Fig 3 — Teleconnection pathway decomposition (pie + bar)
  Fig 4 — SARIMA forecast with CI + observed 2026 transition
  Fig 5 — Super El Niño scenario comparison
  Fig 6 — Future projection heatmap (mid/end century)

All saved to figures/ as 300 dpi PNG files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
FIG_DIR    = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# Colour palette (colour-blind friendly)
C_ELNINO  = "#D62728"   # red
C_LANINA  = "#1F77B4"   # blue
C_NEUTRAL = "#7F7F7F"   # grey
C_SUPER   = "#FF7F0E"   # orange (Super El Niño)
C_FORE    = "#2CA02C"   # green (forecast)
C_BG      = "#F7F7F7"


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — ONI time series 1950–2026
# ══════════════════════════════════════════════════════════════════════════════

def fig1_oni_timeseries():
    print("\nFigure 1: ONI time series...")
    oni = pd.read_csv(os.path.join(DATA_DIR, "enso_oni_index.csv"))

    # Decimal year for x-axis
    oni["dec_year"] = oni["year"] + (oni["month"] - 1) / 12

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")

    # Shaded regions
    ax.axhspan( 0.5,  5.0, color=C_ELNINO,  alpha=0.07, zorder=0)
    ax.axhspan(-5.0, -0.5, color=C_LANINA,  alpha=0.07, zorder=0)
    ax.axhline(0,   color="black", linewidth=0.6, zorder=1)
    ax.axhline( 0.5, color=C_ELNINO, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(-0.5, color=C_LANINA, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline( 2.0, color=C_SUPER,  linewidth=0.7, linestyle=":",  alpha=0.7)

    # Fill above/below
    ax.fill_between(oni["dec_year"], oni["oni"], 0,
                    where=oni["oni"] >= 0.5,
                    color=C_ELNINO, alpha=0.55, zorder=2)
    ax.fill_between(oni["dec_year"], oni["oni"], 0,
                    where=oni["oni"] <= -0.5,
                    color=C_LANINA, alpha=0.55, zorder=2)
    ax.fill_between(oni["dec_year"], oni["oni"], 0,
                    where=(oni["oni"] > -0.5) & (oni["oni"] < 0.5),
                    color=C_NEUTRAL, alpha=0.25, zorder=2)
    ax.plot(oni["dec_year"], oni["oni"],
            color="black", linewidth=0.6, zorder=3)

    # Annotate Super El Niño events
    events = [
        (1983.0, 2.1,  "1982–83"),
        (1997.9, 2.4,  "1997–98"),
        (2015.9, 2.6,  "2015–16"),
        (2024.0, 2.0,  "2023–24"),
    ]
    for yr, peak, label in events:
        ax.annotate(label,
                    xy=(yr, peak), xytext=(yr + 0.3, peak + 0.35),
                    fontsize=7.5, color=C_ELNINO,
                    arrowprops=dict(arrowstyle="-", color=C_ELNINO,
                                    lw=0.8, alpha=0.7),
                    ha="left")

    # 2026 transition marker
    oni_2026 = oni[oni["year"] == 2026]
    if len(oni_2026):
        ax.axvline(2026.0, color=C_FORE, linewidth=1.2,
                   linestyle="-.", alpha=0.8, zorder=4)
        ax.text(2026.05, -1.8, "2026\ntransition",
                color=C_FORE, fontsize=7.5, va="top")

    # Legend
    patches = [
        mpatches.Patch(color=C_ELNINO,  alpha=0.6, label="El Niño (ONI ≥ +0.5°C)"),
        mpatches.Patch(color=C_LANINA,  alpha=0.6, label="La Niña (ONI ≤ −0.5°C)"),
        mpatches.Patch(color=C_NEUTRAL, alpha=0.4, label="Neutral"),
        Line2D([0],[0], color=C_SUPER, linestyle=":", lw=1.2,
               label="Super El Niño threshold (+2.0°C)"),
    ]
    ax.legend(handles=patches, loc="upper left", framealpha=0.9,
              ncol=2, fontsize=8)

    ax.set_xlim(1950, 2027)
    ax.set_ylim(-2.5, 3.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("ONI anomaly (°C)")
    ax.set_title(
        "Figure 1. Oceanic Niño Index (ONI) 1950–2026: ENSO phase classification\n"
        "and major El Niño events relevant to the Carpathian Basin study",
        fontweight="bold", pad=8)

    # Source note
    fig.text(0.01, -0.04,
             "Source: NOAA Climate Prediction Center, ONI index (ERSST.v5). "
             "Threshold lines at ±0.5°C and +2.0°C (Super El Niño).",
             fontsize=7, color="gray", ha="left")

    save(fig, "fig1_oni_timeseries.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Seasonal composite anomalies
# ══════════════════════════════════════════════════════════════════════════════

def fig2_seasonal_composites():
    print("Figure 2: Seasonal composites...")
    comp = pd.read_csv(os.path.join(DATA_DIR, "analysis_composites.csv"))

    seasons  = ["DJF", "MAM", "JJA", "SON"]
    phases   = ["el_nino", "la_nina"]
    p_labels = {"el_nino": "El Niño", "la_nina": "La Niña"}
    p_colors = {"el_nino": C_ELNINO,  "la_nina": C_LANINA}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    x     = np.arange(len(seasons))
    width = 0.35

    for ax, var, ylabel, title_suf in zip(
        axes,
        ["t_anom_c", "p_anom_mm"],
        ["Temperature anomaly (°C)", "Precipitation anomaly (mm/month)"],
        ["Temperature", "Precipitation"],
    ):
        for i, phase in enumerate(phases):
            vals = []
            errs = []
            sigs = []
            for s in seasons:
                row = comp[(comp["season"] == s) & (comp["phase"] == phase)]
                if len(row):
                    vals.append(float(row[var].values[0]))
                    # Use significance markers
                    sig_col = "t_sig" if var == "t_anom_c" else "p_sig"
                    sigs.append(str(row[sig_col].values[0]))
                else:
                    vals.append(0)
                    sigs.append("")

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=p_colors[phase], alpha=0.8,
                          label=p_labels[phase],
                          edgecolor="white", linewidth=0.5)

            # Add significance stars
            for bar, sig in zip(bars, sigs):
                if sig:
                    h = bar.get_height()
                    y_pos = h + 0.05 if h >= 0 else h - 0.15
                    ax.text(bar.get_x() + bar.get_width()/2,
                            y_pos, sig, ha="center",
                            fontsize=9, color="black", fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_suf} anomaly by ENSO phase and season\n"
                     f"(relative to neutral years, 1960–2024)",
                     fontweight="bold")
        ax.legend()

        # Shade DJF column
        ax.axvspan(-0.5, 0.5, color="lightyellow",
                   alpha=0.4, zorder=0, label="_nolegend_")

    fig.suptitle(
        "Figure 2. Kisalföld seasonal climate composites by ENSO phase\n"
        "* p < 0.05,  ** p < 0.01  (two-sample t-test vs neutral years)",
        fontsize=10, fontweight="bold", y=1.02)

    fig.text(0.01, -0.05,
             "Source: OMSZ-based station dataset (Győr); NOAA ONI index. "
             "Neutral baseline: 1960–2024.",
             fontsize=7, color="gray")
    plt.tight_layout()
    save(fig, "fig2_seasonal_composites.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Pathway decomposition
# ══════════════════════════════════════════════════════════════════════════════

def fig3_pathway_decomposition():
    print("Figure 3: Pathway decomposition...")

    # Model results for ONI = 2.5 DJF
    pathways = {
        "Tropospheric\n(NAO)":  0.353,
        "Stratospheric\n(PV)":  0.106,
        "Direct\nthermal":      0.125,
    }
    oni_range   = np.linspace(-2, 3, 100)
    decay       = np.exp(-47.7 / 90.0)
    t_nao       = oni_range * decay * -0.40 * -0.60
    t_direct    = oni_range * 0.05
    t_strat     = t_nao * 0.30
    t_total     = t_nao + t_direct + t_strat

    fig = plt.figure(figsize=(12, 4.5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A — pie chart
    ax0 = fig.add_subplot(gs[0])
    colors_pie = ["#E15759", "#4E79A7", "#F28E2B"]
    wedges, texts, autotexts = ax0.pie(
        list(pathways.values()),
        labels=list(pathways.keys()),
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax0.set_title("(a) Pathway share\n(ONI=+2.5°C, DJF)",
                  fontweight="bold", fontsize=10)

    # Panel B — response curves
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(oni_range, t_nao,   color="#E15759", lw=2,
             label="Tropospheric (NAO)")
    ax1.plot(oni_range, t_strat, color="#4E79A7", lw=2,
             label="Stratospheric (PV)")
    ax1.plot(oni_range, t_direct,color="#F28E2B", lw=2,
             label="Direct thermal")
    ax1.plot(oni_range, t_total, color="black",   lw=2.5,
             linestyle="--", label="Total signal")
    ax1.axhline(0, color="gray", lw=0.6)
    ax1.axvline(0, color="gray", lw=0.6)
    ax1.axvspan(0.5, 3.0, color=C_ELNINO, alpha=0.06)
    ax1.axvspan(2.0, 3.0, color=C_SUPER,  alpha=0.10)
    ax1.set_xlabel("ONI (°C)")
    ax1.set_ylabel("DJF T anomaly at Kisalföld (°C)")
    ax1.set_title("(b) Temperature response\nby pathway vs ONI",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)

    # Panel C — season scaling
    ax2 = fig.add_subplot(gs[2])
    seasons_plot = ["DJF", "MAM", "JJA", "SON"]
    season_scales = [1.00, 0.55, 0.25, 0.45]
    bar_colors = [C_ELNINO, "#F28E2B", C_LANINA, C_NEUTRAL]
    t_season = [2.5 * decay * 0.4 * 0.6 * s for s in season_scales]
    bars = ax2.bar(seasons_plot, t_season,
                   color=bar_colors, alpha=0.8,
                   edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Modelled T anomaly (°C)\nfor ONI = +2.5°C")
    ax2.set_title("(c) Seasonal attenuation\nof teleconnection signal",
                  fontweight="bold", fontsize=10)
    for bar, val in zip(bars, t_season):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}°C", ha="center", fontsize=8.5)

    fig.suptitle(
        "Figure 3. Energy-balance teleconnection model: pathway decomposition\n"
        "and seasonal attenuation of the El Niño signal at the Kisalföld (47.7°N)",
        fontsize=10, fontweight="bold", y=1.02)

    fig.text(0.01, -0.05,
             "Model parameters: Rossby decay α=exp(−47.7/90)=0.59; "
             "NAO coupling −0.40 NAO/°C; NAO-temp sensitivity −0.60°C/NAO. "
             "After Brönnimann (2007); Ineson & Scaife (2009).",
             fontsize=7, color="gray")
    save(fig, "fig3_pathway_decomposition.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SARIMA forecast + observed 2026 transition
# ══════════════════════════════════════════════════════════════════════════════

def fig4_sarima_forecast():
    print("Figure 4: SARIMA forecast...")
    oni = pd.read_csv(os.path.join(DATA_DIR, "enso_oni_index.csv"))
    fc  = pd.read_csv(os.path.join(DATA_DIR, "forecast_oni.csv"))

    oni["dec_year"] = oni["year"] + (oni["month"] - 1) / 12
    fc["dec_year"]  = fc["year"]  + (fc["month"]  - 1) / 12

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios":[2.5,1]})
    fig.patch.set_facecolor("white")

    # ── Top panel: ONI + forecast ────────────────────────────────────────────
    obs_recent = oni[oni["year"] >= 2018]
    ax.plot(obs_recent["dec_year"], obs_recent["oni"],
            color="black", lw=1.8, label="Observed ONI", zorder=4)

    # Observed 2026 points (the real transition)
    obs_2026 = oni[oni["year"] == 2026]
    if len(obs_2026):
        ax.plot(obs_2026["dec_year"], obs_2026["oni"],
                color=C_FORE, lw=2.2, marker="o", markersize=5,
                label="Observed 2026 (live NOAA data)", zorder=5)

    # SARIMA forecast
    ax.plot(fc["dec_year"], fc["oni_forecast"],
            color=C_ELNINO, lw=2, linestyle="--",
            label="SARIMA forecast (trained to 2025)", zorder=3)
    ax.fill_between(fc["dec_year"],
                    fc["oni_ci_lower"], fc["oni_ci_upper"],
                    color=C_ELNINO, alpha=0.15,
                    label="95% confidence interval")

    # Operational consensus (NOAA/IRI/ECMWF, May 2026)
    # Manually placed based on published forecast values
    op_years  = np.array([2026.33, 2026.5, 2026.67,
                           2026.83, 2027.0, 2027.17])
    op_oni    = np.array([0.5, 1.0, 1.5, 2.0, 2.2, 2.0])
    ax.plot(op_years, op_oni, color=C_SUPER, lw=2.5,
            linestyle="-.", marker="D", markersize=5,
            label="Operational consensus (NOAA/IRI/ECMWF, May 2026)",
            zorder=6)

    ax.axhline( 0.5, color=C_ELNINO, lw=0.8, linestyle="--", alpha=0.5)
    ax.axhline(-0.5, color=C_LANINA, lw=0.8, linestyle="--", alpha=0.5)
    ax.axhline( 2.0, color=C_SUPER,  lw=0.8, linestyle=":",  alpha=0.6)
    ax.axhspan( 0.5,  3.5, color=C_ELNINO, alpha=0.05)
    ax.axhspan(-3.5, -0.5, color=C_LANINA, alpha=0.05)

    ax.axvline(2026.0, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.text(2026.02, 2.8, "Forecast\nhorizon",
            fontsize=7.5, color="gray", va="top")

    # Annotate divergence
    ax.annotate("SARIMA predicts\nneutral/La Niña\n(historical pattern)",
                xy=(2026.7, -0.45), xytext=(2025.3, -1.8),
                fontsize=7.5, color=C_ELNINO,
                arrowprops=dict(arrowstyle="->", color=C_ELNINO, lw=1))
    ax.annotate("Operational models\nforecast Super El Niño\n(coupled ocean-atm.)",
                xy=(2026.83, 2.0), xytext=(2025.1, 2.6),
                fontsize=7.5, color=C_SUPER,
                arrowprops=dict(arrowstyle="->", color=C_SUPER, lw=1))

    ax.set_xlim(2018, 2028)
    ax.set_ylim(-3, 3.5)
    ax.set_ylabel("ONI anomaly (°C)")
    ax.set_title(
        "Figure 4. SARIMA statistical forecast vs operational model consensus\n"
        "The divergence illustrates why coupled ocean-atmosphere models are\n"
        "essential for capturing rapid ENSO phase transitions",
        fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ── Bottom panel: forecast uncertainty ───────────────────────────────────
    ci_width = fc["oni_ci_upper"] - fc["oni_ci_lower"]
    ax2.fill_between(fc["dec_year"], ci_width,
                     color=C_ELNINO, alpha=0.4,
                     label="95% CI width (growing uncertainty)")
    ax2.plot(fc["dec_year"], ci_width,
             color=C_ELNINO, lw=1.2)
    ax2.set_ylabel("CI width (°C)")
    ax2.set_xlabel("Year")
    ax2.set_xlim(2018, 2028)
    ax2.set_title("(b) Forecast uncertainty growth",
                  fontweight="bold", fontsize=9)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.text(0.01, -0.02,
             "SARIMA(1,0,1)(1,0,1)[12], AIC=−1185.8. "
             "Operational consensus from NOAA CPC/IRI multi-model ensemble "
             "and ECMWF System 5 (May 2026).",
             fontsize=7, color="gray")
    save(fig, "fig4_sarima_forecast.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Super El Niño scenario comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig5_scenario_comparison():
    print("Figure 5: Scenario comparison...")
    sc = pd.read_csv(os.path.join(DATA_DIR, "model_scenarios.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    events    = sc["event"].tolist()
    colors    = [C_ELNINO if "*" not in e else C_SUPER for e in events]
    colors[-1] = C_FORE  # 2026-27 forecast in green

    x = np.arange(len(events))

    # Panel A — DJF temperature anomaly
    ax = axes[0]
    bars = ax.bar(x, sc["djf_t_anom"], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.bar(x, sc["djf_t_with_bg"] - sc["djf_t_anom"],
           bottom=sc["djf_t_anom"],
           color=colors, alpha=0.35, edgecolor="white",
           hatch="///", label="+ background warming")
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("DJF temperature anomaly (°C)")
    ax.set_title("(a) Winter warming signal\n(DJF, vs neutral baseline)",
                 fontweight="bold", fontsize=9.5)
    ax.axhline(0, color="black", lw=0.7)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, sc["djf_t_anom"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"+{val:.2f}°C", ha="center", fontsize=8)

    # Panel B — JJA precipitation anomaly
    ax = axes[1]
    bars = ax.bar(x, sc["jja_p_anom"], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("JJA precipitation anomaly (mm/month)")
    ax.set_title("(b) Summer drying signal\n(JJA, vs neutral baseline)",
                 fontweight="bold", fontsize=9.5)
    ax.axhline(0, color="black", lw=0.7)
    for bar, val in zip(bars, sc["jja_p_anom"]):
        ypos = val - 0.4 if val < 0 else val + 0.1
        ax.text(bar.get_x() + bar.get_width()/2,
                ypos, f"{val:.1f}mm",
                ha="center", fontsize=8, va="top" if val < 0 else "bottom")

    # Panel C — MAM precipitation anomaly
    ax = axes[2]
    bars = ax.bar(x, sc["mam_p_anom"], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("MAM precipitation anomaly (mm/month)")
    ax.set_title("(c) Spring precipitation signal\n(MAM, van Oldenborgh pathway)",
                 fontweight="bold", fontsize=9.5)
    ax.axhline(0, color="black", lw=0.7)
    for bar, val in zip(bars, sc["mam_p_anom"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"+{val:.1f}mm", ha="center", fontsize=8)

    # Legend patches
    legend_el = [
        mpatches.Patch(color=C_ELNINO, alpha=0.85, label="Historical Super El Niño"),
        mpatches.Patch(color=C_FORE,   alpha=0.85, label="2026–27 forecast"),
    ]
    fig.legend(handles=legend_el, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Figure 5. Kisalföld climate anomalies for historical and forecast "
        "Super El Niño events\n"
        "Teleconnection model output; 2026–27 uses forecast ONI = +2.8°C "
        "(IRI/ECMWF, May 2026)",
        fontsize=10, fontweight="bold")

    plt.tight_layout()
    save(fig, "fig5_scenario_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Future projection heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig6_projection_heatmap():
    print("Figure 6: Projection heatmap...")

    # Data from Table 2 of the paper (EURO-CORDEX synthesis)
    periods   = ["Baseline\n1961–1990", "Mid-century\n2026–2050",
                 "End-century\n2071–2100"]
    variables = [
        "Winter T anomaly\nvs neutral (°C)",
        "Summer precip\ndeficit (%)",
        "Drought freq.\n(events/decade)",
        "Extreme heat days\n(>35°C/yr)",
        "Spring rainfall\nincrease (mm)",
    ]

    # Mid-point of projected ranges
    data = np.array([
        [1.00,  1.70,  2.85],   # Winter T anomaly
        [15.0,  24.0,  35.0],   # Summer precip deficit (absolute %)
        [1.75,  3.00,  5.00],   # Drought frequency
        [4.5,   11.0,  21.5],   # Heat days
        [17.5,  26.5,  37.5],   # Spring rainfall
    ])

    # Normalise each row 0–1 for colour mapping
    data_norm = np.zeros_like(data)
    for i in range(len(variables)):
        mn, mx = data[i].min(), data[i].max()
        data_norm[i] = (data[i] - mn) / (mx - mn) if mx > mn else data[i]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    cmap = plt.cm.RdYlBu_r
    im   = ax.imshow(data_norm, cmap=cmap, aspect="auto",
                     vmin=0, vmax=1)

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, fontsize=9.5)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables, fontsize=9)

    # Cell annotations with actual values
    units = ["°C", "%", "/dec", "days", "mm"]
    for i in range(len(variables)):
        for j in range(len(periods)):
            val  = data[i, j]
            norm = data_norm[i, j]
            txt  = f"{val:.1f}{units[i]}"
            col  = "white" if norm > 0.65 or norm < 0.2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=col)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Relative intensity\n(row-normalised)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Lower", "Mid", "Higher"])

    ax.set_title(
        "Figure 6. Projected ENSO-related climate changes in the Kisalföld\n"
        "under high-emission scenario (RCP8.5/SSP5-8.5) during El Niño years",
        fontweight="bold", pad=10)

    # Vertical lines separating periods
    for j in [0.5, 1.5]:
        ax.axvline(j, color="white", lw=2)

    fig.text(0.01, -0.04,
             "Sources: EURO-CORDEX; Pongrácz et al. (2011, 2014); "
             "Kis et al. (2017); Bartholy et al. (2013); IPCC AR6 (2021). "
             "Values are mid-points of published projection ranges.",
             fontsize=7, color="gray")

    plt.tight_layout()
    save(fig, "fig6_projection_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("El Niño / Kisalföld — Step 5: Generate Figures")
    print("=" * 60)

    fig1_oni_timeseries()
    fig2_seasonal_composites()
    fig3_pathway_decomposition()
    fig4_sarima_forecast()
    fig5_scenario_comparison()
    fig6_projection_heatmap()

    print("\n── All figures saved ───────────────────────────────────")
    print("  figures/fig1_oni_timeseries.png")
    print("  figures/fig2_seasonal_composites.png")
    print("  figures/fig3_pathway_decomposition.png")
    print("  figures/fig4_sarima_forecast.png")
    print("  figures/fig5_scenario_comparison.png")
    print("  figures/fig6_projection_heatmap.png")
    print("\nNext → run src/06_paper_builder.py")