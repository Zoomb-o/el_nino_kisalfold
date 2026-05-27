"""
02_enso_analysis.py
====================
Analyses ENSO teleconnection signals in the Kisalföld climate record.

Produces:
  - Seasonal temperature & precipitation composites by ENSO phase
  - Correlation between ONI and local climate variables
  - Event-by-event anomaly table for the paper
  - Output saved to data/analysis_results.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    oni  = pd.read_csv(os.path.join(DATA_DIR, "enso_oni_index.csv"))
    stat = pd.read_csv(os.path.join(DATA_DIR, "kisalfold_temp_precip.csv"))
    cat  = pd.read_csv(os.path.join(DATA_DIR, "el_nino_event_catalogue.csv"))
    return oni, stat, cat


def season_label(month):
    return {12: "DJF", 1: "DJF", 2: "DJF",
             3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA",
             9: "SON", 10: "SON", 11: "SON"}[month]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Seasonal composites by ENSO phase
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_composites(stat):
    """
    For each season (DJF, MAM, JJA, SON) compute mean temperature and
    precipitation anomaly relative to neutral-phase years, split by
    El Niño / neutral / La Niña.

    Anomaly = phase_mean − neutral_mean
    """
    print("\n── Seasonal composites ─────────────────────────────────")

    # Use only 1960–2024 for stability (avoid sparse early record)
    df = stat[stat["year"] >= 1960].copy()
    df["season"] = df["month"].map(season_label)

    # Climatological monthly baseline (neutral years only)
    neutral = df[df["phase"] == "neutral"]
    baseline_t = neutral.groupby("month")["temp_c"].mean()
    baseline_p = neutral.groupby("month")["precip_mm"].mean()

    df["t_anom"] = df.apply(
        lambda r: r["temp_c"]    - baseline_t[r["month"]], axis=1)
    df["p_anom"] = df.apply(
        lambda r: r["precip_mm"] - baseline_p[r["month"]], axis=1)

    results = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        for phase in ["el_nino", "neutral", "la_nina"]:
            sub = df[(df["season"] == season) & (df["phase"] == phase)]
            if len(sub) < 5:
                continue
            t_an = sub["t_anom"].mean()
            p_an = sub["p_anom"].mean()
            p_pct = 100 * p_an / baseline_p[
                df[df["season"] == season]["month"].iloc[0]]

            # t-test vs neutral
            neutral_sub = df[(df["season"] == season) &
                             (df["phase"] == "neutral")]
            _, p_val_t = stats.ttest_ind(
                sub["t_anom"], neutral_sub["t_anom"])
            _, p_val_p = stats.ttest_ind(
                sub["p_anom"], neutral_sub["p_anom"])

            results.append({
                "season": season, "phase": phase, "n_months": len(sub),
                "t_anom_c":  round(t_an, 2),
                "p_anom_mm": round(p_an, 1),
                "p_anom_pct": round(p_pct, 1),
                "t_pval":    round(p_val_t, 3),
                "p_pval":    round(p_val_p, 3),
                "t_sig":     "**" if p_val_t < 0.01 else
                             "*"  if p_val_t < 0.05 else "",
                "p_sig":     "**" if p_val_p < 0.01 else
                             "*"  if p_val_p < 0.05 else "",
            })

    comp = pd.DataFrame(results)

    # Print table
    print(f"\n{'Season':<6} {'Phase':<10} {'N':>4}  "
          f"{'T anom(°C)':>10}  {'P anom(mm)':>10}  "
          f"{'P anom(%)':>9}  {'T sig':>5}  {'P sig':>5}")
    print("─" * 72)
    for _, r in comp.iterrows():
        print(f"{r.season:<6} {r.phase:<10} {r.n_months:>4}  "
              f"{r.t_anom_c:>+10.2f}  {r.p_anom_mm:>+10.1f}  "
              f"{r.p_anom_pct:>+9.1f}  {r.t_sig:>5}  {r.p_sig:>5}")

    return comp, df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ONI correlation with local climate
# ══════════════════════════════════════════════════════════════════════════════

def oni_correlations(df):
    """
    Pearson correlation between monthly ONI and local temperature /
    precipitation, computed per calendar month and per season.
    Also computes lagged correlations (1–3 month lag) to capture
    the delayed spring precipitation signal.
    """
    print("\n── ONI correlations ────────────────────────────────────")

    results = []
    for month in range(1, 13):
        sub = df[df["month"] == month]
        r_t, p_t = stats.pearsonr(sub["oni"], sub["temp_c"])
        r_p, p_p = stats.pearsonr(sub["oni"], sub["precip_mm"])
        results.append({
            "month": month, "lag": 0,
            "r_temp": round(r_t, 3), "p_temp": round(p_t, 3),
            "r_precip": round(r_p, 3), "p_precip": round(p_p, 3),
        })

    # Lagged: ONI in month M vs precip in month M+lag
    df_sorted = df.sort_values(["year", "month"]).reset_index(drop=True)
    for lag in [1, 2, 3]:
        oni_lag  = df_sorted["oni"].values[:-lag]
        prec_lag = df_sorted["precip_mm"].values[lag:]
        temp_lag = df_sorted["temp_c"].values[lag:]
        r_t, p_t = stats.pearsonr(oni_lag, temp_lag)
        r_p, p_p = stats.pearsonr(oni_lag, prec_lag)
        results.append({
            "month": 0, "lag": lag,
            "r_temp": round(r_t, 3), "p_temp": round(p_t, 3),
            "r_precip": round(r_p, 3), "p_precip": round(p_p, 3),
        })

    corr = pd.DataFrame(results)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"\n{'Month':<5} {'r_temp':>7} {'p_temp':>7}  "
          f"{'r_precip':>8} {'p_precip':>9}")
    print("─" * 45)
    for _, r in corr[corr["lag"] == 0].iterrows():
        sig_t = "*" if r.p_temp   < 0.05 else ""
        sig_p = "*" if r.p_precip < 0.05 else ""
        mn = month_names[int(r.month) - 1]
        print(f"{mn:<5} {r.r_temp:>+7.3f}{sig_t:<1}  "
              f"{r.r_temp:>+7.3f}   "
              f"{r.r_precip:>+8.3f}{sig_p:<1}  "
              f"{r.p_precip:>8.3f}")

    print("\nLagged correlations (ONI → future precip):")
    for _, r in corr[corr["lag"] > 0].iterrows():
        sig = "*" if r.p_precip < 0.05 else ""
        print(f"  Lag {int(r.lag):>1} month(s):  "
              f"r_precip = {r.r_precip:+.3f}{sig}  "
              f"(p = {r.p_precip:.3f})")

    return corr


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Event-by-event anomaly table
# ══════════════════════════════════════════════════════════════════════════════

def event_anomalies(stat, cat):
    """
    For each El Niño event in the catalogue, compute:
      - DJF mean temperature anomaly
      - MAM precipitation anomaly
      - JJA precipitation anomaly (drought signal)
    relative to the long-term neutral baseline.
    """
    print("\n── Event-by-event anomalies ────────────────────────────")

    df = stat[stat["year"] >= 1960].copy()
    df["season"] = df["month"].map(season_label)

    # Neutral baseline per season
    neutral = df[df["phase"] == "neutral"]
    base = neutral.groupby("season").agg(
        t_base=("temp_c", "mean"),
        p_base=("precip_mm", "mean")
    ).reset_index()

    rows = []
    for _, ev in cat.iterrows():
        py = int(ev["peak_year"])

        # DJF spans Dec(py-1)–Feb(py)
        djf = df[((df["year"] == py - 1) & (df["month"] == 12)) |
                 ((df["year"] == py)     & (df["month"].isin([1, 2])))]
        # MAM of peak year
        mam = df[(df["year"] == py) & (df["month"].isin([3, 4, 5]))]
        # JJA of peak year
        jja = df[(df["year"] == py) & (df["month"].isin([6, 7, 8]))]

        t_base = base.loc[base["season"] == "DJF", "t_base"].values[0]
        p_base_mam = base.loc[base["season"] == "MAM", "p_base"].values[0]
        p_base_jja = base.loc[base["season"] == "JJA", "p_base"].values[0]

        djf_t_an  = djf["temp_c"].mean()    - t_base      if len(djf) else np.nan
        mam_p_an  = mam["precip_mm"].mean() - p_base_mam  if len(mam) else np.nan
        jja_p_an  = jja["precip_mm"].mean() - p_base_jja  if len(jja) else np.nan
        jja_p_pct = 100 * jja_p_an / p_base_jja           if len(jja) else np.nan

        rows.append({
            "event":        ev["event"],
            "peak_oni":     ev["peak_oni"],
            "category":     ev["category"],
            "djf_t_anom_c": round(djf_t_an,  2) if not np.isnan(djf_t_an)  else "n/a",
            "mam_p_anom_mm":round(mam_p_an,  1) if not np.isnan(mam_p_an)  else "n/a",
            "jja_p_anom_mm":round(jja_p_an,  1) if not np.isnan(jja_p_an)  else "n/a",
            "jja_p_anom_pct":round(jja_p_pct,1) if not np.isnan(jja_p_pct) else "n/a",
        })

    ev_df = pd.DataFrame(rows)

    print(f"\n{'Event':<10} {'ONI':>5} {'Cat':<8}  "
          f"{'DJF T(°C)':>9}  {'MAM P(mm)':>9}  "
          f"{'JJA P(mm)':>9}  {'JJA P(%)':>8}")
    print("─" * 72)
    for _, r in ev_df.iterrows():
        print(f"{r.event:<10} {r.peak_oni:>5.1f} {r.category:<8}  "
              f"{str(r.djf_t_anom_c):>9}  "
              f"{str(r.mam_p_anom_mm):>9}  "
              f"{str(r.jja_p_anom_mm):>9}  "
              f"{str(r.jja_p_anom_pct):>8}")

    return ev_df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Decadal trend analysis
# ══════════════════════════════════════════════════════════════════════════════

def decadal_trends(stat):
    """
    Computes linear trends in annual mean temperature and total
    precipitation over the Kisalföld record, overall and separated
    by ENSO phase. Used in Section 5 of the paper.
    """
    print("\n── Decadal trends ──────────────────────────────────────")

    annual = stat.groupby("year").agg(
        t_mean=("temp_c", "mean"),
        p_total=("precip_mm", "sum"),
        phase=("phase", lambda x: x.mode()[0])
    ).reset_index()

    # Overall trend
    slope_t, intercept_t, r_t, p_t, _ = stats.linregress(
        annual["year"], annual["t_mean"])
    slope_p, intercept_p, r_p, p_p, _ = stats.linregress(
        annual["year"], annual["p_total"])

    print(f"\n  Temperature trend : {slope_t*10:+.3f} °C/decade  "
          f"(r={r_t:.3f}, p={p_t:.4f})")
    print(f"  Precipitation trend: {slope_p*10:+.1f} mm/decade  "
          f"(r={r_p:.3f}, p={p_p:.4f})")

    # Trend in El Niño years only
    en_years = annual[annual["phase"] == "el_nino"]
    if len(en_years) > 5:
        sl_t, _, r_t2, p_t2, _ = stats.linregress(
            en_years["year"], en_years["t_mean"])
        sl_p, _, r_p2, p_p2, _ = stats.linregress(
            en_years["year"], en_years["p_total"])
        print(f"\n  El Niño years only:")
        print(f"    Temperature trend : {sl_t*10:+.3f} °C/decade  "
              f"(r={r_t2:.3f}, p={p_t2:.4f})")
        print(f"    Precipitation trend: {sl_p*10:+.1f} mm/decade  "
              f"(r={r_p2:.3f}, p={p_p2:.4f})")

    return annual, slope_t, slope_p


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("El Niño / Kisalföld — Step 2: ENSO Analysis")
    print("=" * 60)

    oni, stat, cat = load_data()

    composites, df_annotated = seasonal_composites(stat)
    correlations             = oni_correlations(df_annotated)
    event_df                 = event_anomalies(stat, cat)
    annual_df, slope_t, slope_p = decadal_trends(stat)

    # ── Save results ──────────────────────────────────────────────────────────
    composites.to_csv(
        os.path.join(DATA_DIR, "analysis_composites.csv"), index=False)
    correlations.to_csv(
        os.path.join(DATA_DIR, "analysis_correlations.csv"), index=False)
    event_df.to_csv(
        os.path.join(DATA_DIR, "analysis_events.csv"), index=False)
    annual_df.to_csv(
        os.path.join(DATA_DIR, "analysis_annual.csv"), index=False)

    print("\n── Files saved ─────────────────────────────────────────")
    print("  data/analysis_composites.csv")
    print("  data/analysis_correlations.csv")
    print("  data/analysis_events.csv")
    print("  data/analysis_annual.csv")
    print("\nNext → run src/03_weather_model.py")