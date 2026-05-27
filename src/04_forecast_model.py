"""
04_forecast_model.py
=====================
Statistical forecast model for El Niño impacts on the Kisalföld.

Two approaches:
  A) SARIMA — seasonal ARIMA on the ONI index to forecast future values
  B) Linear regression — ONI → Kisalföld seasonal climate response
     validated with leave-one-out cross-validation on historical events

Produces:
  - ONI forecast through 2027-12
  - Kisalföld seasonal anomaly forecast for 2026–2027
  - Skill scores and confidence intervals
  - data/forecast_output.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    oni  = pd.read_csv(os.path.join(DATA_DIR, "enso_oni_index.csv"))
    stat = pd.read_csv(os.path.join(DATA_DIR, "kisalfold_temp_precip.csv"))
    return oni, stat


def season_label(month):
    return {12:"DJF",1:"DJF",2:"DJF",
             3:"MAM",4:"MAM",5:"MAM",
             6:"JJA",7:"JJA",8:"JJA",
             9:"SON",10:"SON",11:"SON"}[month]


# ══════════════════════════════════════════════════════════════════════════════
# A.  SARIMA forecast of ONI index
# ══════════════════════════════════════════════════════════════════════════════

def sarima_oni_forecast(oni, n_ahead=24):
    """
    Fits a SARIMA(1,0,1)(1,0,1)[12] model to the ONI time series
    and forecasts n_ahead months ahead.

    SARIMA is appropriate because:
      - ONI has strong seasonal autocorrelation (period=12)
      - It is stationary (confirmed by ADF test below)
      - It captures the multi-year ENSO cycle (~48 months)
    """
    print("\n── A. SARIMA ONI forecast ───────────────────────────────")

    # Use data up to end of 2025 for training
    # (2026 partial data used for validation)
    train = oni[oni["year"] <= 2025]["oni"].values

    # ADF stationarity test
    adf_stat, adf_p, *_ = adfuller(train)
    print(f"\n  ADF stationarity test: stat={adf_stat:.3f}, p={adf_p:.4f}")
    print(f"  Series is {'stationary' if adf_p < 0.05 else 'non-stationary'} "
          f"(differencing {'not ' if adf_p < 0.05 else ''}needed)")

    # Fit SARIMA(1,0,1)(1,0,1)[12]
    print("  Fitting SARIMA(1,0,1)(1,0,1)[12]...")
    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    print(f"  AIC = {fit.aic:.1f}   BIC = {fit.bic:.1f}")

    # Forecast
    forecast_obj = fit.get_forecast(steps=n_ahead)
    fc_mean = forecast_obj.predicted_mean
    fc_ci   = forecast_obj.conf_int(alpha=0.05)   # 95% CI

    # Build date index
    last_year  = int(oni[oni["year"] <= 2025]["year"].max())
    last_month = int(oni[oni["year"] == last_year]["month"].max())

    dates, years, months = [], [], []
    y, m = last_year, last_month
    for _ in range(n_ahead):
        m += 1
        if m > 12:
            m = 1
            y += 1
        dates.append(f"{y}-{m:02d}")
        years.append(y)
        months.append(m)

    fc_df = pd.DataFrame({
        "date":   dates,
        "year":   years,
        "month":  months,
        "oni_forecast":  np.round(fc_mean, 3),
        "oni_ci_lower":  np.round(fc_ci[:, 0], 3),
        "oni_ci_upper":  np.round(fc_ci[:, 1], 3),
        "phase_forecast": ["el_nino" if v >= 0.5 else
                           "la_nina" if v <= -0.5 else
                           "neutral" for v in fc_mean],
    })

    # Print forecast table for 2026–2027
    print(f"\n  {'Date':<9} {'ONI fc':>7}  {'CI lower':>9}  "
          f"{'CI upper':>9}  {'Phase':<10}")
    print("  " + "─" * 50)
    for _, r in fc_df.iterrows():
        marker = " ◄" if r.oni_forecast >= 2.0 else ""
        print(f"  {r.date:<9} {r.oni_forecast:>+7.3f}  "
              f"{r.oni_ci_lower:>+9.3f}  "
              f"{r.oni_ci_upper:>+9.3f}  "
              f"{r.phase_forecast:<10}{marker}")

    # In-sample skill (last 5 years held out)
    train_short = train[:-60]
    model_s = SARIMAX(train_short, order=(1,0,1),
                      seasonal_order=(1,0,1,12),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
    fit_s = model_s.fit(disp=False)
    fc_s  = fit_s.get_forecast(steps=60).predicted_mean
    obs_s = train[-60:]
    r_skill, _ = stats.pearsonr(obs_s, fc_s)
    rmse_skill = np.sqrt(np.mean((obs_s - fc_s)**2))
    print(f"\n  Hindcast skill (last 5 yrs held out):")
    print(f"    r = {r_skill:.3f}   RMSE = {rmse_skill:.3f}°C")

    return fc_df, fit


# ══════════════════════════════════════════════════════════════════════════════
# B.  Regression forecast: ONI → Kisalföld seasonal anomalies
# ══════════════════════════════════════════════════════════════════════════════

def regression_forecast(oni, stat, fc_df):
    """
    For each season, fits a linear regression:
        climate_anomaly = α + β × ONI + ε

    Then applies it to the SARIMA-forecast ONI values to predict
    Kisalföld temperature and precipitation anomalies in 2026–2027.

    Leave-one-event-out cross-validation is used to estimate skill.
    """
    print("\n── B. Regression forecast: ONI → Kisalföld ─────────────")

    season_map = {12:"DJF",1:"DJF",2:"DJF",
                   3:"MAM",4:"MAM",5:"MAM",
                   6:"JJA",7:"JJA",8:"JJA",
                   9:"SON",10:"SON",11:"SON"}

    df = stat.copy()
    df["season"] = df["month"].map(season_map)

    # Neutral baseline
    neutral = df[df["phase"] == "neutral"]
    base_t = neutral.groupby("month")["temp_c"].mean()
    base_p = neutral.groupby("month")["precip_mm"].mean()
    df["t_anom"] = df.apply(lambda r: r["temp_c"] - base_t[r["month"]], axis=1)
    df["p_anom"] = df.apply(lambda r: r["precip_mm"] - base_p[r["month"]], axis=1)

    # Annual season means
    seasonal = df.groupby(["year", "season"]).agg(
        t_anom=("t_anom", "mean"),
        p_anom=("p_anom", "mean"),
        oni=("oni", "mean"),
    ).reset_index()

    results = []
    print(f"\n  {'Season':<6} {'β_temp':>8} {'r_temp':>7} "
          f"{'β_precip':>9} {'r_precip':>9} {'CV_r_t':>7} {'CV_r_p':>7}")
    print("  " + "─" * 62)

    reg_models = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        sub = seasonal[seasonal["season"] == season].dropna()
        if len(sub) < 10:
            continue

        # OLS regression
        sl_t, ic_t, r_t, p_t, se_t = stats.linregress(sub["oni"], sub["t_anom"])
        sl_p, ic_p, r_p, p_p, se_p = stats.linregress(sub["oni"], sub["p_anom"])

        # Leave-one-out cross-validation
        loo_t, loo_p = [], []
        for i in range(len(sub)):
            mask = np.ones(len(sub), bool)
            mask[i] = False
            tr = sub.iloc[mask]
            te = sub.iloc[i]
            sl_t_loo, ic_t_loo, *_ = stats.linregress(tr["oni"], tr["t_anom"])
            sl_p_loo, ic_p_loo, *_ = stats.linregress(tr["oni"], tr["p_anom"])
            loo_t.append(sl_t_loo * te["oni"] + ic_t_loo)
            loo_p.append(sl_p_loo * te["oni"] + ic_p_loo)

        cv_r_t, _ = stats.pearsonr(sub["t_anom"], loo_t)
        cv_r_p, _ = stats.pearsonr(sub["p_anom"], loo_p)

        reg_models[season] = {
            "sl_t": sl_t, "ic_t": ic_t, "se_t": se_t,
            "sl_p": sl_p, "ic_p": ic_p, "se_p": se_p,
            "r_t": r_t,   "r_p": r_p,
            "cv_r_t": cv_r_t, "cv_r_p": cv_r_p,
        }

        sig_t = "**" if p_t < 0.01 else "*" if p_t < 0.05 else ""
        sig_p = "**" if p_p < 0.01 else "*" if p_p < 0.05 else ""
        print(f"  {season:<6} {sl_t:>+8.3f}{sig_t:<2} {r_t:>+6.3f} "
              f"{sl_p:>+9.3f}{sig_p:<2} {r_p:>+8.3f} "
              f"{cv_r_t:>+7.3f} {cv_r_p:>+7.3f}")

        results.append({
            "season": season,
            "beta_temp": round(sl_t, 4), "r_temp": round(r_t, 3),
            "beta_precip": round(sl_p, 4), "r_precip": round(r_p, 3),
            "cv_r_temp": round(cv_r_t, 3), "cv_r_precip": round(cv_r_p, 3),
        })

    # ── Apply to SARIMA forecast ──────────────────────────────────────────────
    print(f"\n── 2026–2027 Kisalföld forecast ────────────────────────")
    print(f"\n  {'Period':<14} {'ONI fc':>7}  "
          f"{'T anom':>7}  {'P anom':>8}  {'Phase':<10}")
    print("  " + "─" * 55)

    forecast_rows = []
    target_periods = [
        ("JJA 2026",  2026, "JJA"),
        ("DJF 26/27", 2027, "DJF"),
        ("MAM 2027",  2027, "MAM"),
        ("JJA 2027",  2027, "JJA"),
    ]

    for label, yr, seas in target_periods:
        # Get mean forecast ONI for that season/year
        season_months = {"DJF":[12,1,2],"MAM":[3,4,5],
                         "JJA":[6,7,8],"SON":[9,10,11]}
        fc_sub = fc_df[(fc_df["year"] == yr) &
                       (fc_df["month"].isin(season_months[seas]))]
        if len(fc_sub) == 0:
            # DJF 2026/27 spans Dec2026 + Jan/Feb2027
            if seas == "DJF":
                fc_sub = fc_df[
                    ((fc_df["year"] == yr-1) & (fc_df["month"] == 12)) |
                    ((fc_df["year"] == yr)   & (fc_df["month"].isin([1,2])))
                ]

        if len(fc_sub) == 0:
            continue

        oni_fc   = fc_sub["oni_forecast"].mean()
        oni_lo   = fc_sub["oni_ci_lower"].mean()
        oni_hi   = fc_sub["oni_ci_upper"].mean()

        if seas not in reg_models:
            continue
        m = reg_models[seas]

        t_fc  = m["sl_t"] * oni_fc + m["ic_t"]
        p_fc  = m["sl_p"] * oni_fc + m["ic_p"]

        # 95% prediction interval (regression uncertainty)
        n = 65   # approx sample size
        t_pi = 1.96 * m["se_t"] * np.sqrt(1 + 1/n)
        p_pi = 1.96 * m["se_p"] * np.sqrt(1 + 1/n)

        phase = ("El Niño" if oni_fc >= 0.5 else
                 "La Niña" if oni_fc <= -0.5 else "Neutral")

        print(f"  {label:<14} {oni_fc:>+7.3f}  "
              f"{t_fc:>+7.3f}°C  {p_fc:>+8.1f}mm  {phase:<10}")
        print(f"  {'':14} {'95% CI:':>8}  "
              f"T:[{t_fc-t_pi:+.2f},{t_fc+t_pi:+.2f}]  "
              f"P:[{p_fc-p_pi:+.1f},{p_fc+p_pi:+.1f}]")

        forecast_rows.append({
            "period": label, "season": seas, "year": yr,
            "oni_forecast": round(oni_fc, 3),
            "t_anom_forecast": round(t_fc, 3),
            "t_anom_pi_lower": round(t_fc - t_pi, 3),
            "t_anom_pi_upper": round(t_fc + t_pi, 3),
            "p_anom_forecast": round(p_fc, 1),
            "p_anom_pi_lower": round(p_fc - p_pi, 1),
            "p_anom_pi_upper": round(p_fc + p_pi, 1),
            "phase": phase,
        })

    return pd.DataFrame(results), pd.DataFrame(forecast_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("El Niño / Kisalföld — Step 4: Forecast Model")
    print("=" * 60)

    oni, stat = load_data()

    fc_df, sarima_fit       = sarima_oni_forecast(oni)
    regression_df, fc_out   = regression_forecast(oni, stat, fc_df)

    # Save
    fc_df.to_csv(
        os.path.join(DATA_DIR, "forecast_oni.csv"), index=False)
    regression_df.to_csv(
        os.path.join(DATA_DIR, "forecast_regression_skill.csv"), index=False)
    fc_out.to_csv(
        os.path.join(DATA_DIR, "forecast_kisalfold_2027.csv"), index=False)

    print("\n── Files saved ─────────────────────────────────────────")
    print("  data/forecast_oni.csv")
    print("  data/forecast_regression_skill.csv")
    print("  data/forecast_kisalfold_2027.csv")
    print("\nNext → run src/05_figures.py")