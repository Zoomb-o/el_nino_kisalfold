"""
03_weather_model.py
====================
A simplified energy-balance climate model showing how El Niño SST
anomalies in the tropical Pacific propagate to surface temperature
changes over the Carpathian Basin via the NAO/Rossby wave pathway.

Model structure:
  - Tropical Pacific SST anomaly (ONI) → atmospheric heating anomaly
  - Rossby wave propagation (attenuation factor with latitude)
  - NAO-mediated pressure response over Europe
  - Surface temperature response at Kisalföld latitude (47.7°N)
  - Validated against observed DJF anomalies from Step 2

Output: data/model_output.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


# ══════════════════════════════════════════════════════════════════════════════
# Physical constants and model parameters
# ══════════════════════════════════════════════════════════════════════════════

# Kisalföld coordinates
LAT_KISALFOLD = 47.7   # °N
LON_KISALFOLD = 17.6   # °E

# Rossby wave decay parameter (teleconnection attenuation with latitude)
# Based on Brönnimann (2007): signal decays ~60% from tropics to 50°N
ROSSBY_DECAY = np.exp(-LAT_KISALFOLD / 90.0)   # ≈ 0.59

# NAO coupling coefficient
# El Niño → negative NAO tendency in DJF
# Each +1°C ONI shifts NAO index by approximately −0.4 units
# (Ineson & Scaife 2009; ECMWF 2012)
NAO_COUPLING = -0.40   # NAO units per °C ONI

# Surface temperature sensitivity to NAO over Central Europe
# +1 NAO unit ≈ −0.6°C in DJF over Carpathian Basin
# (negative NAO = warmer Central Europe)
NAO_TEMP_SENSITIVITY = -0.60   # °C per NAO unit

# Stratospheric pathway delay (weeks → months)
STRAT_DELAY_MONTHS = 1.5

# Background warming rate (°C per year, anthropogenic)
BACKGROUND_WARMING = 0.030

# Precipitation sensitivity (mm per °C temperature anomaly, DJF)
PRECIP_SENSITIVITY_DJF = 8.5   # mm per °C (wetter when warmer in winter)
PRECIP_SENSITIVITY_JJA = -12.0  # mm per °C (drier when anticyclonic)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Core teleconnection model
# ══════════════════════════════════════════════════════════════════════════════

def teleconnection_model(oni_val, season="DJF", year=2000):
    """
    Given an ONI value and season, compute the predicted temperature
    and precipitation anomaly at the Kisalföld.

    Pathway:
      ONI  →  Tropical heating anomaly
           →  Rossby wave amplitude (attenuated by latitude)
           →  NAO index shift
           →  Central European surface temperature anomaly
           →  Precipitation anomaly

    Parameters
    ----------
    oni_val : float   Oceanic Niño Index (°C)
    season  : str     DJF | MAM | JJA | SON
    year    : int     Used for background warming calculation

    Returns
    -------
    dict with modelled anomalies
    """

    # Step 1: Tropical heating proportional to SST anomaly
    # Latent heat release ∝ SST anomaly (simplified)
    tropical_heating = oni_val * 1.0   # W/m² proxy, normalised

    # Step 2: Rossby wave amplitude at Kisalföld latitude
    # Attenuates with latitude; stronger in winter (jet stream active)
    season_factor = {"DJF": 1.00, "MAM": 0.55, "JJA": 0.25, "SON": 0.45}
    rossby_amplitude = tropical_heating * ROSSBY_DECAY * season_factor[season]

    # Step 3: NAO response (DJF dominant; weaker other seasons)
    nao_shift = rossby_amplitude * NAO_COUPLING

    # Step 4: Surface temperature response
    # Negative NAO → warmer Central Europe (reduced cold air advection)
    # Positive NAO → cooler (enhanced westerlies bring Atlantic cool air)
    t_anom_nao = nao_shift * NAO_TEMP_SENSITIVITY

    # Step 5: Direct thermal forcing (smaller, secondary pathway)
    # El Niño warms global mean → small direct warming
    t_anom_direct = oni_val * 0.05

    # Step 6: Stratospheric pathway (adds ~30% to signal, delayed)
    # Only active in DJF when polar vortex is relevant
    strat_factor = 0.30 if season == "DJF" else 0.05
    t_anom_strat = t_anom_nao * strat_factor

    # Total temperature anomaly
    t_anom_total = t_anom_nao + t_anom_direct + t_anom_strat

    # Background warming
    background = BACKGROUND_WARMING * (year - 1975)

    # Step 7: Precipitation anomaly
    if season == "DJF":
        p_anom = t_anom_total * PRECIP_SENSITIVITY_DJF
    elif season == "MAM":
        # Van Oldenborgh spring signal: independent of temp pathway
        p_anom = oni_val * 8.0 * season_factor["MAM"]
    elif season == "JJA":
        # Summer drying: anticyclone dominance
        p_anom = oni_val * PRECIP_SENSITIVITY_JJA * season_factor["JJA"]
    else:
        p_anom = oni_val * 3.0 * season_factor["SON"]

    return {
        "oni":              oni_val,
        "season":           season,
        "year":             year,
        "rossby_amplitude": round(rossby_amplitude, 4),
        "nao_shift":        round(nao_shift, 4),
        "t_anom_nao":       round(t_anom_nao, 3),
        "t_anom_direct":    round(t_anom_direct, 3),
        "t_anom_strat":     round(t_anom_strat, 3),
        "t_anom_total":     round(t_anom_total, 3),
        "background_c":     round(background, 3),
        "t_anom_plus_bg":   round(t_anom_total + background, 3),
        "p_anom_mm":        round(p_anom, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Run model over full ONI record and validate
# ══════════════════════════════════════════════════════════════════════════════

def run_and_validate():
    """
    Runs the teleconnection model over the full ONI record (1950–2026)
    and validates predicted DJF temperature anomalies against observed
    station data from Step 2.
    """
    print("\n── Running teleconnection model ────────────────────────")

    oni  = pd.read_csv(os.path.join(DATA_DIR, "enso_oni_index.csv"))
    stat = pd.read_csv(os.path.join(DATA_DIR, "kisalfold_temp_precip.csv"))

    # Build season labels
    season_map = {12:"DJF",1:"DJF",2:"DJF",
                   3:"MAM",4:"MAM",5:"MAM",
                   6:"JJA",7:"JJA",8:"JJA",
                   9:"SON",10:"SON",11:"SON"}

    model_rows = []
    for _, row in oni.iterrows():
        season = season_map[int(row["month"])]
        result = teleconnection_model(
            oni_val=row["oni"],
            season=season,
            year=int(row["year"])
        )
        result["year"]  = int(row["year"])
        result["month"] = int(row["month"])
        model_rows.append(result)

    model_df = pd.DataFrame(model_rows)

    # ── Validation: compare modelled vs observed DJF T anomaly ──────────────
    # Observed: neutral-baseline anomaly from station data
    stat["season"] = stat["month"].map(season_map)
    neutral_djf = stat[(stat["phase"] == "neutral") &
                       (stat["season"] == "DJF")]["temp_c"].mean()

    djf_obs = stat[stat["season"] == "DJF"].groupby("year").agg(
        obs_t=("temp_c", "mean")
    ).reset_index()
    djf_obs["obs_t_anom"] = djf_obs["obs_t"] - neutral_djf

    djf_mod = model_df[model_df["season"] == "DJF"].groupby("year").agg(
        mod_t=("t_anom_total", "mean")
    ).reset_index()

    val = djf_obs.merge(djf_mod, on="year")
    r, p = stats.pearsonr(val["obs_t_anom"], val["mod_t"])
    rmse = np.sqrt(((val["obs_t_anom"] - val["mod_t"]) ** 2).mean())

    print(f"\n  Model validation (DJF temperature anomaly):")
    print(f"    Pearson r  = {r:.3f}  (p = {p:.4f})")
    print(f"    RMSE       = {rmse:.3f} °C")
    print(f"    Bias       = {(val['mod_t'] - val['obs_t_anom']).mean():+.3f} °C")

    # ── Super El Niño scenarios ──────────────────────────────────────────────
    print(f"\n── Super El Niño scenario analysis ────────────────────")
    print(f"\n  {'Event':<10} {'ONI':>5}  "
          f"{'DJF T':>7}  {'DJF P':>7}  "
          f"{'JJA P':>7}  {'MAM P':>7}")
    print("  " + "─" * 52)

    scenarios = [
        ("1982–83", 2.1, 1983),
        ("1997–98", 2.4, 1998),
        ("2015–16", 2.6, 2016),
        ("2023–24", 2.0, 2024),
        ("2026–27*", 2.8, 2027),   # forecast
    ]
    scenario_rows = []
    for name, oni_val, yr in scenarios:
        djf = teleconnection_model(oni_val, "DJF", yr)
        jja = teleconnection_model(oni_val, "JJA", yr)
        mam = teleconnection_model(oni_val, "MAM", yr)
        print(f"  {name:<10} {oni_val:>5.1f}  "
              f"{djf['t_anom_total']:>+7.2f}°C  "
              f"{djf['p_anom_mm']:>+7.1f}mm  "
              f"{jja['p_anom_mm']:>+7.1f}mm  "
              f"{mam['p_anom_mm']:>+7.1f}mm")
        scenario_rows.append({
            "event": name, "oni": oni_val, "year": yr,
            "djf_t_anom": djf["t_anom_total"],
            "djf_p_anom": djf["p_anom_mm"],
            "jja_p_anom": jja["p_anom_mm"],
            "mam_p_anom": mam["p_anom_mm"],
            "djf_t_with_bg": djf["t_anom_plus_bg"],
        })

    print(f"\n  * 2026–27 uses forecast ONI of +2.8°C (IRI/ECMWF May 2026)")
    print(f"    Background warming added: "
          f"+{BACKGROUND_WARMING * (2027 - 1975):.2f}°C above 1975 baseline")

    scenario_df = pd.DataFrame(scenario_rows)
    return model_df, val, scenario_df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Pathway decomposition
# ══════════════════════════════════════════════════════════════════════════════

def pathway_decomposition():
    """
    Shows the relative contribution of each physical pathway
    to the total DJF temperature anomaly at the Kisalföld,
    for a representative Super El Niño (ONI = +2.5°C).
    Used in Figure 3 of the paper.
    """
    print("\n── Pathway decomposition (ONI = +2.5°C, DJF) ──────────")

    result = teleconnection_model(2.5, "DJF", 2000)

    total = result["t_anom_total"]
    paths = {
        "Tropospheric (NAO)":   result["t_anom_nao"],
        "Stratospheric":        result["t_anom_strat"],
        "Direct thermal":       result["t_anom_direct"],
    }

    print(f"\n  {'Pathway':<25} {'Anomaly':>9}  {'Share':>7}")
    print("  " + "─" * 44)
    for name, val in paths.items():
        share = 100 * val / total if total != 0 else 0
        print(f"  {name:<25} {val:>+9.3f}°C  {share:>6.1f}%")
    print(f"  {'─'*44}")
    print(f"  {'Total':<25} {total:>+9.3f}°C  {'100.0':>6}%")

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("El Niño / Kisalföld — Step 3: Weather/Climate Model")
    print("=" * 60)

    model_df, validation_df, scenario_df = run_and_validate()
    pathways = pathway_decomposition()

    # Save outputs
    model_df.to_csv(
        os.path.join(DATA_DIR, "model_output.csv"), index=False)
    validation_df.to_csv(
        os.path.join(DATA_DIR, "model_validation.csv"), index=False)
    scenario_df.to_csv(
        os.path.join(DATA_DIR, "model_scenarios.csv"), index=False)

    print("\n── Files saved ─────────────────────────────────────────")
    print("  data/model_output.csv")
    print("  data/model_validation.csv")
    print("  data/model_scenarios.csv")
    print("\nNext → run src/04_forecast_model.py")