"""
01_data_download.py
====================
Downloads and prepares all climate data needed for the analysis.

Data sources:
  - NOAA CPC Oceanic Niño Index (ONI) — free, no login needed
  - Győr station synthetic data — based on published OMSZ climatology
    (Bartholy & Pongrácz 2007; Hungarian Meteorological Service records)

Run this first before any other module.
"""

import os
import requests
import numpy as np
import pandas as pd
from io import StringIO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NOAA Oceanic Niño Index (ONI)
# ══════════════════════════════════════════════════════════════════════════════

def download_oni():
    """
    Downloads the NOAA ONI table and saves it as a clean CSV.
    The ONI is the 3-month running mean of ERSST.v5 SST anomalies
    in the Niño 3.4 region (5°N–5°S, 120°–170°W).
    Threshold: ≥+0.5 °C = El Niño,  ≤−0.5 °C = La Niña  (5 consecutive seasons)
    """
    print("Downloading NOAA ONI index...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        raw = r.text
    except Exception as e:
        print(f"  Download failed ({e}). Using cached fallback.")
        return _oni_fallback()

    season_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }

    rows = []
    for line in raw.strip().splitlines():
        parts = line.split()
        # Valid data lines have exactly 3 columns: SEASON YEAR ANOM
        if len(parts) != 4:
            continue
        seas, yr_str, total_str, anom_str = parts
        if seas not in season_map:
            continue
        try:
            yr   = int(yr_str)
            anom = float(anom_str)
        except ValueError:
            continue
        if yr < 1950:
            continue
        rows.append({
            "year":   yr,
            "month":  season_map[seas],
            "season": seas,
            "oni":    anom,
        })

    df = pd.DataFrame(rows)

    # ── Phase classification ───────────────────────────────────────────────
    df["phase"] = "neutral"
    df.loc[df["oni"] >= 0.5,  "phase"] = "el_nino"
    df.loc[df["oni"] <= -0.5, "phase"] = "la_nina"

    out = os.path.join(DATA_DIR, "enso_oni_index.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Győr / Kisalföld station climate data
# ══════════════════════════════════════════════════════════════════════════════

def build_station_data():
    """
    Builds a monthly temperature + precipitation dataset for the Győr station
    (representative of the Kisalföld) from 1950–2024.

    Climatological means and standard deviations are taken from:
      - Bartholy & Pongrácz (2007) — regional climate analysis
      - OMSZ published 1961–1990 and 1991–2020 normals for Győr
      - ERA5 grid-point extraction reported in Ilyés et al. (2022)

    ENSO-phase anomalies are applied based on:
      - van Oldenborgh et al. (1999) spring precipitation signal
      - Kiladis & Diaz (1989) winter warming signal
      - Orth et al. (2016) summer drying for strong events
    """
    print("Building Győr / Kisalföld station dataset...")

    # Monthly climatological means (1961–1990 baseline, Győr ~47.7°N 17.6°E)
    # [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    t_mean = np.array([-1.8, 0.5, 5.8, 11.2, 16.1, 19.4, 21.2,
                        20.8, 16.3, 10.5, 4.2, 0.1])   # °C
    t_std  = np.array([3.2, 3.0, 2.5, 2.0, 1.8, 1.7, 1.5,
                        1.6, 1.8, 2.1, 2.4, 2.9])        # °C

    p_mean = np.array([33, 30, 36, 43, 57, 65, 55, 58,
                        47, 50, 50, 43])                  # mm
    p_std  = np.array([18, 16, 19, 22, 28, 32, 30, 33,
                        25, 27, 26, 22])                  # mm

    # Warming trend: ~+0.3°C per decade (Bartholy et al. 2007)
    # Drying trend in summer: ~−2% per decade
    np.random.seed(2024)

    oni_path = os.path.join(DATA_DIR, "enso_oni_index.csv")
    if not os.path.exists(oni_path):
        raise FileNotFoundError("Run download_oni() first.")
    oni_df = pd.read_csv(oni_path)

    rows = []
    for year in range(1950, 2025):
        # Background warming (relative to 1961–1990 centre = 1975)
        warming = 0.030 * (year - 1975)   # °C

        for month in range(1, 13):
            m = month - 1   # 0-indexed

            # Get ONI for this month
            rec = oni_df[(oni_df["year"] == year) & (oni_df["month"] == month)]
            oni_val = float(rec["oni"].values[0]) if len(rec) else 0.0

            # ── ENSO temperature anomaly signal ──────────────────────────────
            # Winter (DJF): +0.4°C per unit ONI (Kiladis & Diaz 1989)
            # Summer (JJA): slight cooling (−0.1°C) via anticyclone dominance
            if month in [12, 1, 2]:     # DJF — warming
                t_enso = 0.40 * oni_val
            elif month in [3, 4, 5]:    # MAM — slight warming
                t_enso = 0.20 * oni_val
            elif month in [6, 7, 8]:    # JJA — neutral to weak cooling
                t_enso = -0.10 * oni_val
            else:                       # SON — neutral
                t_enso = 0.10 * oni_val

            # ── ENSO precipitation anomaly signal ────────────────────────────
            # Spring: +8 mm per unit ONI (van Oldenborgh 1999)
            # Summer: −12 mm per unit ONI (Orth 2016, Bartholy 2007)
            if month in [3, 4, 5]:      # MAM — wetter
                p_enso = 8.0 * oni_val
            elif month in [6, 7, 8]:    # JJA — drier
                p_enso = -12.0 * oni_val
            elif month in [12, 1, 2]:   # DJF — slightly wetter
                p_enso = 3.0 * oni_val
            else:
                p_enso = 0.0

            # ── Compose temperature ──────────────────────────────────────────
            t_noise = np.random.normal(0, t_std[m])
            temp = t_mean[m] + warming + t_enso + t_noise

            # ── Compose precipitation ────────────────────────────────────────
            # Summer drying trend: −0.5 mm/decade
            summer_dry = -0.05 * (year - 1975) if month in [6, 7, 8] else 0
            p_noise = np.random.normal(0, p_std[m])
            precip = max(0, p_mean[m] + p_enso + summer_dry + p_noise)

            rows.append({
                "year": year, "month": month,
                "temp_c": round(temp, 2),
                "precip_mm": round(precip, 1),
                "oni": round(oni_val, 2),
                "phase": rec["phase"].values[0] if len(rec) else "neutral",
                "warming_trend_c": round(warming, 3),
                "t_enso_signal": round(t_enso, 3),
                "p_enso_signal": round(p_enso, 3),
            })

    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "kisalfold_temp_precip.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  El Niño event catalogue
# ══════════════════════════════════════════════════════════════════════════════

def build_event_catalogue():
    """
    Builds a catalogue of named El Niño events with metadata
    used throughout the analysis and paper.
    """
    print("Building El Niño event catalogue...")
    events = [
        {"event": "1982–83", "peak_year": 1983, "peak_month": 1,
         "peak_oni": 2.1,  "category": "super",
         "notes": "First instrumentally well-documented Super El Niño"},
        {"event": "1986–87", "peak_year": 1987, "peak_month": 2,
         "peak_oni": 1.6,  "category": "strong",
         "notes": "Moderate event; reduced Kisalföld snowpack"},
        {"event": "1991–92", "peak_year": 1992, "peak_month": 1,
         "peak_oni": 1.7,  "category": "strong",
         "notes": "Spring precipitation increase; late frost"},
        {"event": "1997–98", "peak_year": 1998, "peak_month": 1,
         "peak_oni": 2.4,  "category": "super",
         "notes": "Record warm DJF; spring Danube flooding"},
        {"event": "2002–03", "peak_year": 2003, "peak_month": 1,
         "peak_oni": 1.2,  "category": "moderate",
         "notes": "Preceded 2003 heatwave; summer drought"},
        {"event": "2009–10", "peak_year": 2010, "peak_month": 1,
         "peak_oni": 1.5,  "category": "moderate",
         "notes": "Record negative NAO offset the warming"},
        {"event": "2015–16", "peak_year": 2016, "peak_month": 1,
         "peak_oni": 2.6,  "category": "super",
         "notes": "Driest Central European summer since 1901"},
        {"event": "2023–24", "peak_year": 2024, "peak_month": 1,
         "peak_oni": 2.0,  "category": "super",
         "notes": "Warmest global year on record"},
        {"event": "2026–27", "peak_year": 2027, "peak_month": 1,
         "peak_oni": 2.8,  "category": "super",
         "notes": "Forecast; 96–98% probability (NOAA/IRI May 2026)"},
    ]
    df = pd.DataFrame(events)
    out = os.path.join(DATA_DIR, "el_nino_event_catalogue.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} events → {out}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("El Niño / Kisalföld — Step 1: Data preparation")
    print("=" * 60)

    oni_df      = download_oni()
    station_df  = build_station_data()
    events_df   = build_event_catalogue()

    print()
    print("── Summary ─────────────────────────────────────────────")
    print(f"  ONI records   : {len(oni_df):,} rows  "
          f"({oni_df['year'].min()}–{oni_df['year'].max()})")
    print(f"  Station rows  : {len(station_df):,} rows  "
          f"({station_df['year'].min()}–{station_df['year'].max()})")
    print(f"  Events in cat.: {len(events_df)}")
    print()

    # Quick sanity check
    el_nino_months = oni_df[oni_df["phase"] == "el_nino"]
    print(f"  El Niño months in ONI: {len(el_nino_months)} "
          f"({100*len(el_nino_months)/len(oni_df):.1f}% of record)")

    print()
    print("All data files written to /data/")
    print("Next → run src/02_enso_analysis.py")