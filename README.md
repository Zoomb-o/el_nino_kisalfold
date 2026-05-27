[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20408440.svg)](https://doi.org/10.5281/zenodo.20408440)

# El Niño Teleconnections and Climatic Impacts on the Carpathian Basin and Kisalföld

**Authors:** [Your names]  
**Status:** Preprint — Zenodo  
**DOI:** *(assigned on upload)*

## Overview
This repository contains all code, data, and figures for our research paper
analysing El Niño–Southern Oscillation (ENSO) teleconnection effects on the
Carpathian Basin and Kisalföld (Little Hungarian Plain), including historical
analysis and projections for the 2026–2027 Super El Niño event.

## Structure
el_nino_kisalfold/
├── data/          # Raw and processed climate data
├── src/           # Python analysis modules (run in order)
├── figures/       # Generated publication figures
├── requirements.txt
└── README.md

## How to run
```bash
pip install -r requirements.txt
python src/01_data_download.py   # fetch/prepare data
python src/02_enso_analysis.py   # run analysis
python src/03_weather_model.py   # run climate model
python src/04_forecast_model.py  # run forecast model
python src/05_figures.py         # generate all figures
python src/06_paper_builder.py   # assemble final paper
```

## Data sources
- NOAA Climate Prediction Center — Oceanic Niño Index (ONI)
- Hungarian Meteorological Service (OMSZ) — station records
- ERA5 Reanalysis — ECMWF (via Copernicus CDS)

## Citation
If you use this work, please cite the Zenodo DOI above.