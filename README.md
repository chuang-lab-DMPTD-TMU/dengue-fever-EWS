# Chuang Lab — Taipei Medical University

Spatiotemporal dengue fever prediction under future climate scenarios, integrating climate projections, reported infection cases, and socioeconomic pathways using ML/DL methods.

Workflows managed with [Snakemake](https://snakemake.readthedocs.io/en/stable/).

## Modules

**climate-projection-module** — Processes AR6 (CMIP6), GWL, and TREAD climate data into merged NetCDF/TSV inputs.

**dengue-infection-module** — Cleans and standardises dengue case data from OpenDengue and Indonesia MoH sources.

**machine-learning-module** — Trains XGBoost, Random Forest, and ST-GAT models on combined climate + infection data.
