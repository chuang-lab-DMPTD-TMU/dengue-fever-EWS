# Chuang Lab @ Taipei Medical University, Department of Molecular Parasitology and Tropical Diseases

We focus on modelling the relationship between dengue fever epidemiology and climate change by analyzing geospatial data, infectious reports text data, government recorded infection cases, and Shared Socioeconomic Pathways models. We leverage  data analysis, machine learning, and deep learning methods to better capture how long term climate trends observed in the past might predict the spread of dengue fever spatiotemporally in the near future.

## Modules

(A) Zonal Statistics Module
* Zonal Statistics via Google Earth Engine (Global)
* TCCIP Historical Downscaling Climate Data (Taiwan-only)

(B) Climate Projection Module
* IAMC/IPCC Shared Socioeconomic Pathways Models (Global)
* TCCIP Shared Socioeconomic Pathways Models (Taiwan-only)

(C) Dengue Infection Module
* [OpenDengue](https://opendengue.org/) (Global)
* Additional infection data of finer spatiotemporal resolution (Indonesia, Taiwan)

(D) Machine Learning Module
* Data preprocessing and feature engineering of the previous modules' outputs
* XGBoost, ST-GNN, and other models for spatiotemporal prediction
* Early warning system design that incorporates real-time data


 [Snakemake](https://snakemake.readthedocs.io/en/stable/) is used for data provenance tracking.
