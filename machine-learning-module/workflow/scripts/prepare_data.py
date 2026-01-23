#!/usr/bin/env python3
"""
prepare_data.py
----------------
Loads full dataset, parses the config string (WT1014, KT0726, etc),
generates lag features, constructs training/test sets, and writes them
to disk for downstream pipeline stages.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_config_string(config_input: str):
    """Parse strings like 'WT1014' into validation type, landuse flag, version."""
    config = config_input.strip().lower()
    if len(config) != 6:
        raise ValueError("Config must be 6 characters, e.g. WT1014 or KT0726")

    vflag = config[0]
    lflag = config[1]
    version = config[2:]

    if vflag not in ("k", "w"):
        raise ValueError("Validation flag must be K or W.")
    validation_strategy = "kfold" if vflag == "k" else "walk"

    if lflag not in ("t", "f"):
        raise ValueError("Land-use flag must be T or F")
    use_landuse = (lflag == "t")

    if not version.isdigit():
        raise ValueError("Version must be 4 digits.")

    return validation_strategy, use_landuse, version


def create_lag_features(df, feature_groups):
    """Generate lags 1–3 for each variable."""
    for group in feature_groups:
        for col in group:
            if col in df.columns:
                for lag in (1, 2, 3):
                    df[f"{col}_lag{lag}"] = df.groupby("ID_2")[col].shift(lag)
    return df


def main(args):
    project_root = Path(args.project_root)
    processed = project_root / "data" / "processed" / "INDONESIA" / "monthly_dengue_env_id_class_log.csv"

    validation_strategy, use_landuse, version = parse_config_string(args.config_input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Parsed Config:")
    print(f"  Validation: {validation_strategy}")
    print(f"  Land-use: {use_landuse}")
    print(f"  Version: {version}")

    # ------------------------------------------
    # Load data
    # ------------------------------------------
    df = pd.read_csv(processed)

    df["Risk_Category"] = df["Risk_Category"].replace({
        "Zero": 0, "Low": 1, "High": 2
    }).astype("int32")

    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    df["Incidence_Rate_lag1"] = df.groupby("ID_2")["Incidence_Rate"].shift(1)

    env_vars = [
        'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
        'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum',
        'evaporative_stress_index', 'aridity_index',
        'temperature_2m_ANOM', 'temperature_2m_min_ANOM', 'temperature_2m_max_ANOM',
        'potential_evaporation_sum_ANOM', 'total_evaporation_sum_ANOM', 'precipitation_ANOM'
    ]

    landuse_vars = [
        'Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95',
        'Class_30', 'Class_20', 'Class_10', 'Class_90', 'Class_80'
    ]

    epidemic_vars = ['Incidence_Rate_lag1']

    climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']

    # ------------------------------------------
    # Generate lag features
    # ------------------------------------------
    df = df.sort_values(["YearMonth", "ID_2"])
    df = create_lag_features(df, [env_vars, climate_vars])

    # Build variable list
    variable_columns = []
    for col in (env_vars + climate_vars + epidemic_vars):
        if col in df.columns:
            variable_columns.append(col)

    if use_landuse:
        for col in landuse_vars:
            if col in df.columns:
                variable_columns.append(col)

    # Add lags
    for col in env_vars + climate_vars:
        for lag in (1, 2, 3):
            lag_col = f"{col}_lag{lag}"
            if lag_col in df.columns:
                variable_columns.append(lag_col)

    metadata = ["YearMonth", "ID_2", "Region_Group", "Incidence_Rate"]

    # Split train (year < 2023) and test (=2023)
    df_train = df[df["YearMonth"].dt.year < 2023].dropna(subset=variable_columns + ["Risk_Category"])
    df_test = df[df["YearMonth"].dt.year == 2023].dropna(subset=variable_columns + ["Risk_Category"])

    df_train.to_csv(outdir / "train.csv", index=False)
    df_test.to_csv(outdir / "test.csv", index=False)

    # Save variable list for downstream
    (outdir / "variable_columns.txt").write_text("\n".join(variable_columns))

    # Save metadata, flags
    settings = {
        "validation_strategy": validation_strategy,
        "use_landuse": use_landuse,
        "version": version
    }
    pd.Series(settings).to_json(outdir / "settings.json")

    print("✅ prepare_data.py completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_input", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args())
