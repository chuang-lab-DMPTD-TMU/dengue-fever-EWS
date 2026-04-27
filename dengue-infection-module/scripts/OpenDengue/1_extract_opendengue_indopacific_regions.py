"""
Extract dengue data for SEARO, WPRO, and EMRO regions between 2000–2025.

Reads the three OpenDengue extract CSVs (National, Temporal, Spatial),
filters to the target WHO regions and date window, then writes the results
to the output directory specified as the first CLI argument (or a default).

Usage:
    python extract_dengue_regions.py [output_dir]
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[3] / "main" / "raw" / "OpenDengue"

NATIONAL_CSV = DATA_DIR / "National_extract_V1_3.csv"
TEMPORAL_CSV = DATA_DIR / "Temporal_extract_V1_3.csv"
SPATIAL_CSV  = DATA_DIR / "Spatial_extract_V1_3.csv"

OUTPUT_DIR = DATA_DIR = Path(__file__).resolve().parents[3] / "main" / "interim" / "OpenDengue"


# ---------------------------------------------------------------------------
# WHO region mapping
# ---------------------------------------------------------------------------
who_region = {
    "AMERICAN SAMOA": "WPRO",
    "CAMBODIA": "WPRO",
    "COTE D'IVOIRE": "AFRO",
    "GUATEMALA": "AMRO",
    "INDIA": "SEARO",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "WPRO",
    "MAYOTTE": "AFRO",
    "MYANMAR": "SEARO",
    "NORTHERN MARIANA ISLANDS": "WPRO",
    "PALAU": "WPRO",
    "SAINT MARTIN": "AMRO",
    "SENEGAL": "AFRO",
    "ANGOLA": "AFRO",
    "BENIN": "AFRO",
    "BRUNEI DARUSSALAM": "WPRO",
    "GHANA": "AFRO",
    "GUADELOUPE": "AMRO",
    "HONDURAS": "AMRO",
    "INDONESIA": "SEARO",
    "MALAYSIA": "WPRO",
    "MALDIVES": "SEARO",
    "MALI": "AFRO",
    "MAURITIUS": "AFRO",
    "NAURU": "WPRO",
    "NEW CALEDONIA": "WPRO",
    "SAINT VINCENT AND THE GRENADINES": "AMRO",
    "SPAIN": "EURO",
    "TONGA": "WPRO",
    "URUGUAY": "AMRO",
    "ANGUILLA": "AMRO",
    "ANTIGUA AND BARBUDA": "AMRO",
    "BAHAMAS": "AMRO",
    "BELIZE": "AMRO",
    "BERMUDA": "AMRO",
    "CAMEROON": "AFRO",
    "CHINA": "WPRO",
    "ETHIOPIA": "AFRO",
    "GUAM": "WPRO",
    "JAMAICA": "AMRO",
    "KIRIBATI": "WPRO",
    "MEXICO": "AMRO",
    "PUERTO RICO": "AMRO",
    "SAINT KITTS AND NEVIS": "AMRO",
    "SAMOA": "WPRO",
    "TAIWAN": "WPRO",
    "TIMOR-LESTE": "SEARO",
    "TOGO": "AFRO",
    "TUVALU": "WPRO",
    "ARGENTINA": "AMRO",
    "ARUBA": "AMRO",
    "BOLIVIA": "AMRO",
    "BONAIRE, SAINT EUSTATIUS AND SABA": "AMRO",
    "CAYMAN ISLANDS": "AMRO",
    "COSTA RICA": "AMRO",
    "CUBA": "AMRO",
    "GRENADA": "AMRO",
    "HAITI": "AMRO",
    "MARSHALL ISLANDS": "WPRO",
    "MARTINIQUE": "AMRO",
    "MAURITANIA": "AFRO",
    "NIGER": "AFRO",
    "NIUE": "WPRO",
    "OMAN": "EMRO",
    "PAKISTAN": "EMRO",
    "PANAMA": "AMRO",
    "PHILIPPINES": "WPRO",
    "PITCAIRN": "WPRO",
    "SAO TOME AND PRINCIPE": "AFRO",
    "SAUDI ARABIA": "EMRO",
    "SUDAN": "EMRO",
    "THAILAND": "SEARO",
    "TURKS AND CAICOS ISLANDS": "AMRO",
    "VENEZUELA": "AMRO",
    "VIRGIN ISLANDS (UK)": "AMRO",
    "WALLIS AND FUTUNA": "WPRO",
    "BRAZIL": "AMRO",
    "CABO VERDE": "AFRO",
    "CENTRAL AFRICAN REPUBLIC": "AFRO",
    "CHILE": "AMRO",
    "COOK ISLANDS": "WPRO",
    "CURACAO": "AMRO",
    "DOMINICAN REPUBLIC": "AMRO",
    "ECUADOR": "AMRO",
    "ERITREA": "AFRO",
    "FIJI": "WPRO",
    "GUYANA": "AMRO",
    "MONTSERRAT": "AMRO",
    "NEPAL": "SEARO",
    "PARAGUAY": "AMRO",
    "PERU": "AMRO",
    "REUNION": "AFRO",
    "SAINT LUCIA": "AMRO",
    "SEYCHELLES": "AFRO",
    "SINGAPORE": "WPRO",
    "SOLOMON ISLANDS": "WPRO",
    "UNITED REPUBLIC OF TANZANIA": "AFRO",
    "VANUATU": "WPRO",
    "AFGHANISTAN": "EMRO",
    "BHUTAN": "SEARO",
    "CHAD": "AFRO",
    "COLOMBIA": "AMRO",
    "FRANCE": "EURO",
    "FRENCH GUIANA": "AMRO",
    "FRENCH POLYNESIA": "WPRO",
    "HONG KONG": "WPRO",
    "JAPAN": "WPRO",
    "MACAU": "WPRO",
    "MICRONESIA (FEDERATED STATES OF)": "WPRO",
    "NICARAGUA": "AMRO",
    "PAPUA NEW GUINEA": "WPRO",
    "SAINT BARTHELEMY": "AMRO",
    "SURINAME": "AMRO",
    "AUSTRALIA": "WPRO",
    "BANGLADESH": "SEARO",
    "BARBADOS": "AMRO",
    "BURKINA FASO": "AFRO",
    "DOMINICA": "AMRO",
    "EL SALVADOR": "AMRO",
    "GUINEA": "AFRO",
    "ITALY": "EURO",
    "KENYA": "AFRO",
    "SINT MAARTEN": "AMRO",
    "SRI LANKA": "SEARO",
    "TOKELAU": "WPRO",
    "TRINIDAD AND TOBAGO": "AMRO",
    "UNITED STATES OF AMERICA": "AMRO",
    "VIET NAM": "WPRO",
    "VIRGIN ISLANDS (US)": "AMRO",
    "YEMEN": "EMRO",
}

TARGET_REGIONS = {"SEARO", "WPRO", "EMRO"}
WINDOW_START = pd.Timestamp("2000-01-01")
WINDOW_END   = pd.Timestamp("2025-12-31")

DTYPE_DICT = {
    "adm_0_name": "object",
    "adm_1_name": "object",
    "adm_2_name": "object",
    "dengue_total": "float64",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Apply region and date-window filters, return filtered copy."""
    original_len = len(df)

    # Region filter
    df = df.copy()
    df["who_region"] = df["adm_0_name"].map(who_region)
    df = df[df["who_region"].isin(TARGET_REGIONS)]
    after_region = len(df)

    # Date filter — only applied when both date columns are present
    date_cols = {"calendar_start_date", "calendar_end_date"}
    if date_cols.issubset(df.columns):
        df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
        df["calendar_end_date"]   = pd.to_datetime(df["calendar_end_date"],   errors="coerce")
        df = df[
            (df["calendar_end_date"]   >= WINDOW_START) &
            (df["calendar_start_date"] <= WINDOW_END)
        ]
    else:
        missing = date_cols - set(df.columns)
        print(f"  [{label}] Warning: date columns {missing} not found — skipping date filter.")

    after_date = len(df)
    print(
        f"  [{label}] {original_len:,} rows → {after_region:,} after region filter "
        f"→ {after_date:,} after date filter"
    )
    return df.reset_index(drop=True)


def load_and_filter(csv_path: Path, label: str) -> pd.DataFrame:
    print(f"Loading {label} ({csv_path.name}) …")
    df = pd.read_csv(csv_path, dtype=DTYPE_DICT, low_memory=False)
    return filter_df(df, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Target regions : {sorted(TARGET_REGIONS)}")
    print(f"Date window    : {WINDOW_START.date()} – {WINDOW_END.date()}")
    print(f"Output dir     : {OUTPUT_DIR}\n")

    datasets = {
        "national": (NATIONAL_CSV, "National"),
        "temporal": (TEMPORAL_CSV, "Temporal"),
        "spatial":  (SPATIAL_CSV,  "Spatial"),
    }

    for key, (csv_path, label) in datasets.items():
        df = load_and_filter(csv_path, label)
        out_path = OUTPUT_DIR / f"{key}_SEARO_WPRO_EMRO_2000_2025.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
