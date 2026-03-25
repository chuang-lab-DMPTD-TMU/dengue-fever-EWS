from pathlib import Path

import dask.dataframe as dd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_RAW_DIR = PROJECT_ROOT / "main" / "raw" / "global"
OUTPUT_DIR = GLOBAL_RAW_DIR / "who_regions_2000_2025"

INPUT_FILES = {
    "national": GLOBAL_RAW_DIR / "National_extract_V1_3.csv",
    "spatial": GLOBAL_RAW_DIR / "Spatial_extract_V1_3.csv",
    "temporal": GLOBAL_RAW_DIR / "Temporal_extract_V1_3.csv",
}

TARGET_REGIONS = {"SEARO", "WPRO", "EMRO"}
START_YEAR = 2000
END_YEAR = 2025
VALID_T_RES = {"year", "month", "week"}

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
    "YEMEN": "EMRO"
}


def extract_by_region_and_year(input_path: Path, output_path: Path) -> None:
    dtype_dict = {
        "adm_0_name": "object",
        "adm_1_name": "object",
        "adm_2_name": "object",
        "Year": "object",
    }

    ddf = dd.read_csv(str(input_path), dtype=dtype_dict, assume_missing=True)

    ddf["Year"] = ddf["Year"].map_partitions(
        pd.to_numeric,
        errors="coerce",
        meta=("Year", "float64"),
    )

    normalized_country = ddf["adm_0_name"].astype(str).str.upper().str.strip()
    ddf["WHO_region"] = normalized_country.map_partitions(
        lambda s: s.map(who_region),
        meta=("WHO_region", "object"),
    )

    filtered = ddf[
        (ddf["Year"] >= START_YEAR)
        & (ddf["Year"] <= END_YEAR)
        & (ddf["WHO_region"].isin(TARGET_REGIONS))
    ]

    filtered.to_csv(str(output_path), single_file=True, index=False)


def filter_national_by_year_coverage(
    national_csv_path: Path,
    output_path: Path,
    max_missing_ratio: float = 0.5,
) -> None:
    df = pd.read_csv(national_csv_path, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["T_res"] = df["T_res"].astype(str).str.lower().str.strip()

    valid = df[
        (df["Year"] >= START_YEAR)
        & (df["Year"] <= END_YEAR)
        & (df["T_res"].isin(VALID_T_RES))
    ].copy()

    valid["Year"] = valid["Year"].astype(int)
    total_years = END_YEAR - START_YEAR + 1

    years_present = valid.groupby("adm_0_name")["Year"].nunique()
    missing_ratio = (total_years - years_present) / total_years
    countries_to_keep = missing_ratio[missing_ratio <= max_missing_ratio].index

    filtered_df = df[df["adm_0_name"].isin(countries_to_keep)].copy()
    filtered_df.to_csv(output_path, index=False)

    print(
        f"Saved coverage-filtered national file: {output_path} "
        f"(kept {len(countries_to_keep)} countries, removed {df['adm_0_name'].nunique() - len(countries_to_keep)} countries)"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, input_path in INPUT_FILES.items():
        output_path = OUTPUT_DIR / f"{dataset_name}_SEARO_WPRO_EMRO_{START_YEAR}_{END_YEAR}.csv"
        extract_by_region_and_year(input_path, output_path)
        print(f"Saved: {output_path}")

    national_output = OUTPUT_DIR / f"national_SEARO_WPRO_EMRO_{START_YEAR}_{END_YEAR}.csv"
    national_coverage_output = OUTPUT_DIR / (
        f"national_SEARO_WPRO_EMRO_{START_YEAR}_{END_YEAR}_missing_le50pct.csv"
    )
    filter_national_by_year_coverage(national_output, national_coverage_output)


if __name__ == "__main__":
    main()