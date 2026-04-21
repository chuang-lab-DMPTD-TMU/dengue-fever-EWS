import geopandas as gpd
import pandas as pd

SHP_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/external/FAO_GAUL_shp"
)
CSV_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/dengue-infection-module/main/interim/OpenDengue"
    "/filtered_sea_2011_2018_SLVC_imputed.csv"
)
OUT_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/dengue-infection-module/main/external/geoparquet"
    "/gaul_2024_sea_filtered.parquet"
)

# CSV adm_0_name (uppercase) → GAUL gaul0_name
# Only entries that differ from plain .title() are listed.
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM": "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Lao People's Democratic Republic",
    "VIET NAM": "Viet Nam",
}

# Manual adm_1 name mappings where CSV name does not match GAUL gaul1_name
# (or gaul2_name for Myanmar) case-insensitively.
# Key: (gaul0_name, csv adm_1_name uppercase) → dissolved-L1 'name' value
MANUAL_ADM1_MAP = {
    # Brunei — CSV appends " DISTRICT"
    ("Brunei Darussalam", "BELAIT DISTRICT"): "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"): "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"): "Tutong",
    # Indonesia — abbreviated names
    ("Indonesia", "BABEL"): "Kepulauan Bangka Belitung",
    ("Indonesia", "BANGKA BELITUNG"): "Kepulauan Bangka Belitung",
    # Malaysia — federal territories have "W.P." prefix in GAUL
    ("Malaysia", "KUALA LUMPUR"): "W.P. Kuala Lumpur",
    ("Malaysia", "LABUAN"): "W.P. Labuan",
    # Thailand — spelling variants
    ("Thailand", "BUNGKAN"): "Bueng Kan",
    ("Thailand", "BURIRAM"): "Buri Ram",
    ("Thailand", "CHAINAT"): "Chai Nat",
    ("Thailand", "CHONBURI"): "Chon Buri",
    ("Thailand", "KAMPAENG PHET"): "Kamphaeng Phet",
    ("Thailand", "LOPBURI"): "Lop Buri",
    ("Thailand", "NONG BUA LAMPHU"): "Nong Bua Lam Phu",
    ("Thailand", "PHACHINBURI"): "Prachin Buri",
    ("Thailand", "PHRA NAKHON SI AYUDHYA"): "Phra Nakhon Si Ayutthaya",
    ("Thailand", "PRACHUAP KHILIKHAN"): "Prachuap Khiri Khan",
    ("Thailand", "SAMUT PRAKARN"): "Samut Prakan",
    ("Thailand", "SAMUT SONGKHAM"): "Samut Songkhram",
    ("Thailand", "SI SAKET"): "Si Sa Ket",
    ("Thailand", "SINGBURI"): "Sing Buri",
    ("Thailand", "SUPHANBURI"): "Suphan Buri",
    ("Thailand", "TRAD"): "Trat",
}

# CSV rows whose adm_1_name refers to the whole country (national-level data).
# All L2 polygons for that country are dissolved into a single polygon.
NATIONAL_LEVEL = {
    ("Cambodia", "CAMBODIA"),
    ("Lao People's Democratic Republic", "LAO PEOPLE'S DEMOCRATIC REPUBLIC"),
    ("Singapore", "SINGAPORE"),
    ("Viet Nam", "VIET NAM"),
}


def dissolve_to_l1(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve GAUL L2 polygons to admin-1 level.

    Myanmar is special: its gaul1_name values are generic categories
    ("Region", "State", "Union Territory"), so the actual admin-1 names
    live in gaul2_name. All other countries use gaul1_name.
    """
    frames = []

    # Myanmar: use gaul2_name as the admin-1 unit
    mm = gdf[gdf["gaul0_name"] == "Myanmar"].copy()
    if not mm.empty:
        mm_l1 = (
            mm.dissolve(by=["gaul0_name", "gaul2_name"])
            .reset_index()[["gaul0_name", "gaul2_name", "geometry"]]
            .rename(columns={"gaul0_name": "admin", "gaul2_name": "name"})
        )
        frames.append(mm_l1)

    # All other countries: use gaul1_name
    other = gdf[gdf["gaul0_name"] != "Myanmar"].copy()
    if not other.empty:
        other_l1 = (
            other.dissolve(by=["gaul0_name", "gaul1_name"])
            .reset_index()[["gaul0_name", "gaul1_name", "geometry"]]
            .rename(columns={"gaul0_name": "admin", "gaul1_name": "name"})
        )
        frames.append(other_l1)

    l1 = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(l1, geometry="geometry", crs=gdf.crs)


def main():
    df = pd.read_csv(CSV_PATH)
    gdf = gpd.read_file(SHP_PATH)

    # Map CSV country names to GAUL gaul0_name
    df["admin_key"] = df["adm_0_name"].apply(
        lambda x: COUNTRY_MAP.get(x, x.title())
    )
    df["adm1_upper"] = df["adm_1_name"].str.upper()

    # Dissolve L2 → L1
    print("Dissolving L2 → L1 (this may take a moment)...")
    l1_gdf = dissolve_to_l1(gdf)
    l1_gdf["name_upper"] = l1_gdf["name"].str.upper()

    csv_pairs = df[["admin_key", "adm_1_name", "adm1_upper"]].drop_duplicates()

    keep_names: set[str] = set()       # dissolved L1 'name' values to retain
    national_countries: set[str] = set()  # gaul0_name values needing full-country dissolve
    national_rows: list[gpd.GeoDataFrame] = []

    for _, row in csv_pairs.iterrows():
        admin = row["admin_key"]
        adm1_up = row["adm1_upper"]

        # 1. Manual mapping
        mapped = MANUAL_ADM1_MAP.get((admin, adm1_up))
        if mapped:
            keep_names.add(mapped)
            continue

        # 2. National-level rows — dissolve entire country
        if (admin, adm1_up) in NATIONAL_LEVEL:
            national_countries.add(admin)
            continue

        # 3. Case-insensitive match on dissolved L1 name
        country_l1 = l1_gdf[l1_gdf["admin"] == admin]
        hit = country_l1[country_l1["name_upper"] == adm1_up]
        if not hit.empty:
            keep_names.update(hit["name"].tolist())
            continue

        print(f"[WARN] No shapefile match for: {admin!r} / {row['adm_1_name']!r}")

    # Build national-level dissolved polygons
    for country in national_countries:
        country_polys = gdf[gdf["gaul0_name"] == country]
        if country_polys.empty:
            print(f"[WARN] No L2 polygons found for country: {country!r}")
            continue
        dissolved_geom = country_polys.geometry.union_all()
        national_rows.append(
            gpd.GeoDataFrame(
                [{"name": country, "admin": country, "geometry": dissolved_geom}],
                crs=gdf.crs,
            )
        )

    # Filter dissolved L1 to matched names and drop helper column
    filtered = l1_gdf[l1_gdf["name"].isin(keep_names)].drop(columns=["name_upper"])

    if national_rows:
        filtered = pd.concat([filtered] + national_rows, ignore_index=True)
        filtered = gpd.GeoDataFrame(filtered, geometry="geometry", crs=gdf.crs)

    print(f"Original GAUL L2 rows  : {len(gdf)}")
    print(f"Dissolved L1 rows      : {len(l1_gdf)}")
    print(f"Filtered output rows   : {len(filtered)}")
    print(f"  — matched L1 regions : {len(filtered) - len(national_rows)}")
    print(f"  — national dissolves : {len(national_rows)}")
    print(f"Saving to {OUT_PATH}")
    filtered.to_parquet(OUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
