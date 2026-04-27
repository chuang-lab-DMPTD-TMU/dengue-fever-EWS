import pandas as pd

ENV_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/machine-learning-module/main/data/processed/SEA_env_combined_monthly_2011-2018.csv"
)
DENGUE_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/dengue-infection-module/main/interim/OpenDengue"
    "/filtered_sea_2011_2018_SLVC_imputed.csv"
)
OUT_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/machine-learning-module/main/data/processed/SEA_dengue_env_monthly_2011-2018.csv"
)

# ── Name mappings ─────────────────────────────────────────────────────────────
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM": "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Lao People's Democratic Republic",
    "VIET NAM": "Viet Nam",
}

NATIONAL_LEVEL = {
    "Cambodia",
    "Lao People's Democratic Republic",
    "Singapore",
    "Viet Nam",
}

MANUAL_ADM1_MAP = {
    # Brunei
    ("Brunei Darussalam", "BELAIT DISTRICT"): "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"): "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"): "Tutong",
    # Indonesia
    ("Indonesia", "BABEL"): "Kepulauan Bangka Belitung",
    ("Indonesia", "BANGKA BELITUNG"): "Kepulauan Bangka Belitung",
    ("Indonesia", "D.I YOGYA"): "Daerah Istimewa Yogyakarta",
    ("Indonesia", "KALIMANTAN SELATA"): "Kalimantan Selatan",
    ("Indonesia", "KEPULAUAN-RIAU"): "Kepulauan Riau",
    ("Indonesia", "NUSATENGGARA BARAT"): "Nusa Tenggara Barat",
    ("Indonesia", "NUSATENGGARA TIMUR"): "Nusa Tenggara Timur",
    ("Indonesia", "SULAWESI SELATA"): "Sulawesi Selatan",
    ("Indonesia", "SUMATERA SELATA"): "Sumatera Selatan",
    # Malaysia
    ("Malaysia", "KUALA LUMPUR"): "W.P. Kuala Lumpur",
    ("Malaysia", "LABUAN"): "W.P. Labuan",
    # Myanmar
    ("Myanmar", "AYAYARWADDY"): "Ayeyarwady",
    ("Myanmar", "BAGO (E)"): "Bago (East)",
    ("Myanmar", "BAGO (W)"): "Bago (West)",
    ("Myanmar", "NAYPYITAW"): "Nay Pyi Taw",
    ("Myanmar", "SHAN (N)"): "Shan (North)",
    ("Myanmar", "SHAN (S)"): "Shan (South)",
    # Philippines
    ("Philippines", "REGION 4"): "Region Iv-A (Calabarzon)",
    # Thailand
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


def resolve_shp_name(admin_key: str, adm1_upper: str) -> str:
    """Map a raw ADM1 name to the canonical shapefile name."""
    if admin_key in NATIONAL_LEVEL:
        return admin_key
    return MANUAL_ADM1_MAP.get((admin_key, adm1_upper), adm1_upper.title())


# ── Load ──────────────────────────────────────────────────────────────────────
env = pd.read_csv(ENV_PATH, parse_dates=["Date"])
env["name"] = env["name"].str.strip()
env["admin"] = env["admin"].str.strip()
print(f"env shape       : {env.shape}")

dengue = pd.read_csv(
    DENGUE_PATH,
    parse_dates=["calendar_start_date"],
    usecols=["adm_0_name", "adm_1_name", "calendar_start_date", "dengue_total", "S_res", "T_res"],
)
dengue = dengue[(dengue["S_res"] == "Admin1") & (dengue["T_res"] == "Month")].copy()
print(f"dengue shape    : {dengue.shape}")

# ── Normalise names ───────────────────────────────────────────────────────────
dengue["admin_key"] = dengue["adm_0_name"].map(COUNTRY_MAP).fillna(dengue["adm_0_name"].str.title())
dengue["shp_name"] = dengue.apply(
    lambda r: resolve_shp_name(r["admin_key"], r["adm_1_name"].upper()), axis=1
).str.strip()
dengue["year_month"] = dengue["calendar_start_date"].dt.to_period("M").dt.to_timestamp()

dengue_agg = (
    dengue.groupby(["admin_key", "shp_name", "year_month"], as_index=False)["dengue_total"]
    .sum(min_count=1)
)
print(f"dengue_agg shape: {dengue_agg.shape}")

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = env.merge(
    dengue_agg,
    left_on=["admin", "name", "Date"],
    right_on=["admin_key", "shp_name", "year_month"],
    how="left",
).drop(columns=["admin_key", "shp_name", "year_month"])

merged["IR"] = merged["dengue_total"] / merged["population_sum"] * 100000

# ── Diagnostics ───────────────────────────────────────────────────────────────
total = len(merged)
matched = merged["dengue_total"].notna().sum()
unmatched = sorted(merged.loc[merged["dengue_total"].isna(), "name"].unique())

print(f"\nmerged shape          : {merged.shape}")
print(f"rows with dengue_total: {matched} / {total} ({100 * matched / total:.1f}%)")
print(f"unmatched env names   : {unmatched[:20]}")
print(f"\nNA counts:\n{merged[['dengue_total', 'population_sum', 'IR']].isna().sum()}")
print(f"\nIR summary:\n{merged['IR'].describe()}")

# ── Save ──────────────────────────────────────────────────────────────────────
merged.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")