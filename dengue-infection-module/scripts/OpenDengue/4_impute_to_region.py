import pandas as pd

INPUT_PATH = "/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/filtered_sea_2011_2018.csv"
OUTPUT_PATH = "/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/filtered_sea_2011_2018_SLVC_imputed.csv"

df = pd.read_csv(INPUT_PATH)

# Fix: ensure name columns are object dtype, not float64 (happens when all-NaN on read)
for col in ["adm_1_name", "adm_2_name"]:
    df[col] = df[col].astype(object)

mask = df["adm_0_name"].isin([
    "SINGAPORE",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "VIET NAM",
    "CAMBODIA",
])

df.loc[mask, "adm_1_name"] = df.loc[mask, "adm_0_name"]
df.loc[mask, "adm_2_name"] = df.loc[mask, "adm_0_name"]

print(f"Number of unique adm_1_name: {df['adm_1_name'].nunique()}")
print(f"Unique adm_1_names: {sorted(df['adm_1_name'].dropna().unique())}")


# ── Aggregate Admin0/Week → synthetic Admin0/Month for countries that have
#    Admin1/Year distribution data but no Admin0/Month data (e.g. Malaysia).
#    These synthetic rows are picked up naturally by the Case 1/2 logic below.
import calendar as _cal

_adm0_month_countries = set(df.loc[(df["S_res"] == "Admin0") & (df["T_res"] == "Month"), "adm_0_name"])
_adm1_year_countries  = set(df.loc[(df["S_res"] == "Admin1") & (df["T_res"] == "Year"),  "adm_0_name"])
_weekly_only = df[
    (df["S_res"] == "Admin0") &
    (df["T_res"] == "Week") &
    (~df["adm_0_name"].isin(_adm0_month_countries)) &
    (df["adm_0_name"].isin(_adm1_year_countries))
].copy()

if not _weekly_only.empty:
    _weekly_only["_month_ts"] = (
        pd.to_datetime(_weekly_only["calendar_start_date"])
        .dt.to_period("M").dt.to_timestamp()
    )
    _synthetic_rows = []
    for (country, month_ts), grp in _weekly_only.groupby(["adm_0_name", "_month_ts"]):
        template = grp.drop(columns=["_month_ts"]).iloc[0].copy()
        year, month = month_ts.year, month_ts.month
        last_day = _cal.monthrange(year, month)[1]
        template["calendar_start_date"] = month_ts.strftime("%Y-%m-%d")
        template["calendar_end_date"]   = f"{year}-{month:02d}-{last_day:02d}"
        template["Year"]          = year
        template["dengue_total"]  = grp["dengue_total"].sum()
        template["T_res"]         = "Month"
        template["UUID"]          = f"WEEKLY-AGG-{country.replace(' ', '_')}-{template['calendar_start_date']}"
        _synthetic_rows.append(template)
    _synthetic_df = pd.DataFrame(_synthetic_rows)
    df = pd.concat([df, _synthetic_df], ignore_index=True)
    print(f"\nAggregated {len(_synthetic_rows)} synthetic Admin0/Month rows from weekly data "
          f"for: {sorted(_weekly_only['adm_0_name'].unique())}")


# Separate the two data types we need
adm0_monthly = df[(df["S_res"] == "Admin0") & (df["T_res"] == "Month")].copy()
adm1_yearly = df[(df["S_res"] == "Admin1") & (df["T_res"] == "Year")].copy()

adm0_keys = set(zip(adm0_monthly["adm_0_name"], adm0_monthly["Year"]))
adm1_keys = set(zip(adm1_yearly["adm_0_name"], adm1_yearly["Year"]))

# Case 1: same-year Admin1 distribution available
same_year_keys = adm0_keys & adm1_keys

# Case 2: Admin0 monthly exists but no same-year Admin1 — borrow nearest year's distribution
# Only consider countries that have Admin1 data in at least one year
adm1_countries = {country for country, _ in adm1_keys}
borrowed_year_keys = {
    (country, year)
    for country, year in adm0_keys - same_year_keys
    if country in adm1_countries
}

print(f"Case 1 — same-year Admin1 distribution: {len(same_year_keys)} country-year pairs")
for k in sorted(same_year_keys):
    print(f"  {k}")

print(f"\nCase 2 — borrowed Admin1 distribution: {len(borrowed_year_keys)} country-year pairs")


def get_nearest_adm1_year(country, target_year, adm1_yearly):
    """Return the Admin1 yearly rows for the year closest to target_year."""
    available_years = adm1_yearly[adm1_yearly["adm_0_name"] == country]["Year"].unique()
    nearest = min(available_years, key=lambda y: abs(y - target_year))
    return adm1_yearly[
        (adm1_yearly["adm_0_name"] == country) & (adm1_yearly["Year"] == nearest)
    ], nearest


def impute_rows(country, year, national_months, regions, donor_year, case):
    rows = []
    adm1_year_total = regions["dengue_total"].sum()
    if adm1_year_total == 0:
        print(f"  WARNING: Admin1 yearly total is 0 for {country} {donor_year}, skipping {year}.")
        return rows

    region_shares = regions.set_index("adm_1_name")["dengue_total"] / adm1_year_total

    for _, month_row in national_months.iterrows():
        for adm1_name, share in region_shares.items():
            region_row = regions[regions["adm_1_name"] == adm1_name].iloc[0]

            new_row = region_row.copy()
            new_row["calendar_start_date"] = month_row["calendar_start_date"]
            new_row["calendar_end_date"] = month_row["calendar_end_date"]
            new_row["Year"] = month_row["Year"]
            new_row["dengue_total"] = month_row["dengue_total"] * share
            new_row["T_res"] = "Month"
            uuid_tag = f"IMPUTED-{case}"
            if donor_year != year:
                uuid_tag += f"-donor{donor_year}"
            new_row["UUID"] = f"{uuid_tag}-{region_row['UUID']}-{month_row['calendar_start_date']}"

            rows.append(new_row)
    return rows


SLVC_COUNTRIES = [
    "SINGAPORE",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "VIET NAM",
    "CAMBODIA",
]
YEARS = list(range(2011, 2019))

MONTH_RANGES = [
    ("01", "01", "31"), ("02", "02", "28"), ("03", "03", "31"), ("04", "04", "30"),
    ("05", "05", "31"), ("06", "06", "30"), ("07", "07", "31"), ("08", "08", "31"),
    ("09", "09", "30"), ("10", "10", "31"), ("11", "11", "30"), ("12", "12", "31"),
]

META_COLS = ["ISO_A0", "FAO_GAUL_code", "RNE_iso_code", "IBGE_code",
             "case_definition_standardised", "who_region"]

# Build month date pairs for a given year (handles leap years for Feb)
def month_dates_for_year(year):
    dates = []
    for mm, _, _ in MONTH_RANGES:
        import calendar
        last_day = calendar.monthrange(year, int(mm))[1]
        start = f"{year}-{mm}-01"
        end = f"{year}-{mm}-{last_day:02d}"
        dates.append((start, end))
    return dates


slvc_rows = []

for country in SLVC_COUNTRIES:
    country_df = df[df["adm_0_name"] == country]

    # Pull metadata from any existing Admin0 row for this country
    meta_source = country_df[country_df["S_res"] == "Admin0"]
    if meta_source.empty:
        meta_source = country_df
    meta = meta_source.iloc[0]

    for year in YEARS:
        year_monthly = country_df[
            (country_df["T_res"] == "Month") &
            (country_df["Year"] == year) &
            (country_df["S_res"] == "Admin0")
        ]

        # Aggregate weekly → monthly by assigning each week to the month of its start date
        year_weekly = country_df[
            (country_df["T_res"] == "Week") &
            (country_df["Year"] == year) &
            (country_df["S_res"] == "Admin0")
        ].copy()
        if not year_weekly.empty:
            year_weekly["month_start"] = pd.to_datetime(
                year_weekly["calendar_start_date"]
            ).dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
            weekly_agg = year_weekly.groupby("month_start")["dengue_total"].sum()
        else:
            weekly_agg = pd.Series(dtype=float)

        for start, end in month_dates_for_year(year):
            # Priority 1: direct Admin0/Month row
            month_match = year_monthly[year_monthly["calendar_start_date"] == start]
            if not month_match.empty:
                dengue_val = month_match.iloc[0]["dengue_total"]
                uuid_tag = f"SLVC-promoted-{month_match.iloc[0]['UUID']}"
            # Priority 2: aggregated weekly data
            elif start in weekly_agg.index:
                dengue_val = weekly_agg[start]
                uuid_tag = f"SLVC-weekly-agg-{country.replace(' ', '_')}-{start}"
            else:
                dengue_val = float("nan")
                uuid_tag = f"SLVC-placeholder-{country.replace(' ', '_')}-{start}"

            slvc_rows.append({
                "adm_0_name": country,
                "adm_1_name": country,
                "adm_2_name": country,
                "full_name": country,
                "ISO_A0": meta["ISO_A0"],
                "FAO_GAUL_code": meta["FAO_GAUL_code"],
                "RNE_iso_code": meta["RNE_iso_code"],
                "IBGE_code": meta["IBGE_code"],
                "calendar_start_date": start,
                "calendar_end_date": end,
                "Year": year,
                "dengue_total": dengue_val,
                "case_definition_standardised": meta["case_definition_standardised"],
                "S_res": "Admin1",
                "T_res": "Month",
                "UUID": uuid_tag,
                "who_region": meta["who_region"],
            })

slvc_df = pd.DataFrame(slvc_rows)
promoted   = slvc_df["UUID"].str.startswith("SLVC-promoted").sum()
weekly_agg_count = slvc_df["UUID"].str.startswith("SLVC-weekly-agg").sum()
placeholders = slvc_df["UUID"].str.startswith("SLVC-placeholder").sum()
print(f"\nSLVC single-region rows: {promoted} promoted + {weekly_agg_count} weekly-agg + {placeholders} placeholder = {len(slvc_df)} total")

# Drop original Admin0/Month and Admin0/Week rows for SLVC countries (replaced by slvc_df)
slvc_adm0_drop_mask = (
    df["adm_0_name"].isin(SLVC_COUNTRIES) &
    (df["S_res"] == "Admin0") &
    (df["T_res"].isin(["Month", "Week"]))
)
df = df[~slvc_adm0_drop_mask].copy()

# Rebuild adm0_monthly and adm1_yearly after dropping SLVC Admin0/Month rows
adm0_monthly = df[(df["S_res"] == "Admin0") & (df["T_res"] == "Month")].copy()
adm1_yearly = df[(df["S_res"] == "Admin1") & (df["T_res"] == "Year")].copy()

adm0_keys = set(zip(adm0_monthly["adm_0_name"], adm0_monthly["Year"]))
adm1_keys = set(zip(adm1_yearly["adm_0_name"], adm1_yearly["Year"]))
same_year_keys = adm0_keys & adm1_keys
adm1_countries = {country for country, _ in adm1_keys}
borrowed_year_keys = {
    (country, year)
    for country, year in adm0_keys - same_year_keys
    if country in adm1_countries
}

imputed_rows = []

# Case 1
for country, year in sorted(same_year_keys):
    national_months = adm0_monthly[
        (adm0_monthly["adm_0_name"] == country) & (adm0_monthly["Year"] == year)
    ]
    regions = adm1_yearly[
        (adm1_yearly["adm_0_name"] == country) & (adm1_yearly["Year"] == year)
    ]
    imputed_rows.extend(impute_rows(country, year, national_months, regions, year, "C1"))

case1_count = len(imputed_rows)

# Case 2
for country, year in sorted(borrowed_year_keys):
    national_months = adm0_monthly[
        (adm0_monthly["adm_0_name"] == country) & (adm0_monthly["Year"] == year)
    ]
    regions, donor_year = get_nearest_adm1_year(country, year, adm1_yearly)
    print(f"  {country} {year} → borrowing distribution from {donor_year}")
    imputed_rows.extend(impute_rows(country, year, national_months, regions, donor_year, "C2"))

case2_count = len(imputed_rows) - case1_count

imputed_df = pd.DataFrame(imputed_rows)
print(f"\nImputed rows: {case1_count} (Case 1) + {case2_count} (Case 2) = {len(imputed_df)} total")

imputed_country_years = same_year_keys | borrowed_year_keys

# Country-years that already have Admin1 monthly data (imputed or original)
adm1_monthly_keys = set(
    zip(imputed_df["adm_0_name"], imputed_df["Year"])
) | set(
    zip(
        df.loc[(df["S_res"] == "Admin1") & (df["T_res"] == "Month"), "adm_0_name"],
        df.loc[(df["S_res"] == "Admin1") & (df["T_res"] == "Month"), "Year"],
    )
)

drop_mask = (
    # Source Admin0 monthly rows that were imputed
    ((df["S_res"] == "Admin0") & (df["T_res"] == "Month") &
     df.apply(lambda r: (r["adm_0_name"], r["Year"]) in imputed_country_years, axis=1))
    |
    # Source Admin1 yearly rows that were used for distribution
    ((df["S_res"] == "Admin1") & (df["T_res"] == "Year") &
     df.apply(lambda r: (r["adm_0_name"], r["Year"]) in imputed_country_years, axis=1))
    |
    # Reason 1: Admin0 weekly rows — cannot be imputed to monthly Admin1
    ((df["S_res"] == "Admin0") & (df["T_res"] == "Week"))
    |
    # Reason 2: Admin0 yearly rows where Admin1 monthly already exists
    ((df["S_res"] == "Admin0") & (df["T_res"] == "Year") &
     df.apply(lambda r: (r["adm_0_name"], r["Year"]) in adm1_monthly_keys, axis=1))
)

result = pd.concat([df[~drop_mask], imputed_df, slvc_df], ignore_index=True)
result = result.sort_values(["adm_0_name", "adm_1_name", "calendar_start_date"]).reset_index(drop=True)

# ── Myanmar 2011: placeholder Admin1/Month rows (dengue_total unknown) ────────
myan_adm1 = df[(df["adm_0_name"] == "MYANMAR") & (df["S_res"] == "Admin1")]
myan_adm1_meta = (
    myan_adm1.drop_duplicates("adm_1_name")
    .set_index("adm_1_name")
    [["adm_2_name", "full_name", "ISO_A0", "FAO_GAUL_code", "RNE_iso_code",
      "IBGE_code", "case_definition_standardised", "who_region"]]
)

month_ranges_2011 = [
    ("2011-01-01", "2011-01-31"), ("2011-02-01", "2011-02-28"),
    ("2011-03-01", "2011-03-31"), ("2011-04-01", "2011-04-30"),
    ("2011-05-01", "2011-05-31"), ("2011-06-01", "2011-06-30"),
    ("2011-07-01", "2011-07-31"), ("2011-08-01", "2011-08-31"),
    ("2011-09-01", "2011-09-30"), ("2011-10-01", "2011-10-31"),
    ("2011-11-01", "2011-11-30"), ("2011-12-01", "2011-12-31"),
]

myan_2011_rows = []
for adm1_name, meta in myan_adm1_meta.iterrows():
    for start, end in month_ranges_2011:
        myan_2011_rows.append({
            "adm_0_name": "MYANMAR",
            "adm_1_name": adm1_name,
            "adm_2_name": meta["adm_2_name"],
            "full_name": meta["full_name"],
            "ISO_A0": meta["ISO_A0"],
            "FAO_GAUL_code": meta["FAO_GAUL_code"],
            "RNE_iso_code": meta["RNE_iso_code"],
            "IBGE_code": meta["IBGE_code"],
            "calendar_start_date": start,
            "calendar_end_date": end,
            "Year": 2011,
            "dengue_total": float("nan"),
            "case_definition_standardised": meta["case_definition_standardised"],
            "S_res": "Admin1",
            "T_res": "Month",
            "UUID": f"PLACEHOLDER-MMR-2011-{adm1_name.replace(' ', '_')}-{start}",
            "who_region": meta["who_region"],
        })

myan_2011_df = pd.DataFrame(myan_2011_rows)
print(f"\nMyanmar 2011 placeholder rows: {len(myan_2011_df)} "
      f"({myan_adm1_meta.shape[0]} regions × 12 months)")

# Drop the original Myanmar 2011 Admin0/Year row (replaced by placeholders)
result = result[~(
    (result["adm_0_name"] == "MYANMAR") &
    (result["Year"] == 2011) &
    (result["S_res"] == "Admin0") &
    (result["T_res"] == "Year")
)]

result = pd.concat([result, myan_2011_df], ignore_index=True)
result = result.sort_values(["adm_0_name", "adm_1_name", "calendar_start_date"]).reset_index(drop=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── Remove MONGAR (stray Bhutan entry, not a SEA region) ─────────────────────
result = result[result["adm_1_name"] != "MONGAR"].copy()
print(f"\nDropped MONGAR (stray Bhutan entry).")
# ─────────────────────────────────────────────────────────────────────────────

# ── Fill remaining gaps with NaN placeholders ─────────────────────────────────
adm1_monthly_result = result[(result["S_res"] == "Admin1") & (result["T_res"] == "Month")].copy()
adm1_monthly_result["YearMonth"] = adm1_monthly_result["calendar_start_date"].astype(str).str[:7]

all_year_months = {
    f"{y}-{m:02d}": (f"{y}-{m:02d}-01", f"{y}-{m:02d}-{__import__('calendar').monthrange(y, m)[1]:02d}")
    for y in YEARS for m in range(1, 13)
}

region_meta = (
    result[result["S_res"] == "Admin1"]
    .drop_duplicates("adm_1_name")
    .set_index("adm_1_name")
    [["adm_0_name", "adm_2_name", "full_name", "ISO_A0", "FAO_GAUL_code",
      "RNE_iso_code", "IBGE_code", "case_definition_standardised", "who_region"]]
)

existing_ym = adm1_monthly_result.groupby("adm_1_name")["YearMonth"].apply(set)

gap_rows = []
for adm1_name, meta in region_meta.iterrows():
    covered = existing_ym.get(adm1_name, set())
    for ym, (start, end) in all_year_months.items():
        if ym not in covered:
            gap_rows.append({
                "adm_0_name": meta["adm_0_name"],
                "adm_1_name": adm1_name,
                "adm_2_name": meta["adm_2_name"],
                "full_name": meta["full_name"],
                "ISO_A0": meta["ISO_A0"],
                "FAO_GAUL_code": meta["FAO_GAUL_code"],
                "RNE_iso_code": meta["RNE_iso_code"],
                "IBGE_code": meta["IBGE_code"],
                "calendar_start_date": start,
                "calendar_end_date": end,
                "Year": int(ym[:4]),
                "dengue_total": float("nan"),
                "case_definition_standardised": meta["case_definition_standardised"],
                "S_res": "Admin1",
                "T_res": "Month",
                "UUID": f"PLACEHOLDER-GAP-{adm1_name.replace(' ', '_')}-{start}",
                "who_region": meta["who_region"],
            })

gap_df = pd.DataFrame(gap_rows)
print(f"Gap-fill placeholders: {len(gap_df)} rows across {gap_df['adm_1_name'].nunique() if not gap_df.empty else 0} regions")

result = pd.concat([result, gap_df], ignore_index=True)
result = result.sort_values(["adm_0_name", "adm_1_name", "calendar_start_date"]).reset_index(drop=True)
# ─────────────────────────────────────────────────────────────────────────────

result.to_csv(OUTPUT_PATH, index=False)
print(f"Output written to: {OUTPUT_PATH}")
print(f"Dropped {drop_mask.sum()} source rows (Admin0-monthly + Admin1-yearly for imputed country-years)")
print(f"Total rows: {(~drop_mask).sum()} kept + {len(imputed_df)} imputed + {len(slvc_df)} SLVC + {len(myan_2011_df)} Myanmar-2011 placeholders = {len(result)}")

# ── Completeness check: each region should have 8 years × 12 months = 96 rows ─
adm1_monthly_result = result[(result["S_res"] == "Admin1") & (result["T_res"] == "Month")].copy()
adm1_monthly_result["YearMonth"] = (
    adm1_monthly_result["calendar_start_date"].astype(str).str[:7]
)

counts = adm1_monthly_result.groupby("adm_1_name")["YearMonth"].nunique()
unique_regions = counts.shape[0]
expected_rows = 8 * 12  # 8 years × 12 months

complete = counts[counts == expected_rows]
incomplete = counts[counts != expected_rows].sort_values()

print(f"\n── Completeness check ──────────────────────────────────────────────")
print(f"Unique Admin1 regions with monthly rows: {unique_regions}")
print(f"Complete (96 year-months): {len(complete)} / {unique_regions}")
if not incomplete.empty:
    print(f"Incomplete regions ({len(incomplete)}):")
    for region, n in incomplete.items():
        print(f"  {region}: {n} / {expected_rows} year-months")
else:
    print("All regions have full 8 × 12 = 96 year-month coverage.")
# ─────────────────────────────────────────────────────────────────────────────
