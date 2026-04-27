"""
Unify OpenDengue spatial CSV to adm_1_name level at monthly granularity.

Outputs (relative to input directory):
  unified_monthly_adm1.csv       - Main output: all entries at adm_1/month level.
  aggregated_duplicates.csv      - Sub-monthly rows aggregated where a monthly entry
                                   already existed for that (adm_0, adm_1, year-month).
  propagated_adm0_to_adm1.csv   - Rows where adm_0_name was propagated to adm_1_name.
  year_resolution.csv            - Entries with T_res == 'Year'.
"""

import pandas as pd
from pathlib import Path

INPUT = Path(__file__).parent.parent.parent.parent / "main/interim/OpenDengue/spatial_SEARO_WPRO_EMRO_2000_2025.csv"
OUT_DIR = INPUT.parent


def month_end(period: pd.Period) -> pd.Timestamp:
    """Return the last day of a Period month as a Timestamp."""
    return period.to_timestamp(how="E").normalize()


def aggregate_group(group: pd.DataFrame, ym: pd.Period) -> pd.Series:
    """Collapse a group of sub-monthly rows into a single monthly row."""
    row = group.iloc[0].copy()
    row["dengue_total"] = group["dengue_total"].sum()
    row["T_res"] = "Month"
    row["calendar_start_date"] = ym.to_timestamp(how="S").strftime("%Y-%m-%d")
    row["calendar_end_date"] = month_end(ym).strftime("%Y-%m-%d")
    row["UUID"] = f"AGGREGATED-{row['adm_0_name']}-{row['adm_1_name']}-{ym}"
    return row


def main():
    df = pd.read_csv(INPUT, dtype=str)

    # Normalise dengue_total to numeric; keep as object col for now
    df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce")

    # ── 1. Split out Year T_res rows ─────────────────────────────────────────
    year_mask = df["T_res"] == "Year"
    year_df = df[year_mask].copy()
    year_df.to_csv(OUT_DIR / "year_resolution.csv", index=False)
    print(f"[year_resolution]        {len(year_df):>7,} rows")

    df_work = df[~year_mask].copy()

    # ── 2. Flag and propagate adm_0_only rows ────────────────────────────────
    adm1_empty = df_work["adm_1_name"].isna() | (df_work["adm_1_name"].str.strip() == "")
    df_work["_propagated"] = adm1_empty
    df_work.loc[adm1_empty, "adm_1_name"] = df_work.loc[adm1_empty, "adm_0_name"]

    # ── 3. Assign year-month based on calendar_start_date ────────────────────
    df_work["_ym"] = pd.to_datetime(df_work["calendar_start_date"], errors="coerce").dt.to_period("M")

    # ── 4. Separate already-monthly rows from sub-monthly (Week, etc.) ───────
    monthly_mask = df_work["T_res"] == "Month"
    df_monthly = df_work[monthly_mask].copy()
    df_submonthly = df_work[~monthly_mask].copy()

    # Build lookup set of existing monthly (adm_0, adm_1, year_month) combos
    existing_monthly = set(
        zip(
            df_monthly["adm_0_name"],
            df_monthly["adm_1_name"].astype(str),
            df_monthly["_ym"].astype(str),
        )
    )

    # ── 5. Aggregate sub-monthly rows to monthly ──────────────────────────────
    group_keys = ["adm_0_name", "adm_1_name", "_ym"]
    no_conflict_rows: list[pd.Series] = []
    conflict_rows: list[pd.Series] = []

    for (adm0, adm1, ym), grp in df_submonthly.groupby(group_keys, dropna=False):
        agg = aggregate_group(grp, ym)
        combo = (adm0, str(adm1), str(ym))
        if combo in existing_monthly:
            conflict_rows.append(agg)
        else:
            no_conflict_rows.append(agg)

    # ── 6. Build main output ──────────────────────────────────────────────────
    parts = [df_monthly]
    if no_conflict_rows:
        parts.append(pd.DataFrame(no_conflict_rows))
    main_df = pd.concat(parts, ignore_index=True)

    # ── 7. Propagated-rows CSV ────────────────────────────────────────────────
    # Includes both monthly and aggregated rows that originated from adm_0_only entries.
    prop_from_monthly = main_df[main_df["_propagated"] == True].copy()

    prop_from_conflict: pd.DataFrame = pd.DataFrame()
    if conflict_rows:
        conflict_df_tmp = pd.DataFrame(conflict_rows)
        prop_from_conflict = conflict_df_tmp[conflict_df_tmp["_propagated"] == True].copy()

    propagated_df = pd.concat(
        [prop_from_monthly, prop_from_conflict], ignore_index=True
    )

    # ── 8. Drop internal helper columns ──────────────────────────────────────
    drop_cols = ["_propagated", "_ym"]

    # Exclude propagated rows from the main output
    main_out_df = main_df[main_df["_propagated"] == False].copy()
    main_out_df = main_out_df.drop(columns=drop_cols, errors="ignore")
    propagated_df = propagated_df.drop(columns=drop_cols, errors="ignore")

    main_out_df.to_csv(OUT_DIR / "unified_monthly_adm1.csv", index=False)
    propagated_df.to_csv(OUT_DIR / "propagated_adm0_to_adm1.csv", index=False)
    print(f"[unified_monthly_adm1]   {len(main_out_df):>7,} rows")
    print(f"[propagated_adm0_to_adm1]{len(propagated_df):>6,} rows")

    # ── 9. Conflict / duplicate CSV ───────────────────────────────────────────
    if conflict_rows:
        conflict_df = pd.DataFrame(conflict_rows).drop(columns=drop_cols, errors="ignore")
        conflict_df.to_csv(OUT_DIR / "aggregated_duplicates.csv", index=False)
        print(f"[aggregated_duplicates]  {len(conflict_df):>7,} rows")
    else:
        print("[aggregated_duplicates]        0 rows — no conflicts found")


if __name__ == "__main__":
    main()
