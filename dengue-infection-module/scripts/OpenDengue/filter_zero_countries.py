"""
Filter out countries from the OpenDengue spatial CSV where dengue_total
is mostly zero (i.e., zero_ratio >= threshold).

Usage:
    python filter_zero_countries.py [--threshold 0.9] [--input ...] [--output ...]
"""

import argparse
import pandas as pd
from pathlib import Path

DEFAULT_INPUT = (
    Path(__file__).parent.parent.parent
    / "main/interim/OpenDengue/spatial_SEARO_WPRO_EMRO_2000_2025.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent
    / "main/interim/OpenDengue/spatial_SEARO_WPRO_EMRO_2000_2025_filtered.csv"
)


def filter_mostly_zero_countries(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop countries whose proportion of zero dengue_total rows >= threshold.

    Parameters
    ----------
    df : DataFrame with at least columns 'adm_0_name' and 'dengue_total'.
    threshold : fraction of zeros above which a country is removed (default 0.9).

    Returns
    -------
    Filtered DataFrame and a summary of removed countries.
    """
    zero_ratio = (
        df.groupby("adm_0_name")["dengue_total"]
        .apply(lambda s: (s == 0).sum() / len(s))
        .rename("zero_ratio")
    )

    kept = zero_ratio[zero_ratio < threshold].index
    removed = zero_ratio[zero_ratio >= threshold]

    print(f"\nThreshold : {threshold:.0%} zeros")
    print(f"Countries before filter : {df['adm_0_name'].nunique()}")
    print(f"Countries removed       : {len(removed)}")
    if not removed.empty:
        print("\nRemoved countries (zero_ratio):")
        for country, ratio in removed.sort_values(ascending=False).items():
            print(f"  {country:<40} {ratio:.1%}")
    print(f"\nCountries kept          : {len(kept)}")

    return df[df["adm_0_name"].isin(kept)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Filter mostly-zero dengue countries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Zero-ratio threshold; countries at or above this are dropped (default: 0.9)",
    )
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)

    filtered = filter_mostly_zero_countries(df, threshold=args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    print(f"\nSaved filtered data to: {args.output}")
    print(f"Rows: {len(df):,} → {len(filtered):,}")


if __name__ == "__main__":
    main()
