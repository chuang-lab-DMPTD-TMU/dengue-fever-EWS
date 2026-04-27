import random
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import Queen
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/dengue-infection-module/main/interim/OpenDengue"
    "/filtered_sea_2011_2018_SLVC_imputed.csv"
)
GEO_PATH = (
    "/home/patwuch/Documents/projects/Chuang-Lab-TMU"
    "/dengue-infection-module/main/external/geoparquet"
    "/gaul_2024_sea_filtered.parquet"
)
RANDOM_SEED = 42
N_SLICES = 10

# ── Name mappings (mirrors script 5) ─────────────────────────────────────────
# Maps CSV adm_0_name (uppercase) → GAUL gaul0_name stored in the parquet.
# Only entries that differ from plain .title() are listed.
COUNTRY_MAP = {
    "BRUNEI DARUSSALAM": "Brunei Darussalam",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Lao People's Democratic Republic",
    "VIET NAM": "Viet Nam",
}

# Maps (gaul0_name, CSV adm_1_name uppercase) → dissolved-L1 'name' in the parquet.
# Covers cases where CSV name differs from the GAUL gaul1_name/gaul2_name.
# Myanmar, Philippines, and most Indonesia entries match by title-case and need no entry.
MANUAL_ADM1_MAP = {
    # Brunei — CSV appends " DISTRICT"
    ("Brunei Darussalam", "BELAIT DISTRICT"): "Belait",
    ("Brunei Darussalam", "BRUNEI MUARA DISTRICT"): "Brunei And Muara",
    ("Brunei Darussalam", "TEMBURONG DISTRICT"): "Temburong",
    ("Brunei Darussalam", "TUTONG DISTRICT"): "Tutong",
    # Indonesia — abbreviated names
    ("Indonesia", "BABEL"): "Kepulauan Bangka Belitung",
    ("Indonesia", "BANGKA BELITUNG"): "Kepulauan Bangka Belitung",
    # Malaysia — federal territories carry a "W.P." prefix in GAUL
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

# National-level CSV rows: adm_1_name is the country name.
# The parquet stores these as a single dissolved polygon with name == gaul0_name.
# resolve_shp_name returns admin_key (== gaul0_name) for these entries.
NATIONAL_LEVEL = {
    ("Cambodia", "CAMBODIA"),
    ("Lao People's Democratic Republic", "LAO PEOPLE'S DEMOCRATIC REPUBLIC"),
    ("Singapore", "SINGAPORE"),
    ("Viet Nam", "VIET NAM"),
}


def resolve_shp_name(admin_key: str, adm1_upper: str) -> str | None:
    """Return the parquet 'name' for a given CSV admin/adm1 pair, or None to skip."""
    # National-level rows: parquet name == gaul0_name == admin_key
    if (admin_key, adm1_upper) in NATIONAL_LEVEL:
        return admin_key
    manual = MANUAL_ADM1_MAP.get((admin_key, adm1_upper))
    if manual:
        return manual
    # Default: title-case matches GAUL gaul1_name (or gaul2_name for Myanmar)
    return adm1_upper.title()


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH, parse_dates=["calendar_start_date"])
gdf_base = gpd.read_parquet(GEO_PATH)

# Keep only Admin1 monthly rows
df = df[(df["S_res"] == "Admin1") & (df["T_res"] == "Month")].copy()

# Build shapefile name lookup column on the dengue dataframe
df["admin_key"] = df["adm_0_name"].apply(lambda x: COUNTRY_MAP.get(x, x.title()))
df["adm1_upper"] = df["adm_1_name"].str.upper()
df["shp_name"] = df.apply(
    lambda r: resolve_shp_name(r["admin_key"], r["adm1_upper"]), axis=1
)

# Drop rows that have no shapefile match
df = df[df["shp_name"].notna()].copy()

# Where multiple CSV adm1 rows share the same shapefile polygon (e.g. Bago East/West),
# aggregate dengue_total by shp_name × time-slice before joining to geometry.
df["year_month"] = df["calendar_start_date"].dt.to_period("M").astype(str)

df_agg = (
    df.groupby(["shp_name", "year_month"], as_index=False)["dengue_total"]
    .sum(min_count=1)          # preserves NaN if all contributing rows are NaN
)

# ── Build joined GeoDataFrame ─────────────────────────────────────────────────
# Use shapefile 'name' as the join key
joined = gdf_base[["name", "geometry"]].merge(
    df_agg, left_on="name", right_on="shp_name", how="inner"
)
joined = joined.rename(columns={"name": "shp_name_geo"})

# ── Randomly select 10 time slices ───────────────────────────────────────────
all_slices = sorted(joined["year_month"].unique())
random.seed(RANDOM_SEED)
selected_slices = sorted(random.sample(all_slices, min(N_SLICES, len(all_slices))))

print(f"\nRandomly selected {len(selected_slices)} time slices (seed={RANDOM_SEED}):")
for s in selected_slices:
    print(f"  {s}")

# ── Moran's I per time slice ──────────────────────────────────────────────────
results = []

for ym in selected_slices:
    slice_gdf = joined[joined["year_month"] == ym].copy()

    # Drop rows with NaN dengue (islands / placeholder months)
    slice_gdf = slice_gdf.dropna(subset=["dengue_total"])
    slice_gdf = slice_gdf.reset_index(drop=True)

    if len(slice_gdf) < 4:
        print(f"[SKIP] {ym}: only {len(slice_gdf)} valid regions after NaN removal.")
        results.append({"year_month": ym, "n": len(slice_gdf),
                        "moran_i": np.nan, "p_sim": np.nan, "z_sim": np.nan})
        continue

    # Build Queen contiguity weights; use KNN-4 fallback for isolated islands
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w = Queen.from_dataframe(slice_gdf, silence_warnings=True)
        except Exception:
            from libpysal.weights import KNN
            w = KNN.from_dataframe(slice_gdf, k=4)

    w.transform = "r"   # row-standardise

    y = slice_gdf["dengue_total"].values.astype(float)

    mi = Moran(y, w, permutations=999)

    results.append({
        "year_month": ym,
        "n": len(slice_gdf),
        "moran_i": mi.I,
        "expected_i": mi.EI,
        "z_sim": mi.z_sim,
        "p_sim": mi.p_sim,
        "significant": mi.p_sim < 0.05,
    })
    sig = "**" if mi.p_sim < 0.01 else ("*" if mi.p_sim < 0.05 else "")
    print(
        f"  {ym}  I={mi.I:+.4f}  E[I]={mi.EI:.4f}  "
        f"z={mi.z_sim:+.3f}  p={mi.p_sim:.4f} {sig}  n={len(slice_gdf)}"
    )

results_df = pd.DataFrame(results)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Moran's I Results ────────────────────────────────────────────────")
print(results_df.to_string(index=False, float_format="{:.4f}".format))

# ── Plot: Moran's I across selected time slices ───────────────────────────────
fig, axes = plt.subplots(2, N_SLICES // 2, figsize=(20, 8))
axes = axes.flatten()

cmap = plt.cm.RdYlGn
norm_global = Normalize(
    vmin=joined["dengue_total"].quantile(0.05),
    vmax=joined["dengue_total"].quantile(0.95),
)

for ax, ym in zip(axes, selected_slices):
    slice_gdf = joined[joined["year_month"] == ym].copy().reset_index(drop=True)

    # Use log1p scale for choropleth (dengue counts are skewed)
    slice_gdf["plot_val"] = np.log1p(slice_gdf["dengue_total"].fillna(0))
    slice_gdf.plot(
        column="plot_val",
        ax=ax,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        linewidth=0.3,
        edgecolor="white",
    )

    row = results_df[results_df["year_month"] == ym].iloc[0]
    sig_label = (
        "p<0.01" if row["p_sim"] < 0.01
        else "p<0.05" if row["p_sim"] < 0.05
        else "n.s."
    )
    ax.set_title(
        f"{ym}\nI={row['moran_i']:.3f}  {sig_label}",
        fontsize=8,
    )
    ax.axis("off")

fig.suptitle(
    f"Dengue Moran's I — {N_SLICES} random monthly slices (SEA, 2011–2018)",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig("morans_i_slices.png", dpi=150, bbox_inches="tight")
print("\nMap saved to morans_i_slices.png")
plt.show()

# ── Moran scatter plot for the most significant slice ────────────────────────
best = results_df.dropna(subset=["moran_i"]).sort_values("p_sim").iloc[0]
ym_best = best["year_month"]

slice_gdf = (
    joined[joined["year_month"] == ym_best]
    .dropna(subset=["dengue_total"])
    .reset_index(drop=True)
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w_best = Queen.from_dataframe(slice_gdf, silence_warnings=True)
w_best.transform = "r"

y_best = slice_gdf["dengue_total"].values.astype(float)
mi_best = Moran(y_best, w_best, permutations=999)

fig2, ax2 = plt.subplots(figsize=(6, 5))
y_std = (y_best - y_best.mean()) / y_best.std()
wy_std = mi_best.z_sim  # spatial lag of standardised y (approximation via weights)

# Recompute spatial lag manually for the scatter
from libpysal.weights import lag_spatial
y_lag = lag_spatial(w_best, y_best)
y_lag_std = (y_lag - y_lag.mean()) / y_lag.std()

ax2.scatter(y_std, y_lag_std, alpha=0.6, edgecolors="k", linewidths=0.4)
ax2.axhline(0, color="grey", lw=0.8, linestyle="--")
ax2.axvline(0, color="grey", lw=0.8, linestyle="--")

m, b = np.polyfit(y_std, y_lag_std, 1)
x_line = np.linspace(y_std.min(), y_std.max(), 100)
ax2.plot(x_line, m * x_line + b, color="tomato", lw=1.5,
         label=f"slope ≈ I = {mi_best.I:.4f}")

ax2.set_xlabel("Standardised dengue_total")
ax2.set_ylabel("Spatial lag (standardised)")
ax2.set_title(f"Moran scatter — {ym_best}\nI={mi_best.I:.4f}  p={mi_best.p_sim:.4f}")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig("morans_i_scatter.png", dpi=150, bbox_inches="tight")
print("Moran scatter saved to morans_i_scatter.png")
plt.show()
