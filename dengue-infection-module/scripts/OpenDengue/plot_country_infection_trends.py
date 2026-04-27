"""
Plot annual dengue infection counts per country from INPAC_v1.csv.
All countries are shown on the same curves plot (year on x-axis).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "main" / "data" / "INPAC_v2.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "report"
OUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Year"] = df["Year"].astype(int)

# For each country × year, prefer Admin1 rows (sum sub-national) if they exist,
# otherwise fall back to the Admin0 (national) row for that year.
# This avoids double-counting while still capturing years only covered at Admin0.
admin1_annual = (
    df[df["S_res"] == "Admin1"]
    .groupby(["adm_0_name", "Year"])["dengue_total"]
    .sum()
    .reset_index()
)
admin1_annual["_source"] = "Admin1"

admin0_annual = (
    df[df["S_res"] == "Admin0"]
    .groupby(["adm_0_name", "Year"])["dengue_total"]
    .sum()
    .reset_index()
)
admin0_annual["_source"] = "Admin0"

# Keep Admin0 rows only for country × year combos not covered by Admin1
admin1_keys = set(zip(admin1_annual["adm_0_name"], admin1_annual["Year"]))
admin0_fallback = admin0_annual[
    ~admin0_annual.apply(lambda r: (r["adm_0_name"], r["Year"]) in admin1_keys, axis=1)
]

annual = (
    pd.concat([admin1_annual, admin0_fallback], ignore_index=True)
    .rename(columns={"adm_0_name": "Country", "dengue_total": "Cases"})
    .drop(columns="_source")
)

countries = sorted(annual["Country"].unique())
n = len(countries)
colors = cm.tab20(np.linspace(0, 1, n))

fig, ax = plt.subplots(figsize=(14, 7))

for country, color in zip(countries, colors):
    cdf = annual[annual["Country"] == country].sort_values("Year")
    ax.plot(cdf["Year"], cdf["Cases"], marker="o", markersize=3,
            linewidth=1.4, label=country.title(), color=color)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Annual Dengue Cases", fontsize=12)
ax.set_title("Annual Dengue Infection Counts by Country (INPAC v2)", fontsize=13)
ax.set_yscale("log")
ax.set_xlim(annual["Year"].min() - 0.5, annual["Year"].max() + 0.5)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.tick_params(axis="x", rotation=45)
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    fontsize=8,
    framealpha=0.9,
    ncol=1,
)

plt.tight_layout()
out_file = OUT_PATH / "country_infection_trends.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight")
print(f"Saved: {out_file}")
plt.show()
