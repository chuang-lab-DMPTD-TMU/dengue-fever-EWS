import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/filtered_sea_2011_2018_SLVC_imputed.csv", low_memory=False)
df["calendar_start_date"] = pd.to_datetime(df["calendar_start_date"])

# Full monthly grid: Jan 2008 – Dec 2020
all_months = pd.date_range("2008-01-01", "2020-12-01", freq="MS")

COUNTRIES = [
    "MALAYSIA",
    "INDONESIA",
    "VIET NAM",
    "THAILAND",
    "PHILIPPINES",
    "SINGAPORE",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "CAMBODIA",
    "MYANMAR",
    "TIMOR-LESTE",
    "BRUNEI DARUSSALAM",
]

DISPLAY_NAMES = {
    "CAMBODIA": "Cambodia",
    "INDONESIA": "Indonesia",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Laos",
    "MALAYSIA": "Malaysia",
    "PHILIPPINES": "Philippines",
    "SINGAPORE": "Singapore",
    "THAILAND": "Thailand",
    "VIET NAM": "Viet Nam",
    "TIMOR-LESTE": "Timor-Leste",
    "BRUNEI DARUSSALAM": "Brunei",
    "MYANMAR": "Myanmar",
}

RESOLUTION_ALIASES = {
    "Week": "Weekly",
    "week": "Weekly",
    "Weekly": "Weekly",
}
RESOLUTION_PRIORITY = {"Year": 1, "Month": 2, "Weekly": 3}
RESOLUTION_ORDER = ["Missing", "Year", "Month", "Weekly"]
RESOLUTION_COLORS = {
    "Missing": "#FFFFFF",
    "Year": "#AED6F1",
    "Month": "#2980B9",
    "Weekly": "#1A2D6B",
}
MAX_RESOLUTION_INDEX = len(RESOLUTION_ORDER) - 1 if len(RESOLUTION_ORDER) > 1 else 1
RESOLUTION_COLORSCALE = [
    [i / MAX_RESOLUTION_INDEX, RESOLUTION_COLORS[name]]
    for i, name in enumerate(RESOLUTION_ORDER)
]
RESOLUTION_CODES = {name: idx for idx, name in enumerate(RESOLUTION_ORDER)}


def normalize_resolution(value):
    """Normalize temporal frequency labels to the canonical set so we can color-code them."""
    if pd.isna(value):
        return value
    label = str(value).strip()
    if not label:
        return value
    return RESOLUTION_ALIASES.get(label, label.title())


# ── province name normalisation (fixes spelling variants in source data) ─────
PROVINCE_ALIASES = {
    "MYANMAR": {
        "AYAYARWADDY": "AYEYARWADY",
        "BAGO (E)": "BAGO (EAST)",
        "BAGO (W": "BAGO (WEST)",
        "NAYPYITAW": "NAY PYI TAW",
        "SHAN (N)": "SHAN (NORTH)",
        "SHAN (S)": "SHAN (SOUTH)",
        "MONGAR": None,  # Bhutan district misattributed to Myanmar — drop
    }
}

# ── pre-process Philippines: collapse adm2 → adm1 by summing dengue_total ────
ph_mask = df["adm_0_name"] == "PHILIPPINES"
ph_collapsed = (
    df[ph_mask]
    .groupby(["adm_0_name", "adm_1_name", "calendar_start_date"], as_index=False)
    .agg(dengue_total=("dengue_total", "sum"), T_res=("T_res", "first"))
)
df = pd.concat([df[~ph_mask], ph_collapsed], ignore_index=True)

# ── build per-country pivot: rows=province, cols=month, value=preferred temporal resolution ────
country_data = {}
for country in COUNTRIES:
    sub = df[df["adm_0_name"] == country].copy()

    # Apply province name aliases (fix spelling variants, drop bad rows)
    if country in PROVINCE_ALIASES:
        aliases = PROVINCE_ALIASES[country]
        drop_names = {k for k, v in aliases.items() if v is None}
        rename_map = {k: v for k, v in aliases.items() if v is not None}
        sub = sub[~sub["adm_1_name"].isin(drop_names)]
        sub["adm_1_name"] = sub["adm_1_name"].replace(rename_map)

    # Drop national-level rows (NaN adm_1_name) when province data exists
    has_provinces = sub["adm_1_name"].notna().any()
    if has_provinces:
        sub = sub[sub["adm_1_name"].notna()]
    else:
        sub["adm_1_name"] = sub["adm_1_name"].fillna("Unknown Province (Likely national)")

    provinces = sorted(sub["adm_1_name"].unique())

    resolution_rows = sub[["adm_1_name", "calendar_start_date", "T_res"]].copy()
    resolution_rows["T_res"] = resolution_rows["T_res"].apply(normalize_resolution)
    resolution_rows["rank"] = resolution_rows["T_res"].map(RESOLUTION_PRIORITY).fillna(0)

    # Expand yearly entries to fill all 12 months of that year so the grid
    # shows a continuous block. Finer-resolution rows (Month / Weekly) added
    # later win via the rank sort + dedup below.
    yearly_mask = resolution_rows["T_res"] == "Year"
    if yearly_mask.any():
        yearly_rows = resolution_rows[yearly_mask]
        expanded = []
        for _, row in yearly_rows.iterrows():
            year = row["calendar_start_date"].year
            for month in range(1, 13):
                new_row = row.copy()
                new_row["calendar_start_date"] = pd.Timestamp(year=year, month=month, day=1)
                expanded.append(new_row)
        resolution_rows = pd.concat(
            [pd.DataFrame(expanded), resolution_rows[~yearly_mask]],
            ignore_index=True,
        )

    resolution_rows = (
        resolution_rows.sort_values(
            ["adm_1_name", "calendar_start_date", "rank"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["adm_1_name", "calendar_start_date"], keep="first")
    )

    type_grid = (
        resolution_rows.pivot(
            index="adm_1_name", columns="calendar_start_date", values="T_res"
        )
        .reindex(columns=all_months)
        .reindex(provinces)
    )

    presence = type_grid.notna()
    country_data[country] = (presence, all_months, type_grid)


# ── summary bar (% of expected months covered) ───────────────────────────────
summary_rows = []
for country, (presence, months, _) in country_data.items():
    n_provinces = len(presence)
    expected = n_provinces * len(months)
    observed = int(presence.values.sum())
    summary_rows.append(
        {
            "country": DISPLAY_NAMES[country],
            "provinces": n_provinces,
            "coverage_pct": 100 * observed / expected if expected > 0 else 0,
        }
    )
summary_df = pd.DataFrame(summary_rows).sort_values("coverage_pct", ascending=True)


# ── layout: summary bar + one heatmap per country ────────────────────────────
n_countries = len(COUNTRIES)
row_heights = [0.12] + [max(0.04, len(country_data[c][0]) / 350) for c in COUNTRIES]
total = sum(row_heights)
row_heights = [h / total for h in row_heights]

fig = make_subplots(
    rows=n_countries + 1,
    cols=1,
    subplot_titles=["Overall Coverage (% of expected province-months)"]
    + [DISPLAY_NAMES[c] for c in COUNTRIES],
    row_heights=row_heights,
    vertical_spacing=0.02,
)

# — summary bar —
fig.add_trace(
    go.Bar(
        x=summary_df["coverage_pct"],
        y=summary_df["country"],
        orientation="h",
        marker_color="#2196F3",
        text=[f"{v:.1f}%" for v in summary_df["coverage_pct"]],
        textposition="outside",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.update_xaxes(range=[0, 110], row=1, col=1, title_text="% covered")

# — heatmap per country —
for i, country in enumerate(COUNTRIES, start=2):
    _, months, type_grid = country_data[country]
    provinces = list(type_grid.index)
    display_grid = type_grid.fillna("Missing")
    display_grid = display_grid.where(display_grid.isin(RESOLUTION_ORDER), "Missing")
    z = display_grid.replace(RESOLUTION_CODES).values
    resolution_labels = display_grid.values

    year_ticks = [m for m in months if m.month == 1]
    year_tick_indices = [list(months).index(m) for m in year_ticks]
    year_tick_labels = [str(m.year) for m in year_ticks]

    text_dates = [[m.strftime("%b %Y") for m in months] for _ in provinces]
    heatmap_kwargs = dict(
        z=z,
        x=list(range(len(months))),
        y=provinces,
        colorscale=RESOLUTION_COLORSCALE,
        zmin=0,
        zmax=len(RESOLUTION_ORDER) - 1,
        showscale=i == 2,
        hovertemplate="<b>%{y}</b><br>%{text}<br>Resolution: %{customdata}<extra></extra>",
        text=text_dates,
        customdata=resolution_labels.tolist(),
    )
    if i == 2:
        heatmap_kwargs["colorbar"] = dict(
            tickmode="array",
            tickvals=list(range(len(RESOLUTION_ORDER))),
            ticktext=RESOLUTION_ORDER,
            lenmode="fraction",
            len=0.75,
        )
    fig.add_trace(
        go.Heatmap(**heatmap_kwargs),
        row=i,
        col=1,
    )
    fig.update_xaxes(
        tickvals=year_tick_indices,
        ticktext=year_tick_labels,
        tickangle=45,
        row=i,
        col=1,
    )
    fig.update_yaxes(
        tickfont=dict(size=9),
        row=i,
        col=1,
    )

total_height = 300 + sum(max(120, len(country_data[c][0]) * 14) for c in COUNTRIES)

fig.update_layout(
    title=dict(
        text="OpenDengue SEA — Province Coverage & Temporal Resolution (2008–2020)",
        font=dict(size=16),
    ),
    height=total_height,
    width=1400,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=200, r=60, t=80, b=40),
)

fig.write_html("coverage_sea.html")
print("Saved: coverage_sea.html")