import pandas as pd

SEA_COUNTRIES = [
    "MALAYSIA",
    "INDONESIA",
    "VIET NAM",
    "THAILAND",
    "PHILIPPINES",
    "SINGAPORE",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
    "CAMBODIA",
    'MYANMAR',
    'BRUNEI DARUSSALAM'
]

df = pd.read_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/spatial_SEARO_WPRO_EMRO_2000_2025.csv")
filtered = df[df["adm_0_name"].isin(SEA_COUNTRIES)]
filtered["calendar_start_date"] = pd.to_datetime(filtered["calendar_start_date"])

filtered = filtered[
    (filtered['calendar_start_date'] >= pd.to_datetime("2011-01-01")) &
    (filtered['calendar_start_date'] <= pd.to_datetime("2018-12-31"))
]
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered)}")
print(f"Countries found: {sorted(filtered['adm_0_name'].unique())}")

filtered.to_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/filtered_sea_2011_2018.csv", index=False)
