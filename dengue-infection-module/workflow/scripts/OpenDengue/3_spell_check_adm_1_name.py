import csv

replacements = {
    'AYAYARWADDY':              'AYEYARWADY',
    'BAGO (E)':                 'BAGO (EAST)',
    'BAGO (W':                  'BAGO (WEST)',
    'D.I YOGYA':                'DAERAH ISTIMEWA YOGYAKARTA',
    'KALIMANTAN SELATA':        'KALIMANTAN SELATAN',
    'KEPULAUAN-RIAU':           'KEPULAUAN RIAU',
    'NAYPYITAW':                'NAY PYI TAW',
    'NUSATENGGARA BARAT':       'NUSA TENGGARA BARAT',
    'NUSATENGGARA TIMUR':       'NUSA TENGGARA TIMUR',
    'SULAWESI SELATA':          'SULAWESI SELATAN',
    'SUMATERA SELATA':          'SUMATERA SELATAN',
    'SHAN (N)':                 'SHAN (NORTH)',
    'SHAN (S)':                 'SHAN (SOUTH)',
    'REGION 4':                 'REGION IV-A (CALABARZON)',
}

filepath = '/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/main/interim/OpenDengue/filtered_sea_2011_2018.csv'

with open(filepath, newline='') as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

changed = 0
for row in rows:
    old = row['adm_1_name']
    if old in replacements:
        row['adm_1_name'] = replacements[old]
        changed += 1

with open(filepath, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

unique_names = sorted(set(row['adm_1_name'] for row in rows))

print(f'Done. {changed} rows updated.')
print(f'\n{len(unique_names)} unique adm_1_name values after correction:')
for name in unique_names:
    print(f'  {name}')