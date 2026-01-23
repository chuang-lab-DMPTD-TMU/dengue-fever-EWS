# Import basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os 
from pathlib import Path

# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Load in dengue data

# dengue = pd.read_csv(RAW_DIR)
# print(RAW_DIR)

'''Import CSV Files of Basic Statistics Area and their Population'''



# Walk through the BSA demographic directory to find all CSV files
for root, dirs, files in os.walk(RAW_DIR/'BSA_population_TN'):
    for file in files:
        if file.endswith(".csv"):
            # Extract the first number from the filename
            number = int(''.join(filter(str.isdigit, file.split("年")[0])))  # Get the number before '年'
            year = number + 1911  # Add 1911
            
            # Full path to the CSV file
            file_path = os.path.join(root, file)
            
            # Attempt to load the CSV file with both encodings
            for encoding in ["utf-8", "big5"]:
                try:
                    df_name = f"df_{year}"
                    globals()[df_name] = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded {file} as {df_name} using {encoding} encoding.")
                    break  # Exit the loop if successfully loaded
                except Exception as e:
                    print(f"Failed to load {file} with {encoding} encoding. Error: {e}")
