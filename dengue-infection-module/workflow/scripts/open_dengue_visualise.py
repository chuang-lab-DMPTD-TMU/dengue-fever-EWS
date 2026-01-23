# %%
# Import basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import os 
from pathlib import Path
import sys
from tqdm import tqdm
import pycwt as wavelet
from pycwt import wct_significance
from pycwt import helpers
import matplotlib.dates as mdates
import scipy
from scipy.signal import detrend
from matplotlib.ticker import LogLocator, FormatStrFormatter

# Load dengue case data
global_dengue_national = pd.read_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/work/global/National_extract_V1_3.csv")
global_dengue_temporal = pd.read_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/work/global/Temporal_extract_V1_3.csv")
global_dengue_spatial = pd.read_csv("/home/patwuch/Documents/projects/Chuang-Lab-TMU/dengue-infection-module/work/global/Spatial_extract_V1_3.csv")

print(global_dengue_national.head())
print(global_dengue_temporal.head())
print(global_dengue_spatial.head())

print