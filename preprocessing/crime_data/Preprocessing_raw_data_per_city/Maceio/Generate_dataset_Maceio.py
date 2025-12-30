import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
import glob

# Get all CSV files in the data folder
data_folder = './data'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))

# Read and combine all CSV files
df_list = []
for csv_file in csv_files:
    print(f"Reading {csv_file}...")
    df_temp = pd.read_csv(csv_file, delimiter=";", low_memory=False)
    df_list.append(df_temp.query("CIDADE_FATO == 'Arapiraca'"))

# Concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)

print("Size dataframe after filtering for Arapiraca: ", df.shape)
