import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
import glob
import json

def extract_multipolygon_city(file_path,city_name):
    '''
    Extracts the entry in the geojson file corresponding to the city selected and outputs the
    corresponding geodataframe with the multipolygon.

        Parameters:
            file_path (str): File path to the geojson
            city_name (str): Name of the city we selected

        Returns:
            feature (geopandas): The geopandas dataframe for that city
    '''
    # Read the geojson file - it's an array of FeatureCollections
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Iterate through each FeatureCollection in the array
    for feature_collection in data:
        if feature_collection.get("type") == "FeatureCollection":
            # Check each feature in the FeatureCollection
            for feature in feature_collection.get("features", []):
                if feature.get("properties", {}).get("city") == city_name:
                    return gpd.GeoDataFrame.from_features([feature], crs="EPSG:4326")
    
    raise ValueError(f"City '{city_name}' not found in {file_path}")

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
print("Size dataframe after filtering for Arapiraca and combining all CSV files: ", df.shape)

# Create datetime column from DATA_HORA_FATO
# Format is "D/M/YYYY HH:MM:SS" (Brazilian format: day/month/year)
# Using dayfirst=True to handle day/month/year format correctly
df['crime_date_time'] = pd.to_datetime(df['DATA_HORA_FATO'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
# Drop rows with invalid dates
df.dropna(subset='crime_date_time', inplace=True)
print("\nSize dataframe after creating datetime column and removing invalid dates: ", df.shape)

# make compatible
df.rename(columns={'NATUREZA_FATO':'crime_type','LATITUDE': 'latitude', 'LONGITUDE': 'longitude'}, inplace=True)

# Convert latitude and longitude from string (with comma as decimal separator) to float
# Brazilian format uses comma instead of dot for decimals
df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)

# Drop rows with invalid coordinates (NaN or out of range)
df = df.dropna(subset=['latitude', 'longitude'])
# Basic validation: latitude should be between -90 and 90, longitude between -180 and 180
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
        (df['longitude'] >= -180) & (df['longitude'] <= 180)]

# keep only relevant columns
df = df[['crime_date_time', 'crime_type', 'latitude', 'longitude']]

# TODO: aggregate and maybe transform the crime types?

# select dates for 2012 to 2020 (outside pandemic)
lower_bound = "2012/01/01"
upper_bound = "2020/01/01"
df_years = df.copy()
df_years = df_years.loc[(df_years["crime_date_time"] >= lower_bound) & (df_years["crime_date_time"] < upper_bound)]
print("Shape dataframe after selecting crimes from 2019 to 2023 incl.", df_years.shape)

# reset index
df_years.reset_index(drop=True,inplace=True)

gdf = extract_multipolygon_city(file_path='./city_multipolygons.geojson', city_name='Arapiraca')

# remove points that aren't within the multipolygon
df_clean = df_years.copy()
for i, entry in df_years.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df_years.index[i])
df_clean.reset_index(drop=True,inplace=True)

# save final dataset
df_clean.to_csv("./preprocessing/crime_data/Preprocessing_raw_data_per_city/Maceio/arapiraca_crimes_clean_2012_to_2020.csv")
print("Shape final dataframe after geo cleaning: ", df_clean.shape)