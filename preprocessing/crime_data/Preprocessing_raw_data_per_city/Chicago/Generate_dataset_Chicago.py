import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
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

df = pd.read_csv('raw_data/Crimes_-_2001_to_Present_5ys.csv',low_memory=False)
print("Size dataframe initially: ", df.shape)

# change type of Date column from object to datetime
df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y %I:%M:%S %p') # we mark as NaT dates that are too old given datetime lowerbound

# make compatible
df.rename(columns={'Date': 'crime_date_time', 'Primary Type':'crime_type','Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)


# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# select dates for 2019 to 2023
lower_bound = "2019/01/01"
upper_bound = "2024/01/01"
df_years = df.copy()
df_years = df_years.loc[(df_years["crime_date_time"] >= lower_bound) & (df_years["crime_date_time"] < upper_bound)]
print("Shape dataframe after selecting crimes from 2019 to 2023 incl.", df_years.shape)

# reset index
df_years.reset_index(drop=True,inplace=True)

# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../../city_multipolygons.geojson',city_name='Chicago')

# remove points that aren't within the multipolygon
df_clean = df_years.copy()
for i, entry in df_years.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df_years.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

# save final dataset
df_clean.to_csv("Chicago_crimes_clean_all_5ys.csv")
print("Shape final dataframe: ", df_clean.shape) # takes around 57 min to run
