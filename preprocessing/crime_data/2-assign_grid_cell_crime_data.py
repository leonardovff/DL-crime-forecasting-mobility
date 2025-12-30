from shapely.geometry import Point
import shapely.geometry
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from IPython.display import display
from pandarallel import pandarallel
from pathlib import Path
import os
import json

# Ensure at least 1 worker, handle case where os.cpu_count() returns None or 0
cpu_count = os.cpu_count() or 1
nb_workers = max(1, min(cpu_count, 12))
pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)

"""### Useful functions"""

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

def generate_grid(selected_city,grid_size,plot=True):
  gdf = extract_multipolygon_city(file_path='./city_multipolygons.geojson',city_name=selected_city)

  bbox = gdf.total_bounds
  min_lon, min_lat, max_lon, max_lat = (bbox[2],bbox[1],bbox[0],bbox[3])

  lon = np.linspace(min_lon, max_lon, grid_size+1)
  lat = np.linspace(min_lat, max_lat, grid_size+1)

  latlons = []
  for i in range(len(lat)-1):
      for k in range(len(lon)-1):
          latlons.append((lat[k], lon[i], lat[k+1], lon[i+1]))

  if plot==True:
    m = folium.Map(location=((min_lat+max_lat)/2,(min_lon+max_lon)/2), zoom_start=11)
    for k in latlons:
        folium.Rectangle([(k[0], k[1]), (k[2], k[3])],
                        color='red',
                        fill='pink',
                        fill_opcity=0.5).add_to(m)
    cgeo = (
            gdf.set_crs("epsg:4326")
            .sample(1)
            .pipe(lambda d: d.to_crs(d.estimate_utm_crs()))["geometry"]
            #.centroid.buffer(10000)
            .to_crs("epsg:4326")
            .__geo_interface__
        )

    geo_j = folium.GeoJson(data=cgeo)
    geo_j.add_to(m)
    display(m)
    return (min_lon, min_lat, max_lon, max_lat, latlons)
  else:
    return (min_lon, min_lat, max_lon, max_lat, latlons)

"""### Generate crime files with grid number"""

def find_polygon(row,df):
    for p_idx,elem in df.iterrows():
      if elem['polygon'].contains(Point(row['longitude'],row['latitude'])):
        polygon_num = p_idx
        break
    return pd.Series({'cell': polygon_num})

def generate_crime_files_with_grid_num(city_name,city_folder,crimes_list,output_folder,grid_size=50):

  # load crime dataset for that city
  df_crimes = pd.read_csv(f'Crime_data_outputs/{city_folder}_selected_crimes_clean_all.csv',index_col=0)

  # generate grid boxes
  min_lon, min_lat, max_lon, max_lat, grid_bboxes = generate_grid(selected_city=city_name,grid_size=grid_size,plot=False)

  # turn boxes into polygons and that list into a dataframe
  polygon_list = []
  for elem in grid_bboxes:
    polygon = shapely.geometry.box(elem[1],elem[0],elem[3],elem[2],ccw=True)
    polygon_list.append(polygon)
  df_poly = pd.DataFrame(polygon_list,columns=['polygon'])

  # generate file for each crime
  for crime in crimes_list:
    print("Crime type: ", crime)
    df_crime = df_crimes.copy()
    df_crime = df_crime[df_crime['crime_type'] == crime]
    df_crime.reset_index(drop=True,inplace=True)
    print(f"Shape dataset after selecting only {crime}: ", df_crime.shape)

    print("Finding the grid cell for each point...")
    df_final = df_crime.merge(df_crime.parallel_apply(find_polygon,df=df_poly,axis=1),left_index=True, right_index= True)

    # save final dataset
    os.makedirs(f'Crime_data_outputs/{output_folder}/', exist_ok=True)
    df_final.to_csv(f'Crime_data_outputs/{output_folder}/{city_folder}_{crime}_clean_all_grid.csv')
    print("Final dataset saved!\n")

generate_crime_files_with_grid_num(city_name = 'Arapiraca',
                                   city_folder = 'Arapiraca',
                                   crimes_list = ['Burglary', 'Motor Vehicle Theft', 'Robbery'],
                                   output_folder = 'Grid_cells_0.2gu/Clean_all_grid',
                                   grid_size=39,
                                   )
