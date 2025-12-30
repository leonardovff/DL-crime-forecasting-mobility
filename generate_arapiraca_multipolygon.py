"""
Script to generate multipolygon for Arapiraca and add it to city_multipolygons.geojson

This script uses OSMnx to fetch the city boundary from OpenStreetMap
and adds it to the existing geojson file in the correct format.

Requirements:
    - osmnx: pip install osmnx
    - geopandas: pip install geopandas
    - shapely: pip install shapely
"""

import json
import os
import geopandas as gpd
from shapely.geometry import mapping

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("Warning: osmnx is not installed. Install it with: pip install osmnx")

def get_city_boundary_osmnx(city_name, state=None, country="Brazil"):
    """
    Fetch city boundary using OSMnx (recommended method).
    
    Parameters:
        city_name (str): Name of the city
        state (str): Optional state name
        country (str): Country name
    
    Returns:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with city boundary
    """
    if not OSMNX_AVAILABLE:
        raise ImportError("osmnx is required. Install it with: pip install osmnx")
    
    # Build query string
    if state:
        query = f"{city_name}, {state}, {country}"
    else:
        query = f"{city_name}, {country}"
    
    print(f"Fetching boundary for: {query}")
    
    try:
        # Get boundary using OSMnx
        gdf = ox.geocode_to_gdf(query)
        
        if gdf.empty:
            raise ValueError(f"No boundary found for {query}")
        
        print(f"Successfully fetched boundary for {city_name}")
        print(f"Boundary shape: {gdf.geometry.iloc[0].geom_type}")
        print(f"CRS: {gdf.crs}")
        
        # Ensure CRS is WGS84 (EPSG:4326)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        return gdf
    except Exception as e:
        print(f"Error fetching with OSMnx: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the city name and state are correct")
        print("3. Try a more specific query (e.g., include state name)")
        return None


def convert_to_multipolygon_feature(gdf, city_name, existing_count=0):
    """
    Convert GeoDataFrame to the format used in city_multipolygons.geojson
    
    Parameters:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with city boundary
        city_name (str): Name of the city
        existing_count (int): Number of existing features in the file
    
    Returns:
        feature_dict (dict): Feature dictionary in the correct format
    """
    from shapely.geometry import MultiPolygon
    
    # Get the geometry
    geometry = gdf.geometry.iloc[0]
    
    # Convert to MultiPolygon if it's a Polygon
    if geometry.geom_type == 'Polygon':
        geometry = MultiPolygon([geometry])
    elif geometry.geom_type != 'MultiPolygon':
        raise ValueError(f"Geometry type {geometry.geom_type} not supported. Expected Polygon or MultiPolygon.")
    
    # Convert to GeoJSON format
    geojson_geom = mapping(geometry)
    
    # Create feature in the same format as the existing file
    feature = {
        "type": "Feature",
        "geometry": geojson_geom,
        "properties": {"city": city_name},
        "id": str(existing_count)
    }
    
    return feature

def add_city_to_geojson(city_name, geojson_path='city_multipolygons.geojson', state=None):
    """
    Main function to fetch city boundary and add it to the geojson file.
    
    Parameters:
        city_name (str): Name of the city (e.g., "Arapiraca")
        geojson_path (str): Path to the geojson file
        state (str): Optional state name (e.g., "Alagoas")
    """
    import os
    
    # Try to fetch boundary
    gdf = get_city_boundary_osmnx(city_name, state=state)
    
    if gdf is None or gdf.empty:
        print("Failed to fetch boundary. Please check:")
        print("1. City name is correct")
        print("2. Internet connection is available")
        print("3. OSMnx is properly installed")
        return None
    
    # Read existing geojson to count features
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            existing_data = json.load(f)
        existing_count = len(existing_data)
    else:
        existing_data = []
        existing_count = 0
    
    # Convert to feature format
    feature = convert_to_multipolygon_feature(gdf, city_name, existing_count=existing_count)
    
    # Create new FeatureCollection for this city
    new_feature_collection = {
        "type": "FeatureCollection",
        "features": [feature]
    }
    
    # Add to existing data
    existing_data.append(new_feature_collection)
    
    # Write back to file
    with open(geojson_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"\nSuccessfully added {city_name} to {geojson_path}")
    print(f"Feature ID: {feature['id']}")
    
    return feature

if __name__ == "__main__":
    import os
    
    # Generate multipolygon for Arapiraca
    # Arapiraca is in Alagoas state, Brazil
    city_name = "Arapiraca"
    state = "Alagoas"
    
    print("=" * 60)
    print(f"Generating multipolygon for {city_name}, {state}, Brazil")
    print("=" * 60)
    
    feature = add_city_to_geojson(city_name, state=state)
    
    if feature:
        print("\n✓ Success! The multipolygon has been added to city_multipolygons.geojson")
        print("\nYou can now use it in your preprocessing scripts with:")
        print(f'  extract_multipolygon_city(file_path="city_multipolygons.geojson", city_name="{city_name}")')
    else:
        print("\n✗ Failed to generate multipolygon. Please check the error messages above.")

