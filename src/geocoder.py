import pandas as pd
from geopy.geocoders import ArcGIS
import os
import time

def enrich_anomalies_with_coords():
    print("🛰️ [Geocoder] Connecting to ArcGIS Satellite...")
    
    # 1. Load the scored data
    input_path = "data/final_scored_data.csv"
    if not os.path.exists(input_path):
        print("❌ Data not found. Please run 'python src/engine.py' first.")
        return

    df = pd.read_csv(input_path)
    
    # 2. Filter for Critical Anomalies ONLY
    # We only geocode the high-risk items to save time/resources
    critical_mask = df['risk_status'] == 'CRITICAL'
    critical_df = df[critical_mask].copy()
    
    print(f"📍 Found {len(critical_df)} Critical Anomalies to geolocate.")
    print("   Fetching exact coordinates... (This uses ArcGIS)")

    # 3. Initialize Geocoder (ArcGIS is robust & free for this use)
    geolocator = ArcGIS(timeout=10)

    # Function to fetch coords
    def get_lat_long(row):
        try:
            # Query 1: Try Pincode + India (Most Accurate)
            query = f"{row['pincode']}, India"
            res = geolocator.geocode(query)
            
            # Query 2: Fallback to District + State (If Pincode fails)
            if not res:
                query_fallback = f"{row['district']}, {row['state']}, India"
                res = geolocator.geocode(query_fallback)
            
            if res:
                return pd.Series([res.latitude, res.longitude])
            else:
                return pd.Series([None, None])
        except Exception as e:
            # Silently fail on individual errors to keep the loop moving
            return pd.Series([None, None])

    # 4. Apply Geocoding
    # Using apply with the function above
    cols = critical_df.apply(get_lat_long, axis=1)
    critical_df[['real_lat', 'real_lon']] = cols

    # 5. Merge coordinates back into the main dataset
    print("   Merging satellite data...")
    
    # Initialize default columns
    df['lat'] = 0.0
    df['lon'] = 0.0
    df['geo_accuracy'] = 'Low'
    
    # Update only the rows we found
    for index, row in critical_df.iterrows():
        if not pd.isna(row['real_lat']):
            df.at[index, 'lat'] = row['real_lat']
            df.at[index, 'lon'] = row['real_lon']
            df.at[index, 'geo_accuracy'] = 'High'

    # 6. Save the enriched dataset
    output_path = "data/final_scored_data_geocoded.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Geocoding Complete. High-precision data saved to '{output_path}'")

if __name__ == "__main__":
    enrich_anomalies_with_coords()