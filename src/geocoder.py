import pandas as pd
from geopy.geocoders import ArcGIS
import time
import os

def enrich_anomalies_with_coords():
    print("🛰️ [Geocoder] Initializing Satellite Link (ArcGIS)...")
    
    # Load the scored data
    input_path = "data/final_scored_data.csv"
    if not os.path.exists(input_path):
        print("❌ Data not found. Run engine.py first.")
        return

    df = pd.read_csv(input_path)
    
    # Filter for Critical Anomalies ONLY
    critical_mask = df['risk_status'] == 'CRITICAL'
    critical_df = df[critical_mask].copy()
    
    print(f"📍 Found {len(critical_df)} Critical Anomalies to geolocate.")
    print("   Fetching exact coordinates... (This uses ArcGIS, it's faster but give it time)")

    # SWITCH TO ARCGIS (More robust than Nominatim)
    geolocator = ArcGIS(timeout=10) # 10-second timeout to prevent errors

    # Function to fetch coords
    def get_lat_long(row):
        try:
            # Query format: "110001, India" (Simple is often better for Pincodes)
            query = f"{row['pincode']}, India"
            location = geolocator.geocode(query)
            
            if location:
                return pd.Series([location.latitude, location.longitude])
            else:
                # Fallback: Try with District
                query_district = f"{row['district']}, {row['state']}, India"
                location = geolocator.geocode(query_district)
                if location:
                    return pd.Series([location.latitude, location.longitude])
                
                return pd.Series([None, None])
        except Exception as e:
            print(f"   ⚠️ Error locating {row['pincode']}: {e}")
            return pd.Series([None, None])

    # Apply Geocoding
    # We do it in a loop to show progress bar-like output
    total = len(critical_df)
    for index, row in critical_df.iterrows():
        print(f"   Searching: {row['pincode']}...", end="\r")
        res = get_lat_long(row)
        critical_df.at[index, 'real_lat'] = res[0]
        critical_df.at[index, 'real_lon'] = res[1]
        
    print("\n   Merging coordinates...")
    
    # Update the original DF
    df['lat'] = 0.0
    df['lon'] = 0.0
    df['geo_accuracy'] = 'Low'
    
    for index, row in critical_df.iterrows():
        if not pd.isna(row['real_lat']):
            df.at[index, 'lat'] = row['real_lat']
            df.at[index, 'lon'] = row['real_lon']
            df.at[index, 'geo_accuracy'] = 'High'

    # Save
    output_path = "data/final_scored_data_geocoded.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Geocoding Complete. Saved to {output_path}")

if __name__ == "__main__":
    enrich_anomalies_with_coords()
    