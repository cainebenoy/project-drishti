import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import os

def enrich_anomalies_with_coords():
    print("🛰️ [Geocoder] Initializing Satellite Link (Nominatim)...")
    
    # Load the scored data
    input_path = "data/final_scored_data.csv"
    if not os.path.exists(input_path):
        print("❌ Data not found. Run engine.py first.")
        return

    df = pd.read_csv(input_path)
    
    # Filter for Critical Anomalies ONLY (To save time/API limits)
    # We will only pinpoint the threats. 
    critical_mask = df['risk_status'] == 'CRITICAL'
    critical_df = df[critical_mask].copy()
    
    print(f"📍 Found {len(critical_df)} Critical Anomalies to geolocate.")
    print("   Fetching exact coordinates... (This may take 2-3 minutes)")

    # Initialize Geocoder
    geolocator = Nominatim(user_agent="project_drishti_hackathon_v1")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    # Function to fetch coords
    def get_lat_long(row):
        try:
            # Query format: "110001, Delhi, India"
            query = f"{row['pincode']}, {row['district']}, India"
            location = geocode(query)
            if location:
                return pd.Series([location.latitude, location.longitude])
            else:
                # Fallback: Try just Pincode + India
                location = geocode(f"{row['pincode']}, India")
                if location:
                    return pd.Series([location.latitude, location.longitude])
                return pd.Series([None, None])
        except Exception as e:
            print(f"   ⚠️ Error locating {row['pincode']}: {e}")
            return pd.Series([None, None])

    # Apply Geocoding
    # We use a progress indicator logic here implicitly by the speed
    cols = critical_df.apply(get_lat_long, axis=1)
    critical_df[['real_lat', 'real_lon']] = cols

    # Merge back into main dataframe
    # We left join so non-critical rows get NaN (which is fine, we filter them in UI)
    print("   Merging coordinates...")
    
    # Update the original DF with these new real coordinates
    # We iterate and update to ensure precision
    df['lat'] = 0.0
    df['lon'] = 0.0
    
    for index, row in critical_df.iterrows():
        if not pd.isna(row['real_lat']):
            df.at[index, 'lat'] = row['real_lat']
            df.at[index, 'lon'] = row['real_lon']
            # Flag that this row has high-precision coords
            df.at[index, 'geo_accuracy'] = 'High'
        else:
             df.at[index, 'geo_accuracy'] = 'Low'

    # Save
    output_path = "data/final_scored_data_geocoded.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Geocoding Complete. Saved to {output_path}")

if __name__ == "__main__":
    enrich_anomalies_with_coords()