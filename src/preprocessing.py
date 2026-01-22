import pandas as pd
import numpy as np
import os

class DataEngine:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        
    def load_and_merge_data(self):
        print("⚡ [DataEngine] Scanning data folder...")
        
        # 1. Initialize empty containers
        enrol_dfs = []
        bio_dfs = []
        demo_dfs = []

        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"❌ Data folder '{self.data_path}' not found!")

        for file in os.listdir(self.data_path):
            path = os.path.join(self.data_path, file)
            try:
                if 'enrolment' in file:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    enrol_dfs.append(df)
                elif 'biometric' in file:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    bio_dfs.append(df)
                elif 'demographic' in file:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    demo_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Warning: Could not read {file}. Error: {e}")

        # 3. Concatenate chunks
        print("⚡ [DataEngine] Combining chunks...")
        df_enrol = pd.concat(enrol_dfs, ignore_index=True) if enrol_dfs else pd.DataFrame()
        df_bio = pd.concat(bio_dfs, ignore_index=True) if bio_dfs else pd.DataFrame()
        df_demo = pd.concat(demo_dfs, ignore_index=True) if demo_dfs else pd.DataFrame()

        # --- NEW: TIME SERIES EXTRACTION ---
        print("⚡ [DataEngine] Extracting Time-Series for Forecasting...")
        
        # We need a master timeline of 'Total Updates' (Bio + Demo)
        # 1. Standardize Dates
        for df in [df_bio, df_demo]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

        # 2. Group Bio by Date
        bio_ts = df_bio.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
        bio_ts['total_updates'] = bio_ts['bio_age_5_17'] + bio_ts['bio_age_17_']
        
        # 3. Group Demo by Date
        demo_ts = df_demo.groupby('date')[['demo_age_5_17', 'demo_age_17_']].sum().reset_index()
        demo_ts['total_updates'] = demo_ts['demo_age_5_17'] + demo_ts['demo_age_17_']

        # 4. Merge and Sum
        # Rename for Prophet (ds = date, y = value)
        ts_master = pd.merge(bio_ts[['date', 'total_updates']], demo_ts[['date', 'total_updates']], on='date', how='outer', suffixes=('_bio', '_demo')).fillna(0)
        ts_master['y'] = ts_master['total_updates_bio'] + ts_master['total_updates_demo']
        ts_master = ts_master.rename(columns={'date': 'ds'})
        ts_master = ts_master.sort_values('ds')

        # Save Time Series Data
        ts_master.to_csv("data/daily_timeseries.csv", index=False)
        print("💾 Saved Time-Series Data to 'data/daily_timeseries.csv'")

        # --- EXISTING: PINCODE AGGREGATION ---
        print("⚡ [DataEngine] Aggregating vectors for Anomaly Detection...")
        
        enrol_grouped = df_enrol.groupby('pincode').agg({'age_0_5': 'sum', 'age_18_greater': 'sum', 'state': 'first', 'district': 'first'}).reset_index()
        bio_cols = [c for c in df_bio.columns if 'bio_age' in c]
        bio_grouped = df_bio.groupby('pincode')[bio_cols].sum().reset_index()
        demo_cols = [c for c in df_demo.columns if 'demo_age' in c]
        demo_grouped = df_demo.groupby('pincode')[demo_cols].sum().reset_index()

        master_df = pd.merge(enrol_grouped, bio_grouped, on='pincode', how='outer').fillna(0)
        master_df = pd.merge(master_df, demo_grouped, on='pincode', how='outer').fillna(0)

        total_activity = master_df['age_0_5'] + master_df['age_18_greater'] + 1 
        master_df['adult_spike_ratio'] = master_df['age_18_greater'] / total_activity
        
        bio_adult_col = [c for c in master_df.columns if 'bio' in c and '17' in c][0]
        demo_adult_col = [c for c in master_df.columns if 'demo' in c and '17' in c][0]
        
        master_df['velocity_index'] = master_df[bio_adult_col] + master_df[demo_adult_col]
        master_df['ghost_ratio'] = master_df[demo_adult_col] / (master_df[bio_adult_col] + 1)

        print(f"✅ [DataEngine] Pipeline Complete. Processed {len(master_df)} unique pincodes.")
        return master_df

if __name__ == "__main__":
    eng = DataEngine()
    df = eng.load_and_merge_data()
    df.to_csv("data/master_processed_data.csv", index=False)