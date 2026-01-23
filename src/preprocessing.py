import pandas as pd
import numpy as np
import os

class DataEngine:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        
    def load_and_merge_data(self):
        print("⚡ [DataEngine] Initializing Big Data Pipeline...")
        
        # 1. Initialize empty containers
        enrol_dfs = []
        bio_dfs = []
        demo_dfs = []

        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"❌ Data folder '{self.data_path}' not found! Please create it and add your CSVs.")

        # 2. Smart Loader: Finds files based on keywords in filename
        files_found = 0
        print(f"   📂 Scanning '{self.data_path}' for data chunks...")
        
        for file in os.listdir(self.data_path):
            path = os.path.join(self.data_path, file)
            try:
                # We normalize column names to lowercase/stripped to prevent KeyErrors
                if 'enrolment' in file.lower():
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    enrol_dfs.append(df)
                    print(f"      -> Loaded {file}: {len(df):,} rows")
                    files_found += 1
                elif 'biometric' in file.lower():
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    bio_dfs.append(df)
                    print(f"      -> Loaded {file}: {len(df):,} rows")
                    files_found += 1
                elif 'demographic' in file.lower():
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    demo_dfs.append(df)
                    print(f"      -> Loaded {file}: {len(df):,} rows")
                    files_found += 1
            except Exception as e:
                print(f"⚠️ Warning: Could not read {file}. Error: {e}")

        if files_found == 0:
            raise ValueError("❌ No valid CSV files found in 'data/' folder.")

        # 3. Concatenate chunks
        print("\n⚡ [DataEngine] Merging Data Lakes...")
        df_enrol = pd.concat(enrol_dfs, ignore_index=True) if enrol_dfs else pd.DataFrame()
        df_bio = pd.concat(bio_dfs, ignore_index=True) if bio_dfs else pd.DataFrame()
        df_demo = pd.concat(demo_dfs, ignore_index=True) if demo_dfs else pd.DataFrame()

        # PRINT THE PROOF
        print(f"   📊 TOTAL RAW DATA INGESTED:")
        print(f"      - Enrolment Records:   {len(df_enrol):,}")
        print(f"      - Biometric Updates:   {len(df_bio):,}")
        print(f"      - Demographic Updates: {len(df_demo):,}")
        total_raw = len(df_enrol) + len(df_bio) + len(df_demo)
        print(f"      ==================================")
        print(f"      GRAND TOTAL:           {total_raw:,} rows processed.")

        # --- PART 1: TIME SERIES EXTRACTION (For Prophet Forecasting) ---
        print("\n⚡ [DataEngine] Extracting Time-Series for Forecasting...")
        
        # Standardize Dates
        for df in [df_bio, df_demo]:
            if 'date' in df.columns:
                # Handle DD-MM-YYYY format common in Indian datasets
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

        # Group Bio by Date
        if not df_bio.empty:
            bio_cols = [c for c in df_bio.columns if 'bio_age' in c]
            bio_ts = df_bio.groupby('date')[bio_cols].sum().reset_index()
            bio_ts['total_updates'] = bio_ts[bio_cols].sum(axis=1)
        else:
            bio_ts = pd.DataFrame(columns=['date', 'total_updates'])

        # Group Demo by Date
        if not df_demo.empty:
            demo_cols = [c for c in df_demo.columns if 'demo_age' in c]
            demo_ts = df_demo.groupby('date')[demo_cols].sum().reset_index()
            demo_ts['total_updates'] = demo_ts[demo_cols].sum(axis=1)
        else:
            demo_ts = pd.DataFrame(columns=['date', 'total_updates'])

        # Merge for Prophet
        ts_master = pd.merge(
            bio_ts[['date', 'total_updates']], 
            demo_ts[['date', 'total_updates']], 
            on='date', 
            how='outer', 
            suffixes=('_bio', '_demo')
        ).fillna(0)
        
        ts_master['y'] = ts_master['total_updates_bio'] + ts_master['total_updates_demo']
        ts_master = ts_master.rename(columns={'date': 'ds'})
        ts_master = ts_master.sort_values('ds')

        ts_master.to_csv("data/daily_timeseries.csv", index=False)
        print("💾 Saved Time-Series Data to 'data/daily_timeseries.csv'")

        # --- PART 2: PINCODE AGGREGATION (For Anomaly Detection) ---
        print("\n⚡ [DataEngine] Aggregating vectors for Anomaly Detection...")
        
        # Enrolment Summary
        if not df_enrol.empty:
            enrol_grouped = df_enrol.groupby('pincode').agg({
                'age_0_5': 'sum', 
                'age_18_greater': 'sum', 
                'state': 'first', 
                'district': 'first'
            }).reset_index()
        else:
            enrol_grouped = pd.DataFrame(columns=['pincode'])

        # Biometric Summary
        if not df_bio.empty:
            bio_cols = [c for c in df_bio.columns if 'bio_age' in c]
            bio_grouped = df_bio.groupby('pincode')[bio_cols].sum().reset_index()
        else:
            bio_grouped = pd.DataFrame(columns=['pincode'])

        # Demographic Summary
        if not df_demo.empty:
            demo_cols = [c for c in df_demo.columns if 'demo_age' in c]
            demo_grouped = df_demo.groupby('pincode')[demo_cols].sum().reset_index()
        else:
            demo_grouped = pd.DataFrame(columns=['pincode'])

        # Master Merge (Outer join to keep all pincodes)
        master_df = pd.merge(enrol_grouped, bio_grouped, on='pincode', how='outer').fillna(0)
        master_df = pd.merge(master_df, demo_grouped, on='pincode', how='outer').fillna(0)

        # --- FEATURE ENGINEERING (The Secret Sauce) ---
        print("⚡ [DataEngine] Calculating Risk Features...")

        if 'age_0_5' in master_df.columns and 'age_18_greater' in master_df.columns:
            total_activity = master_df['age_0_5'] + master_df['age_18_greater'] + 1 
            master_df['adult_spike_ratio'] = master_df['age_18_greater'] / total_activity
        else:
            master_df['adult_spike_ratio'] = 0.0

        try:
            bio_adult_col = [c for c in master_df.columns if 'bio' in c and '17' in c][0]
            demo_adult_col = [c for c in master_df.columns if 'demo' in c and '17' in c][0]
            
            master_df['velocity_index'] = master_df[bio_adult_col] + master_df[demo_adult_col]
            master_df['ghost_ratio'] = master_df[demo_adult_col] / (master_df[bio_adult_col] + 1)
        except IndexError:
            print("⚠️ Warning: Could not find specific adult update columns. Setting default 0.")
            master_df['velocity_index'] = 0.0
            master_df['ghost_ratio'] = 0.0

        print(f"✅ [DataEngine] Pipeline Complete. Processed {len(master_df)} unique pincodes from {total_raw:,} records.")
        return master_df

if __name__ == "__main__":
    eng = DataEngine()
    df = eng.load_and_merge_data()
    df.to_csv("data/master_processed_data.csv", index=False)
    print("💾 Saved Master Dataset to 'data/master_processed_data.csv'")
