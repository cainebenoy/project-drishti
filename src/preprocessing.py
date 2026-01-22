import pandas as pd
import numpy as np
import os

class DataEngine:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        
    def load_and_merge_data(self):
        """
        Reads all CSVs from the data folder and merges them into a single Pincode-level dataframe.
        """
        print("⚡ [DataEngine] Scanning data folder...")
        
        # 1. Initialize empty containers
        enrol_dfs = []
        bio_dfs = []
        demo_dfs = []

        # 2. Smart Loader: Finds files even if names vary slightly
        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"❌ Data folder '{self.data_path}' not found! Please create it.")

        files_found = 0
        for file in os.listdir(self.data_path):
            path = os.path.join(self.data_path, file)
            try:
                if 'enrolment' in file:
                    df = pd.read_csv(path)
                    # Normalize columns just in case
                    df.columns = df.columns.str.strip().str.lower()
                    enrol_dfs.append(df)
                    files_found += 1
                elif 'biometric' in file:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    bio_dfs.append(df)
                    files_found += 1
                elif 'demographic' in file:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip().str.lower()
                    demo_dfs.append(df)
                    files_found += 1
            except Exception as e:
                print(f"⚠️ Warning: Could not read {file}. Error: {e}")

        print(f"   Found: {len(enrol_dfs)} Enrolment, {len(bio_dfs)} Biometric, {len(demo_dfs)} Demographic files.")

        if files_found == 0:
            raise ValueError("❌ No valid CSV files found in 'data/' folder. Please check file placement.")

        # 3. Concatenate chunks
        print("⚡ [DataEngine] Combining chunks...")
        df_enrol = pd.concat(enrol_dfs, ignore_index=True) if enrol_dfs else pd.DataFrame(columns=['pincode', 'age_0_5', 'age_18_greater'])
        df_bio = pd.concat(bio_dfs, ignore_index=True) if bio_dfs else pd.DataFrame(columns=['pincode', 'bio_age_5_17', 'bio_age_17_greater'])
        df_demo = pd.concat(demo_dfs, ignore_index=True) if demo_dfs else pd.DataFrame(columns=['pincode', 'demo_age_5_17', 'demo_age_17_greater'])

        # 4. Aggregation Strategy (Group by Pincode)
        print("⚡ [DataEngine] Aggregating vectors...")
        
        # Handle potential missing columns if a file is empty/weird
        # We fill NaNs with 0 before summing
        
        # Vector A: Growth (Enrolment)
        enrol_grouped = df_enrol.groupby('pincode').agg({
            'age_0_5': 'sum',
            'age_18_greater': 'sum',
            'state': 'first',     # Just keep the name
            'district': 'first'
        }).reset_index()

        # Vector B: Sustenance (Biometric)
        # Note: CSV column names might be partial, so we rely on the clean names from step 2
        bio_cols = [c for c in df_bio.columns if 'bio_age' in c]
        bio_grouped = df_bio.groupby('pincode')[bio_cols].sum().reset_index()

        # Vector C: Mobility (Demographic)
        demo_cols = [c for c in df_demo.columns if 'demo_age' in c]
        demo_grouped = df_demo.groupby('pincode')[demo_cols].sum().reset_index()

        # 5. Merge Strategy (Outer Join to keep all pincodes)
        master_df = pd.merge(enrol_grouped, bio_grouped, on='pincode', how='outer').fillna(0)
        master_df = pd.merge(master_df, demo_grouped, on='pincode', how='outer').fillna(0)

        # 6. Feature Engineering (The Secret Sauce)
        print("⚡ [DataEngine] Calculating Risk Features...")
        
        # Feature 1: Adult Spike Ratio (New Adults vs Total Activity)
        # High value = Suspicious (Mature people usually have Aadhaar already)
        # We add 1 to denominator to avoid division by zero
        total_activity = master_df['age_0_5'] + master_df['age_18_greater'] + 1 
        master_df['adult_spike_ratio'] = master_df['age_18_greater'] / total_activity

        # Feature 2: Velocity (Total Updates)
        # We use a proxy for 'days' (assuming 1 day if not specified, or relative velocity)
        # Here we just sum the volume to find "Busy" centers
        # We look for cols that end with 'greater' (adults) or '17_' (adults)
        bio_adult_col = [c for c in master_df.columns if 'bio' in c and '17' in c][0]
        demo_adult_col = [c for c in master_df.columns if 'demo' in c and '17' in c][0]
        
        master_df['velocity_index'] = master_df[bio_adult_col] + master_df[demo_adult_col]

        # Feature 3: Ghost Ratio (Address Changes vs Biometric)
        # High Address changes without Biometric check = Suspicious
        master_df['ghost_ratio'] = master_df[demo_adult_col] / (master_df[bio_adult_col] + 1)

        print(f"✅ [DataEngine] Pipeline Complete. Processed {len(master_df)} unique pincodes.")
        return master_df

# Quick test
if __name__ == "__main__":
    eng = DataEngine()
    df = eng.load_and_merge_data()
    print(df.head())
    df.to_csv("data/master_processed_data.csv", index=False)
    print("💾 Saved processed data to data/master_processed_data.csv")