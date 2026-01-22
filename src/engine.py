import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
import os
import joblib

class AnomalyDetector:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
    def train_and_score(self, df, contamination=0.01):
        """
        Trains Isolation Forest and calculates SHAP values for explainability.
        """
        print("🧠 [AI Engine] Initializing Neural Pathways...")
        
        # 1. Select the Tri-Vector Features
        features = ['adult_spike_ratio', 'velocity_index', 'ghost_ratio']
        
        # Sanity Check: Ensure no NaNs
        X = df[features].fillna(0)
        
        # 2. Train Isolation Forest
        # contamination=0.01 means we expect roughly 1% of data to be anomalies
        print(f"🧠 [AI Engine] Training on {len(X)} records...")
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X)
        
        # 3. Predict Anomalies
        # -1 = Anomaly, 1 = Normal
        df['anomaly_label'] = model.predict(X)
        df['anomaly_score'] = model.decision_function(X) # Lower score = More anomalous
        
        # 4. Explainability (SHAP)
        # This tells us WHY it was flagged (e.g. "Velocity was too high")
        print("🧠 [AI Engine] Calculating Explanations (SHAP)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Store SHAP values in the dataframe (summed absolute impact for ranking)
        # Note: For strict SHAP plots in UI, we usually load the model + X, 
        # but here we save the 'dominant reason' for the CSV report.
        shap_df = pd.DataFrame(shap_values, columns=features)
        
        # Find the feature with the max impact for each row
        df['primary_risk_factor'] = shap_df.abs().idxmax(axis=1)
        
        # Convert -1/1 to meaningful text
        df['risk_status'] = df['anomaly_label'].apply(lambda x: 'CRITICAL' if x == -1 else 'Low')
        
        # 5. Save Model
        model_file = os.path.join(self.model_path, 'isolation_forest.pkl')
        joblib.dump(model, model_file)
        print(f"💾 Model saved to {model_file}")
        
        return df

if __name__ == "__main__":
    # Load the processed data
    data_path = "data/master_processed_data.csv"
    if not os.path.exists(data_path):
        print("❌ Run preprocessing.py first!")
    else:
        df = pd.read_csv(data_path)
        
        detector = AnomalyDetector()
        scored_df = detector.train_and_score(df)
        
        # Filter to show only anomalies
        anomalies = scored_df[scored_df['risk_status'] == 'CRITICAL']
        
        print(f"\n🚨 ANALYSIS COMPLETE")
        print(f"   Total Pincodes Scanned: {len(df)}")
        print(f"   Anomalies Detected: {len(anomalies)}")
        print("\n   Top 5 High-Risk Zones:")
        print(anomalies[['pincode', 'state', 'velocity_index', 'ghost_ratio', 'primary_risk_factor']].head())
        
        # Save Final Report
        scored_df.to_csv("data/final_scored_data.csv", index=False)
        print("\n✅ Final Intelligence Report saved to 'data/final_scored_data.csv'")