# 👁️ Project Drishti | AI-Powered Integrity Surveillance

Built by Team Builders&Breakers for UIDAI Hackathon 2026 > Theme: Data-Driven Innovation for Aadhaar

## 🚨 The Challenge: Silent Fraud & Operational Blindspots

UIDAI manages one of the world's largest biometric databases, processing millions of daily transactions. However, the sheer scale creates operational blindspots where "Silent Fraud" can thrive:

Ghost Enrolments: Fake identities created in bulk by rogue operators.

Impossible Velocities: Centers processing updates faster than humanly possible (indicating batch processing scripts).

Lifecycle Mismatches: High volumes of demographic (address) updates without corresponding biometric verification, a key indicator of identity theft or "Ghost Beneficiaries."

Manual auditing of 1.5 million daily packets is impossible. UIDAI needs an automated, intelligent sentry.

## 🛡️ The Solution: Project Drishti

Project Drishti is an **Unsupervised Anomaly Detection Engine** that acts as a real-time digital surveillance layer. It shifts the audit paradigm from **Reactive Random Sampling** to **Proactive Targeted Investigation.**

Instead of using static, hard-coded rules (which are easily bypassed), Drishti uses **Isolation Forest** ML algorithms to learn the unique "heartbeat" of every Pincode in India. It automatically detects statistical outliers that deviate from normal patterns—without requiring labeled fraud examples (which UIDAI doesn't have).

### Why Unsupervised Learning?

- 🚫 No historical fraud labels available
- 🔍 Learns normal behavior patterns per pincode  
- ⚡ Scales to 2+ billion Aadhaar records
- 🎯 Detects both known and novel fraud patterns

## 🌟 Key Innovations

### Tri-Vector Analysis Engine

Drishti analyzes three independent risk vectors for each pincode:

- **Velocity Index**: Updates happening at machine-speed (>50/hour indicates bot/script automation)
- **Adult Spike Ratio**: Unnatural surge in new adult enrolments (saturation breach detection)
- **Ghost Ratio**: Address-only updates without corresponding biometric verification (identity theft indicator)

All three signals are combined by Isolation Forest to generate a unified anomaly score.

### Explainable AI (XAI)

Solves the "Black Box" problem. When Drishti flags a pincode, it uses **SHAP (SHapley Additive exPlanations)** to break down exactly which vector contributed how much:

```bash
Risk Score: 0.78 (CRITICAL)
├─ Velocity Index: +0.45 (60% contribution) ⚡
├─ Ghost Ratio: +0.22 (28% contribution) 👻  
└─ Adult Spike: +0.11 (12% contribution) 👶
```

Auditors get explainability—not just a red flag.

### Actionable Intelligence

- **Mission Orders (PDF)**: One-click generation of field inspection reports with GPS coordinates and anomaly details
- **SITAA Bridge**: Simulated integration with UIDAI's "System for Integrated Tools for Aadhaar Analysis"
- **Predictive Forecasting**: Uses Prophet time-series AI to predict update demand for better resource allocation (Mobile Kits)

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **Data Engineering** | Pandas, NumPy | Vectorized aggregation of raw CSV logs |
| **ML/AI Core** | Scikit-Learn (Isolation Forest) | Unsupervised anomaly detection |
| **Explainability** | SHAP (TreeExplainer) | Feature attribution for risk scores |
| **Time-Series Forecasting** | Facebook Prophet | Weekly/monthly update volume prediction |
| **Visualization** | Streamlit, Plotly, Folium | Interactive dashboard with geospatial maps |
| **Geospatial** | Geopy, ArcGIS | High-precision pincode geocoding |
| **Reporting** | ReportLab | Dynamic PDF mission order generation |

## 🚀 How to Run Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/project-drishti.git
cd project-drishti
```

### Step 2: Environment Setup

We recommend using a Python virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data

Place your UIDAI raw CSV files into the `data/` folder:

```bash
data/
├── api_data_aadhar_enrolment_*.csv           (Enrollment records)
├── api_data_aadhar_biometric_*.csv           (Biometric updates)
├── api_data_aadhar_demographic_*.csv         (Address/demographic updates)
└── daily_timeseries.csv                      (Optional: for forecasting)
```

> **Note**: Due to privacy and size constraints, raw data is not included in the repository.

### Step 5: Execute the Pipeline (In Order)

Run the following commands sequentially:

```bash
# 1. ETL: Transform & aggregate raw data
python src/preprocessing.py
# Output: data/master_processed_data.csv

# 2. ML: Train Isolation Forest & detect anomalies
python src/engine.py
# Output: data/final_scored_data.csv

# 3. Geolocation: Enrich critical cases with GPS coordinates
python src/geocoder.py
# Output: data/final_scored_data_geocoded.csv

# 4. Forecasting: Train Prophet for demand prediction
python src/forecaster.py
# Output: models/prophet_model.pkl, data/forecast_*.csv

# 5. Dashboard: Launch interactive Streamlit app
streamlit run src/app.py
# Access at: http://localhost:8501
```

### Important

Each step depends on the previous one. Don't skip steps or run them out of order!

---

## � Project Structure & File Guide

```bash
project-drishti/
├── src/
│   ├── preprocessing.py      # ETL: Load & transform UIDAI CSVs
│   ├── engine.py             # ML: Isolation Forest + SHAP explainability
│   ├── geocoder.py           # Geospatial: ArcGIS satellite geocoding
│   ├── forecaster.py         # Time-series: Prophet demand forecasting
│   ├── reporter.py           # PDF: Mission order generation
│   └── app.py                # Dashboard: Streamlit interactive UI
├── data/                      # Input/output data folder
│   ├── api_data_aadhar_*.csv  # Raw UIDAI data (not in repo)
│   ├── master_processed_data.csv      # After preprocessing
│   ├── final_scored_data.csv           # After engine.py
│   ├── final_scored_data_geocoded.csv  # After geocoder.py
│   └── daily_timeseries.csv            # For Prophet forecasting
├── models/                    # Trained ML models
│   ├── isolation_forest.pkl   # Trained IF model
│   └── prophet_model.pkl      # Trained Prophet model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### What Each File Does

| File | Input | Output | Purpose |
| --- | --- | --- | --- |
| `preprocessing.py` | Raw UIDAI CSVs | `master_processed_data.csv` | Aggregates data by pincode, calculates velocity/ghost ratio/spike |
| `engine.py` | `master_processed_data.csv` | `final_scored_data.csv` | Trains Isolation Forest, generates risk scores & SHAP explanations |
| `geocoder.py` | `final_scored_data.csv` | `final_scored_data_geocoded.csv` | Uses ArcGIS to get lat/lon for critical anomalies |
| `forecaster.py` | `daily_timeseries.csv` | Forecast charts | Prophet time-series forecasting for 30-day demand |
| `reporter.py` | Risk row data | PDF file | Generates "Mission Order" PDFs for field audits |
| `app.py` | All processed data | Web UI | Interactive Streamlit dashboard |

## 🔬 Technical Deep Dive

### Data Flow Architecture

```bash
RAW UIDAI DATA (1.5M+ daily records)
      ↓
[preprocessing.py]
  ├─ Load enrolment/biometric/demographic CSVs
  ├─ Aggregate by pincode
  ├─ Calculate risk vectors:
  │  ├─ velocity_index (updates/hour)
  │  ├─ adult_spike_ratio (new adults/time period)
  │  └─ ghost_ratio (demo_updates / bio_updates)
  ├─ Extract time-series for Prophet
  └─> Output: master_processed_data.csv
      ↓
[engine.py]
  ├─ Normalize features (velocity, spike, ghost)
  ├─ Train Isolation Forest (contamination=1%)
  ├─ Generate anomaly_score for each pincode
  ├─ Calculate SHAP values (explain why flagged)
  ├─ Identify primary_risk_factor per row
  ├─ Save trained model
  └─> Output: final_scored_data.csv (with risk_status: CRITICAL/Low)
      ↓
[geocoder.py]
  ├─ Filter for CRITICAL cases only
  ├─ Query ArcGIS for lat/lon
  ├─ Fallback: District/State level geocoding
  └─> Output: final_scored_data_geocoded.csv (with GPS coordinates)
      ├→
      │
[forecaster.py]
  ├─ Load daily_timeseries.csv
  ├─ Train Prophet model (weekly seasonality)
  ├─ Generate 30-day forecast with confidence intervals
  └─> Output: Forecast visualization in dashboard
      │
      └→ [app.py] Interactive Dashboard
         ├─ Tab 1: Geospatial map (Folium)
         ├─ Tab 2: Time-series forecast
         └─ Tab 3: Migration trends
```

### Isolation Forest Algorithm

Why Isolation Forest for anomaly detection?

- **Unsupervised**: No labeled fraud examples needed ✅
- **Efficient**: O(n log n) complexity handles millions of records ✅
- **Robust**: Works well with the 3-dimensional feature space ✅
- **Outlier-focused**: Specifically designed for anomaly detection (not classification) ✅
- **No feature scaling**: Robust to different magnitudes of velocity vs. spike ✅

**How it works**:

1. Randomly selects a feature and a split value
2. Recursively partitions the data
3. Anomalies get isolated faster (fewer recursive steps)
4. Lower decision_function score = more anomalous

**Contamination Rate**: Set to 1% (expects ~1% of pincodes to be anomalous)

### SHAP TreeExplainer

SHAP computes the contribution of each risk vector to the final anomaly score:

```bash
For a flagged pincode:
├─ Velocity Index (updates/hour): +0.45 contribution
├─ Ghost Ratio (address-only %): +0.22 contribution
└─ Adult Spike (new enrollments): +0.11 contribution
= Total Risk Score: 0.78 (CRITICAL)
```

This tells auditors which vector triggered the flag, enabling targeted investigations.

### Prophet Forecasting

Time-series forecasting for biometric update volume:

- **Seasonality**: Weekly patterns (post-weekend surges, monthly cycles)
- **Confidence Intervals**: 80% and 95% bounds for uncertainty
- **Changepoint Detection**: Identifies sudden shifts in demand
- **Use Case**: Resource planning (how many Mobile Kits needed per region?)

## 📊 Dashboard Features

The Streamlit app provides three main tabs:

### Tab 1: 🚨 Surveillance Map

- **Geospatial heatmap**: Red dots = CRITICAL, Cyan = Low risk
- **Popup details**: Pincode, risk factor, GPS accuracy
- **Risk breakdown pie chart**: Which vectors are dominant?
- **Priority audit list**: Sortable table of flagged pincodes
- **One-click PDF generation**: Download "Mission Orders" for field teams
- **SITAA alert button**: Simulate ticket creation in UIDAI's system

### Tab 2: 📈 Predictive Analytics

- **30-day demand forecast**: Prophet prediction with confidence bands
- **Seasonality detection**: Identify weekly/monthly patterns
- **Resource planning**: Estimate Mobile Kit deployment needs
- **Anomaly contextualization**: Is this surge normal for this pincode?

### Tab 3: 👥 Societal Trends

- **Migration seasonality chart**: Weekly update volume patterns
- **High-migration zones**: Districts with highest address-change ratios
- **Demographic risk split**: Breakdown of anomaly types by location
- **Insights**: Correlate with harvest seasons, festival periods, etc.

## 🎯 Use Cases

| Scenario | Detection Method | Action |
| --- | --- | --- |
| Ghost Enrolments | High velocity + Young age cluster | Field audit |
| Lifecycle Mismatch | Demographic ≫ Biometric updates | Identity theft investigation |
| Impossible Velocities | >50 updates/hour (single operator) | Operator credential audit |
| Saturation Breach | Growth curve inflection point | Resource reallocation |

## 🔐 Privacy & Security

Drishti is designed with privacy-first architecture:

- **No PII in risk scores**: Only aggregated metrics at pincode level
- **Pincode-level geocoding**: Not exact address coordinates
- **Anonymized vectors**: Velocity/ghost ratio are statistical, not tied to individuals
- **Audit trails**: All queries logged for compliance
- **GDPR-compatible**: Data can be deleted on request
- **Sandboxed**: No internet calls except for geocoding (ArcGIS only)

## 🌍 Deployment Scenarios

### Local Demo (Current)

- Runs on Windows/Mac/Linux with Python
- Uses sample data or your own UIDAI CSVs
- Ideal for POC and testing

### Production Deployment (Future)

- Containerize with Docker
- Deploy on cloud (AWS/Azure/GCP)
- Connect to UIDAI's data pipeline
- Horizontal scaling for 2+ billion records
- Real-time alert streaming to field teams

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

- **Built for**: UIDAI Hackathon 2026
- **Theme**: Data-Driven Innovation for Aadhaar
- **Team**: AI Engineers, Data Scientists, Geospatial Specialists

## 🤝 Support & Contributions

Found a bug? Have a feature request? Want to contribute?

1. Open an issue on GitHub
2. For contributions, create a pull request with:
   - Clear description of changes
   - Any new dependencies added to requirements.txt
   - Test on your local environment first

## ❓ FAQ

**Q: Do I need the full UIDAI dataset?**  
A: No! Start with sample data. The pipeline scales from thousands to millions of records.

**Q: How long does preprocessing take?**  
A: Depends on data size. ~1-5 minutes for typical datasets, hours for 1M+ records.

**Q: Can I use this with non-Aadhaar data?**  
A: Yes! Adapt preprocessing.py to load your data and calculate similar risk vectors.

**Q: Is the model production-ready?**  
A: This is a POC/hackathon project. For production, add:

- Model versioning
- Continuous retraining pipeline
- Monitoring & alerting
- Data validation checks
- Rate limiting on API calls

**Q: How often should I retrain the model?**  
A: Recommend weekly retraining to adapt to seasonal patterns and fraud evolution.

---

**Project Status**: 🚀 Proof of Concept for Field Testing  
**Last Updated**: January 2026
