# 👁️ Project Drishti | AI-Powered Integrity Surveillance

Built for UIDAI Hackathon 2026 > Theme: Data-Driven Innovation for Aadhaar

## 🚨 The Challenge: Silent Fraud & Operational Blindspots

UIDAI manages one of the world's largest biometric databases, processing millions of daily transactions. However, the sheer scale creates operational blindspots where "Silent Fraud" can thrive:

Ghost Enrolments: Fake identities created in bulk by rogue operators.

Impossible Velocities: Centers processing updates faster than humanly possible (indicating batch processing scripts).

Lifecycle Mismatches: High volumes of demographic (address) updates without corresponding biometric verification, a key indicator of identity theft or "Ghost Beneficiaries."

Manual auditing of 1.5 million daily packets is impossible. UIDAI needs an automated, intelligent sentry.

## 🛡️ The Solution: Project Drishti

Project Drishti is an Unsupervised Anomaly Detection Engine that acts as a real-time digital surveillance layer. It shifts the audit paradigm from "Reactive Random Sampling" to "Proactive Targeted Investigation."

Instead of using static, hard-coded rules (which are easily bypassed), Drishti uses Isolation Forests to learn the unique "heartbeat" of every Pincode in India. It detects statistical outliers that deviate from the norm.

## 🌟 Key Innovations

### Tri-Vector Analysis Engine

Velocity Vector: Detects machine-speed processing.

Growth Vector: Flags suspicious spikes in new adult enrolments (saturation breach).

Ghost Ratio: Identifies address-only updates that lack biometric proof.

Explainable AI (XAI): Solves the "Black Box" problem. When Drishti flags a pincode, it tells the auditor exactly why (e.g., "Velocity Index contributed 80% to Risk Score").

### Actionable Intelligence

Mission Orders (PDF): One-click generation of field inspection reports with GPS coordinates and checklists.

SITAA Bridge: Simulated integration with UIDAI's existing "System for Integrated Tools for Aadhaar Analysis."

Predictive Forecasting: Uses Prophet (Time-Series AI) to predict future update demand for better resource allocation (Mobile Kits).

## 🛠️ Technology Stack

Component

Tech Stack

Purpose

Data Engineering

Pandas, NumPy

Vectorized aggregation of raw CSV logs.

AI Core

Scikit-Learn (Isolation Forest)

Unsupervised anomaly detection.

Explainability

SHAP (TreeExplainer)

Feature attribution for risk scores.

Forecasting

Facebook Prophet

Time-series demand prediction.

Visualization

Streamlit, Plotly, Folium

Interactive Mission Control Dashboard.

Geospatial

ArcGIS (Geopy)

High-precision satellite geocoding.

Reporting

ReportLab

Dynamic PDF generation for field ops.

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/project-drishti.git
cd project-drishti
```

### 2. Environment Setup

We recommend using a virtual environment.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Data Setup

Place your UIDAI raw CSV files into the data/ folder.

Note: Due to privacy and size constraints, raw data is ignored by Git.

### 5. Ignite the Engine (Execution Order)

```bash
python src/preprocessing.py     # ETL pipeline
python src/engine.py             # Anomaly detection
python src/forecaster.py         # Time-series forecasting
python src/reporter.py           # PDF report generation
streamlit run src/app.py         # Interactive dashboard
```

---

## 📊 Output Artifacts

- **Risk Registry** (`final_scored_data.csv`): Anomaly scores per pincode
- **Mission Orders** (`reports/`): Geo-tagged PDF inspection reports
- **Forecast Models** (`models/`): Serialized Prophet/XGBoost models
- **Dashboard**: Real-time interactive visualization

## 🔬 Technical Deep Dive

### Isolation Forest Algorithm

Drishti uses **Isolation Forest** because:

- Handles high-dimensional data (velocity, growth, ghost ratio)
- Unsupervised (no labeled fraud cases needed)
- Fast inference (O(n log n))
- Robust to scaling differences

### SHAP TreeExplainer

When a pincode is flagged, SHAP decomposes the risk score:

```bash
Risk Score = 0.78
├─ Velocity Index: +0.45 (60%)
├─ Ghost Ratio: +0.22 (28%)
└─ Growth Vector: +0.11 (12%)
```

Auditors see exactly which signals triggered the alert.

### Prophet Forecasting

Predicts weekly update volume with confidence intervals. Useful for:

- Resource planning (Mobile Kit allocation)
- Anomaly contextualization (Is 500 new enrollments for a pincode normal? Prophet says yes for urban centers, no for villages)

## 🎯 Use Cases

| Scenario | Detection Method | Action |
| --- | --- | --- |
| Ghost Enrolments | High velocity + Young age cluster | Field audit |
| Lifecycle Mismatch | Demographic ≫ Biometric updates | Identity theft investigation |
| Impossible Velocities | >50 updates/hour (single operator) | Operator credential audit |
| Saturation Breach | Growth curve inflection point | Resource reallocation |

## 🔐 Privacy & Security

- No PII retained in risk scores (only aggregated metrics)
- All geolocation geocoded to pincode level (not exact addresses)
- Audit logs tracked for compliance
- GDPR-compatible (data deletion on request)

## 📝 License

MIT License - See LICENSE file

## 👥 Contributors

- Built for UIDAI Hackathon 2026
- Team: AI Engineers, Data Scientists, Geospatial Specialists

---

**Status**: Proof of Concept for field testing
