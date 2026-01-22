# 👁️ Project Drishti | AI-Powered Integrity Surveillance

**Submission for UIDAI Hackathon 2026**  
**Theme:** Data-Driven Innovation for Aadhaar

---

## 🚨 The Problem

UIDAI manages millions of daily transactions. Manual oversight is operationally impossible, leading to **silent fraud** such as:

- **Ghost Enrolments**  
  Fake identities created in bulk.

- **Impossible Velocities**  
  Operators processing updates faster than humanly possible.

- **Lifecycle Mismatches**  
  Address updates without biometric verification.

---

## 🛡️ The Solution: Project Drishti

**Drishti** is an **Unsupervised Anomaly Detection Engine** that acts as a digital sentry.

It does **not** rely on static rules. Instead, it uses **Isolation Forests** to learn the normal behavioral patterns of every **Pincode in India** and flag statistical outliers.

### 🔑 Key Features

- **Tri-Vector Analysis**  
  Correlates **Growth**, **Velocity**, and **Sustenance** vectors.

- **Explainable AI (XAI)**  
  Uses **SHAP values** to explain why a pincode was flagged  
  (example: *Velocity Index contributed 80% to Risk Score*).

- **Geospatial Dashboard**  
  Interactive dark-mode map for real-time threat monitoring.

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/project-drishti.git
cd project-drishti
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Setup

Place your UIDAI CSV files inside the `data/` folder.

> **Note:** Data is not included in this repository due to privacy and size constraints.

### 4. Ignite the Engine

```bash
# Step 1: Preprocess raw CSVs into Pincode Vectors
python src/preprocessing.py

# Step 2: Train AI Model & Score Data
python src/engine.py

# Step 3: Launch Mission Control
streamlit run src/app.py
```

---

## 🧠 Tech Stack

- **Preprocessing:** Pandas, NumPy (Vectorized Aggregation)
- **ML Core:** Scikit-Learn (Isolation Forest)
- **Explainability:** SHAP (TreeExplainer)
- **Visualization:** Streamlit, Folium, Plotly Express

---

### Built with ❤️ by **Builders&Breakers**
