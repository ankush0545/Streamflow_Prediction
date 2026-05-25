# 🌊 Streamflow Prediction using Deep Learning

A machine learning/deep learning project for predicting streamflow and flood-related values using historical hydrological and rainfall data.

---

# 📌 Project Overview

This project predicts future streamflow values using environmental and weather-related features such as:

- Rainfall
- Basin characteristics
- Upstream area
- Historical streamflow
- Lag features
- Rolling statistics

The model is built using Deep Learning techniques and deployed with Streamlit for interactive predictions.

---

# 🚀 Features

- Data preprocessing pipeline
- Feature engineering
- Lag & rolling window features
- Deep learning prediction model
- Streamlit web application
- Scaler integration
- Real-time prediction interface

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Joblib

---

# 📂 Project Structure

```bash
project/
│
├── app/
│   └── app.py
│
├── preprocessing/
│   └── feature_engineering.py
│
├── models/
│   ├── model1.keras
│   ├── model2.keras
│   └── scaler.pkl
│
├── data/
│   └── sample_data.csv
│
├── requirements.txt
├── README.md
└── .gitignore