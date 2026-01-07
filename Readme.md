# 🏠 Multimodal House Price Prediction using Satellite Imagery & Geospatial Features

This project presents a **full end-to-end multimodal machine learning pipeline** for house price prediction by combining:

- 📊 **Tabular housing attributes**
- 🛰️ **Satellite imagery embeddings (EfficientNet-B4)**
- 🌍 **Geospatial transport proximity features** (Metro, Railway, Airport)
- 🚀 **XGBoost regression with hyperparameter tuning**

The goal is to demonstrate how **visual context and spatial information** can improve traditional tabular regression models.

---

## 📌 Key Highlights

- 🔹 Satellite images fetched using **Mapbox Static API**
- 🔹 Transport proximity features extracted via **OSMnx**
- 🔹 Image embeddings generated using **pretrained EfficientNet-B4**
- 🔹 Dimensionality reduction with **PCA**
- 🔹 Leak-free **target encoding for zipcode**
- 🔹 Model explainability using **feature importance & Grad-CAM**
- 🔹 Clear separation of **data fetching, preprocessing, and modeling**

---

## 📂 Project Structure

```
CDC-House-Price-Prediction/
│
├── data/
│   ├── train.csv                    # Raw dataset
│   ├── train_with_transport.csv     # With geospatial features
│   ├── final_features.csv           # Fully processed features
│
├── data_fetcher.py                  # Satellite + OSM data fetching
├── preprocessing.ipynb              # Feature engineering, embeddings, EDA
├── model_training.ipynb             # XGBoost training & evaluation
├── README.md
```

---

## 🔧 Tech Stack

- **Python**
- **Pandas, NumPy**
- **OSMnx, Scikit-learn**
- **PyTorch, Torchvision**
- **XGBoost**
- **Matplotlib, Seaborn**

---

## 🛰️ Data Sources

### 1️⃣ Tabular Housing Data
Includes attributes such as:
- Bedrooms, bathrooms
- Living area, lot size
- Location (latitude, longitude)
- Construction & renovation year
- Zipcode, condition, grade

---

### 2️⃣ Satellite Imagery
- Source: **Mapbox Satellite Tiles**
- Zoom level: `18`
- Resolution: `512×512`
- One image per property based on latitude & longitude

---

### 3️⃣ Geospatial Transport Features
Computed distances (in meters) to nearest:
- 🚇 Metro station
- 🚆 Railway station
- ✈️ Airport

Features extracted using **OSMnx + BallTree (Haversine distance)**.

---

## ⚙️ Pipeline Overview

### 🔹 Step 1: Data Fetching (`data_fetcher.py`)
- Downloads satellite images
- Fetches OSM transport POIs
- Computes nearest distances
- Outputs cleaned CSV for modeling

---

### 🔹 Step 2: Preprocessing (`preprocessing.ipynb`)
- Data cleaning & deduplication
- Date feature extraction
- Housing feature engineering
- EfficientNet-B4 image embeddings (1792-D)
- Exploratory Data Analysis (EDA)
- Grad-CAM visualization for interpretability

---

### 🔹 Step 3: Model Training (`model_training.ipynb`)
- Train / validation split
- Leak-free target encoding for zipcode
- PCA on image embeddings
- XGBoost regression:
  - Tabular only
  - Tabular + image embeddings
- Hyperparameter tuning (RandomizedSearchCV)
- Performance comparison (RMSE, R²)

---

## 📈 Results

| Model | RMSE ↓ | R² ↑ |
|-------|--------|------|
| Tabular Only | Baseline | Baseline |
| Tabular + Image + Transport | **Improved** | **Improved** |

> ✅ Multimodal features consistently improve predictive performance.

---

## 🔍 Model Explainability

- **XGBoost Feature Importance** (Gain-based)
- **Grad-CAM visualizations** highlight spatial patterns such as:
  - Urban density
  - Proximity to water bodies
  - Infrastructure development

---

## 🧠 Key Learnings

- Satellite imagery adds **contextual signals** not present in tabular data
- Proper data leakage prevention is **critical** for fair evaluation
- PCA is essential when combining high-dimensional embeddings with tabular data
- Multimodal ML pipelines require **clear modularization**

---

## 🚀 How to Run

```bash
# Step 1: Fetch external data
python data_fetcher.py

# Step 2: Run preprocessing & EDA
jupyter notebook preprocessing.ipynb

# Step 3: Train models
jupyter notebook model_training.ipynb
```



