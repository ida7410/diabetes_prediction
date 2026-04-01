# Diabetes Prediction

Simple Python project for predicting diabetes in the Pima Indians dataset.

## 📦 Repository structure
- `data/diabetes.csv` (not tracked, download from Kaggle)
- `model/` (output from training, includes `diabetes_model.pkl`, `scaler.pkl`)
- `predictor.py` model inference helper
- `notebook/Model.ipynb` training + model saving
- `notebook/explore_data.ipynb` EDA + model comparison
- `requirements.txt` Python dependencies

## 🔗 Data download
The dataset is not included due license and is large. Download from Kaggle:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

After download, place `diabetes.csv` in the `data/` folder.

## ⚙️ Setup
1. Create a virtual environment (optional but recommended):
   - `python -m venv .venv`
2. Activate:
   - PowerShell: `.\.venv\Scripts\Activate`
3. Install requirements:
   - `pip install -r requirements.txt`

## ▶️ Train model
1. Open `notebook/Model.ipynb` in Jupyter.
2. Run all cells.
3. Confirm `model/diabetes_model.pkl` and `model/scaler.pkl` are created.

## 🧪 Quick inference
```
from predictor import Predictor
p = Predictor()
print(p.is_ready())  # True after model files exist
print(p.predict_w_params(1, 120, 70, 20, 79, 33.0, 0.5, 50))
```

## 📌 Notes
- `notebook/explore_data.ipynb` is for model comparison (`AUC`, `accuracy`, `confusion matrix`).
- `notebook/Model.ipynb` is the active training/notebook to create the final model.