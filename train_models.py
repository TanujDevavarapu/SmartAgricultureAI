import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score
import json

# Create models directory
os.makedirs('models', exist_ok=True)

print("Starting Phase 1: Model Training Pipeline...")

# ---------------------------------------------------------
# 1. Crop Recommendation Model
# ---------------------------------------------------------
print("Training Crop Recommendation Model...")
df_rec = pd.read_csv('Crop_recommendation.csv')

X_rec = df_rec.drop('label', axis=1)
y_rec = df_rec['label']

# Encode labels
le_rec = LabelEncoder()
y_rec_encoded = le_rec.fit_transform(y_rec)

# Scale features
scaler_rec = StandardScaler()
X_rec_scaled = scaler_rec.fit_transform(X_rec)

X_train, X_test, y_train, y_test = train_test_split(X_rec_scaled, y_rec_encoded, test_size=0.2, random_state=42)

# Evaluate Algorithms for Graph (LightGBM and CatBoost)
print("Evaluating algorithms for UI graph...")
metrics = {"Crop Recommendation": {}, "Yield Prediction R2": {}}

# 1. LightGBM
clf = LGBMClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_lgbm = accuracy_score(y_test, y_pred) * 100
metrics["Crop Recommendation"]["LightGBM"] = round(acc_lgbm, 2)

# 2. CatBoost
cb = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
cb.fit(X_train, y_train)
acc_cb = accuracy_score(y_test, cb.predict(X_test)) * 100
metrics["Crop Recommendation"]["CatBoost"] = round(acc_cb, 2)

print(f"Crop Recommendation Accuracies: {metrics['Crop Recommendation']}")

# Save best models & scalers (LightGBM is best)
joblib.dump(clf, 'models/crop_recommendation_model.pkl')
joblib.dump(scaler_rec, 'models/recommendation_scaler.pkl')
joblib.dump(le_rec, 'models/recommendation_label_encoder.pkl')
joblib.dump(X_train, 'models/X_train_rec.pkl') # For SHAP baseline

# ---------------------------------------------------------
# 2. Yield Prediction Model
# ---------------------------------------------------------
print("\nTraining Yield Prediction Model...")
df_prod = pd.read_csv('crop_production.csv')

# Handle missing values
df_prod = df_prod.dropna()

# Clean strings
df_prod['State_Name'] = df_prod['State_Name'].str.strip()
df_prod['Season'] = df_prod['Season'].str.strip()
df_prod['Crop'] = df_prod['Crop'].str.strip()

# Engineer Yield target (Production / Area)
# To avoid division by zero or extreme outliers, add a small epsilon to Area
df_prod['Yield'] = df_prod['Production'] / (df_prod['Area'] + 1e-6)

# Cap extreme outliers in yield (e.g., above 99th percentile) to make the model robust
q_high = df_prod['Yield'].quantile(0.99)
df_prod = df_prod[df_prod['Yield'] <= q_high]

# We will use State, District, Season, Crop, Area to predict Production.
features = ['State_Name', 'District_Name', 'Season', 'Crop', 'Area']
target = 'Yield'

X_prod = df_prod[features]
y_prod = df_prod[target]

# Encode categorical variables
le_state = LabelEncoder()
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

X_prod.loc[:, 'State_Name'] = le_state.fit_transform(X_prod['State_Name'])
X_prod.loc[:, 'District_Name'] = le_district.fit_transform(X_prod['District_Name'])
X_prod.loc[:, 'Season'] = le_season.fit_transform(X_prod['Season'])
X_prod.loc[:, 'Crop'] = le_crop.fit_transform(X_prod['Crop'])

# Scale features
scaler_prod = StandardScaler()
X_prod_scaled = scaler_prod.fit_transform(X_prod)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prod_scaled, y_prod, test_size=0.2, random_state=42)

# Train LightGBM Regressor
reg = LGBMRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_p, y_train_p)

y_pred_p = reg.predict(X_test_p)
r2 = r2_score(y_test_p, y_pred_p)
metrics["Yield Prediction R2"]["LightGBM"] = round(r2 * 100, 2)

# Train CatBoost Regressor
cb_reg = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
cb_reg.fit(X_train_p, y_train_p)
r2_cb = r2_score(y_test_p, cb_reg.predict(X_test_p))
metrics["Yield Prediction R2"]["CatBoost"] = round(r2_cb * 100, 2)

print(f"Yield Prediction R2 Scores: {metrics['Yield Prediction R2']}")

# Save models & scalers
joblib.dump(reg, 'models/yield_prediction_model.pkl')
joblib.dump(scaler_prod, 'models/yield_scaler.pkl')
joblib.dump(le_state, 'models/le_state.pkl')
joblib.dump(le_district, 'models/le_district.pkl')
joblib.dump(le_season, 'models/le_season.pkl')
joblib.dump(le_crop, 'models/le_crop.pkl')

# Save metrics for frontend
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\nModel Training Complete! All artifacts and metrics saved to 'models/' directory.")
