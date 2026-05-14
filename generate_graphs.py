import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, r2_score

# Set artifact dir
artifact_dir = r"C:\Users\tanuj\.gemini\antigravity\brain\792d506b-baed-4b62-9852-eafbc8249b30\artifacts"
os.makedirs(artifact_dir, exist_ok=True)

print("Loading models and data...")

# 1. Crop Recommendation Model
df_rec = pd.read_csv('Crop_recommendation.csv')
X_rec = df_rec.drop('label', axis=1)
y_rec = df_rec['label']

crop_clf = joblib.load('models/crop_recommendation_model.pkl')
crop_scaler = joblib.load('models/recommendation_scaler.pkl')
crop_le = joblib.load('models/recommendation_label_encoder.pkl')

y_rec_encoded = crop_le.transform(y_rec)
X_rec_scaled = crop_scaler.transform(X_rec)
X_train, X_test, y_train, y_test = train_test_split(X_rec_scaled, y_rec_encoded, test_size=0.2, random_state=42)

# Predictions
y_pred = crop_clf.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=crop_le.classes_, yticklabels=crop_le.classes_)
plt.title('Crop Recommendation: Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Crop', fontsize=12)
plt.ylabel('Actual Crop', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'confusion_matrix.png'), dpi=150)
plt.close()
print("Saved confusion matrix")

# Feature Importance (Crop Rec)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(crop_clf.feature_importances_, index=X_rec.columns)
feat_importances.nlargest(7).sort_values().plot(kind='barh', color='#2ecc71')
plt.title('Feature Importance: Crop Recommendation (LightGBM)', fontsize=16)
plt.xlabel('Importance (Split)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'feature_importance_crop.png'), dpi=150)
plt.close()
print("Saved crop feature importance")

# 2. Yield Prediction Model
df_prod = pd.read_csv('crop_production.csv').dropna()
df_prod['State_Name'] = df_prod['State_Name'].str.strip()
df_prod['Season'] = df_prod['Season'].str.strip()
df_prod['Crop'] = df_prod['Crop'].str.strip()
df_prod['Yield'] = df_prod['Production'] / (df_prod['Area'] + 1e-6)
q_high = df_prod['Yield'].quantile(0.99)
df_prod = df_prod[df_prod['Yield'] <= q_high]

yield_reg = joblib.load('models/yield_prediction_model.pkl')
yield_scaler = joblib.load('models/yield_scaler.pkl')
le_state = joblib.load('models/le_state.pkl')
le_season = joblib.load('models/le_season.pkl')
le_crop = joblib.load('models/le_crop.pkl')

X_prod = df_prod[['State_Name', 'Season', 'Crop', 'Area']].copy()
y_prod = df_prod['Production']

# Encode
X_prod['State_Name'] = le_state.transform(X_prod['State_Name'])
X_prod['Season'] = le_season.transform(X_prod['Season'])
X_prod['Crop'] = le_crop.transform(X_prod['Crop'])

X_prod_scaled = yield_scaler.transform(X_prod)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prod_scaled, y_prod, test_size=0.2, random_state=42)

y_pred_p = yield_reg.predict(X_test_p)

# Actual vs Predicted Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test_p, y_pred_p, alpha=0.3, color='#3498db')
plt.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'r--', lw=2)
plt.title('Yield Prediction: Actual vs Predicted Production', fontsize=16)
plt.xlabel('Actual Production', fontsize=12)
plt.ylabel('Predicted Production', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'actual_vs_predicted.png'), dpi=150)
plt.close()
print("Saved yield prediction plot")

# Feature Importance (Yield)
plt.figure(figsize=(10, 6))
feat_importances_y = pd.Series(yield_reg.feature_importances_, index=['State_Name', 'Season', 'Crop', 'Area'])
feat_importances_y.sort_values().plot(kind='barh', color='#9b59b6')
plt.title('Feature Importance: Yield Prediction (LightGBM)', fontsize=16)
plt.xlabel('Importance (Split)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(artifact_dir, 'feature_importance_yield.png'), dpi=150)
plt.close()
print("Saved yield feature importance")

print("All graphs successfully generated!")
