import joblib
import pandas as pd
import numpy as np
import shap
import os

# Mock data
data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}

try:
    # Load Models
    crop_clf = joblib.load('models/crop_recommendation_model.pkl')
    crop_scaler = joblib.load('models/recommendation_scaler.pkl')
    crop_le = joblib.load('models/recommendation_label_encoder.pkl')
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(crop_clf)
    
    # Build & scale soil feature vector
    rec_features = pd.DataFrame(
        [[data['N'], data['P'], data['K'], data['temperature'], data['humidity'], data['ph'], data['rainfall']]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )
    rec_scaled = crop_scaler.transform(rec_features)
    
    # Predict
    proba = crop_clf.predict_proba(rec_scaled)[0]
    crop_pred_idx = np.argmax(proba)
    recommended_crop = crop_le.inverse_transform([crop_pred_idx])[0]
    
    print(f"Recommended Crop: {recommended_crop} (Index: {crop_pred_idx})")
    
    # SHAP
    shap_values = explainer.shap_values(rec_scaled)
    
    print(f"Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"List length: {len(shap_values)}")
        print(f"Shape of first element: {shap_values[0].shape}")
        shap_values_for_class = shap_values[crop_pred_idx][0]
    else:
        print(f"Shape of shap_values: {shap_values.shape}")
        if len(shap_values.shape) == 3:
            shap_values_for_class = shap_values[0, :, crop_pred_idx]
        else:
            shap_values_for_class = shap_values[0]
            
    print(f"SHAP values for class {crop_pred_idx}: {shap_values_for_class}")
    
    feature_names = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    shap_data = {
        "features": feature_names,
        "values": shap_values_for_class.tolist()
    }
    print(f"SHAP Data: {shap_data}")

except Exception as e:
    import traceback
    traceback.print_exc()
