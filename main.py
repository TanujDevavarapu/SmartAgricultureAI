from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import requests
import os
import io
from gtts import gTTS

app = FastAPI(title="Smart Agriculture API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    crop_clf = joblib.load('models/crop_recommendation_model.pkl')
    crop_scaler = joblib.load('models/recommendation_scaler.pkl')
    crop_le = joblib.load('models/recommendation_label_encoder.pkl')
    X_train_rec = joblib.load('models/X_train_rec.pkl')
    
    yield_reg = joblib.load('models/yield_prediction_model.pkl')
    yield_scaler = joblib.load('models/yield_scaler.pkl')
    le_state = joblib.load('models/le_state.pkl')
    le_district = joblib.load('models/le_district.pkl')
    le_season = joblib.load('models/le_season.pkl')
    le_crop = joblib.load('models/le_crop.pkl')
    
    # Initialize SHAP explainer
    # We use TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(crop_clf)
    
except Exception as e:
    print(f"Error loading models: {e}")

class PredictRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    state: str
    district: str
    season: str
    area: float  # User provides acres; converted to hectares internally

ACRES_TO_HA = 0.404686  # 1 acre = 0.404686 hectares

# Simple state coordinates for Open-Meteo
STATE_COORDS = {
    "Andhra Pradesh": (15.91, 79.74),
    "Maharashtra": (19.75, 75.71),
    "Uttar Pradesh": (26.85, 80.95),
    "Punjab": (31.14, 75.34),
    "Karnataka": (15.31, 75.71),
    "Gujarat": (22.25, 71.19),
    "Madhya Pradesh": (22.97, 78.65),
    "Tamil Nadu": (11.12, 78.65),
    "Rajasthan": (27.02, 74.21),
    "Kerala": (10.85, 76.27),
}

@app.post("/api/predict")
async def predict_all(data: PredictRequest):
    try:
        # ── Area conversion: user enters ACRES, model trained on HECTARES ──
        area_acres = data.area
        area_ha = area_acres * ACRES_TO_HA

        # ── 1. Build & scale soil feature vector ──
        rec_features = pd.DataFrame(
            [[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        rec_scaled = crop_scaler.transform(rec_features)

        # ── 2. Rank ALL crops by soil suitability (predict_proba) ──
        #    Then walk down the ranked list and pick the first crop whose
        #    predicted production is > 0.  This ensures we NEVER recommend
        #    a crop with a negative or zero yield.
        proba = crop_clf.predict_proba(rec_scaled)[0]          # shape: (n_classes,)
        ranked_indices = np.argsort(proba)[::-1]               # best → worst

        state_enc  = le_state.transform([data.state])[0]  if data.state  in le_state.classes_  else 0
        district_enc = le_district.transform([data.district])[0] if data.district in le_district.classes_ else 0
        season_enc = le_season.transform([data.season])[0] if data.season in le_season.classes_ else 0

        chosen_crop       = None
        chosen_crop_idx   = None
        predicted_production = 0.0
        
        CROP_MAPPING = {
            'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Gram',
            'kidneybeans': 'Bean', 'pigeonpeas': 'Arhar/Tur',
            'mothbeans': 'Bean', 'mungbean': 'Moong(Green Gram)',
            'blackgram': 'Blackgram', 'lentil': 'Lentil',
            'pomegranate': 'Pomegranate', 'banana': 'Banana',
            'mango': 'Mango', 'grapes': 'Grapes', 'watermelon': 'Water Melon',
            'muskmelon': 'Melon', 'apple': 'Apple', 'orange': 'Orange',
            'papaya': 'Papaya', 'coconut': 'Coconut', 'cotton': 'Cotton(lint)',
            'jute': 'Jute', 'coffee': 'Coffee'
        }

        for idx in ranked_indices:
            candidate = crop_le.inverse_transform([idx])[0]
            mapped_candidate = CROP_MAPPING.get(candidate, candidate.title())
            crop_enc  = le_crop.transform([mapped_candidate])[0] if mapped_candidate in le_crop.classes_ else 0

            prod_features = pd.DataFrame(
                [[state_enc, district_enc, season_enc, crop_enc, area_ha]],
                columns=['State_Name', 'District_Name', 'Season', 'Crop', 'Area']
            )
            prod_scaled = yield_scaler.transform(prod_features)
            prod_pred   = yield_reg.predict(prod_scaled)[0]

            if prod_pred > 0:
                chosen_crop       = candidate
                chosen_crop_idx   = idx
                predicted_production = prod_pred
                break

        # Fallback: if every single crop somehow yields ≤ 0, use the top-ranked one
        # (extremely rare edge-case) and return 0 production
        if chosen_crop is None:
            chosen_crop_idx  = ranked_indices[0]
            chosen_crop      = crop_le.inverse_transform([chosen_crop_idx])[0]
            predicted_production = 0.0

        recommended_crop = chosen_crop
        crop_pred_idx    = chosen_crop_idx
        # Yield per acre (user-facing unit)
        predicted_yield  = predicted_production / (area_acres + 1e-6)

        # ── 3. SHAP Explainability for the chosen crop ──
        shap_values = explainer.shap_values(rec_scaled)

        if isinstance(shap_values, list):
            shap_values_for_class = shap_values[crop_pred_idx][0]
        elif len(shap_values.shape) == 3:
            shap_values_for_class = shap_values[0, :, crop_pred_idx]
        else:
            shap_values_for_class = shap_values[0]

        feature_names = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        shap_data = {
            "features": feature_names,
            "values": shap_values_for_class.tolist()
        }

        # ── 4. Fertilizer Recommendation (Heuristic) ──
        fertilizer_advice = "Your soil is balanced. Use standard compost."
        if data.N < 50:
            fertilizer_advice = "Nitrogen is low. Recommend Urea or Nitrogen-rich fertilizers."
        elif data.P < 40:
            fertilizer_advice = "Phosphorous is low. Recommend DAP (Diammonium Phosphate)."
        elif data.K < 40:
            fertilizer_advice = "Potassium is low. Recommend MOP (Muriate of Potash)."

        # ── 5. Soil Health Score ──
        soil_score = 100
        if data.ph < 5.5 or data.ph > 8.5: soil_score -= 20
        if data.N < 40: soil_score -= 15
        if data.P < 30: soil_score -= 15
        if data.K < 30: soil_score -= 15

        # ── 6. Live Weather from Open-Meteo ──
        weather_insights = "Normal seasonal conditions."
        coords = STATE_COORDS.get(data.state, (20.59, 78.96))
        try:
            w_res = requests.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={coords[0]}&longitude={coords[1]}&current_weather=true",
                timeout=4
            )
            if w_res.status_code == 200:
                current_w = w_res.json().get('current_weather', {})
                w_temp = current_w.get('temperature', data.temperature)
                weather_insights = f"Current temp in {data.state} is {w_temp}\u00b0C."
        except:
            pass

        return {
            "recommended_crop": recommended_crop,
            # Production in tonnes (model output)
            "predicted_production": round(float(predicted_production), 2),
            # Yield per acre (user-facing)
            "predicted_yield_per_area": round(float(predicted_yield), 4),
            # Area echo-back in both units
            "area_acres": round(area_acres, 2),
            "area_ha":    round(area_ha, 4),
            "shap_data": shap_data,
            "fertilizer_advice": fertilizer_advice,
            "soil_health_score": max(0, soil_score),
            "weather_insights": weather_insights
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class TTSRequest(BaseModel):
    text: str
    lang: str

@app.post("/api/tts")
async def text_to_speech(data: TTSRequest):
    try:
        # Map languages for gTTS
        gtts_lang = 'en'
        if data.lang == 'hi': gtts_lang = 'hi'
        elif data.lang == 'te': gtts_lang = 'te'
        
        tts = gTTS(text=data.text, lang=gtts_lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weather")
async def get_weather(state: str, district: str):
    try:
        # Search for district coords
        geo_res = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={district}&count=1&language=en")
        if geo_res.status_code == 200 and geo_res.json().get('results'):
            loc = geo_res.json()['results'][0]
            lat = loc['latitude']
            lon = loc['longitude']
            
            # Get weather
            w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain")
            if w_res.status_code == 200:
                cur = w_res.json().get('current', {})
                return {
                    "temperature": cur.get("temperature_2m"),
                    "humidity": cur.get("relative_humidity_2m"),
                    "rainfall": cur.get("rain")
                }
    except Exception as e:
        pass
    # Fallback to defaults
    return {}

@app.get("/")
async def serve_frontend():
    return FileResponse('index.html')

# Mount static files and crops images
if not os.path.exists('crops'):
    os.makedirs('crops')

app.mount("/crops", StaticFiles(directory="crops"), name="crops")
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
