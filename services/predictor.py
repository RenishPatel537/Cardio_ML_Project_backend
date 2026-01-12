import joblib
import pandas as pd
from config import MODEL_PATHS

_models = {}

def load_model(model_name):
    if model_name not in _models:
        _models[model_name] = joblib.load(MODEL_PATHS[model_name])
    return _models[model_name]


def predict_cardio(data: dict, model_name: str):
    model = load_model(model_name)

    # ADD BMI HERE
    height_m = data["height"] / 100
    data["bmi"] = data["weight"] / (height_m ** 2)

    df = pd.DataFrame([data])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0].max())

    return {
        "model": model_name,
        "cardio": prediction,
        "confidence": round(probability, 3)
    }
