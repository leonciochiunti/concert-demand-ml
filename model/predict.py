from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_demand(input_data: dict) -> str:
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction
