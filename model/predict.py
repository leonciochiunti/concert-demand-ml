"""
predict.py

Funciones de predicción para ConcertDemandAI.

¿Por qué este archivo?
- Permite usar model.pkl sin volver a entrenar.
- Centraliza la forma correcta de construir las variables antes de predecir.
- Sirve como puente entre el modelo entrenado y app.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import pandas as pd

try:
    from .model import build_feature_dataframe
except ImportError:
    from model import build_feature_dataframe


# ==============================
# 1. Ruta del modelo entrenado
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"


# ==============================
# 2. Carga del modelo
# ==============================
def load_model(model_path: Path = MODEL_PATH):
    """Carga el modelo entrenado desde model.pkl."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Primero ejecuta python model/train.py o corre el notebook 02_train_model."
        )

    return joblib.load(model_path)


# ==============================
# 3. Predicción individual
# ==============================
def predict_event(event: Dict[str, Any], model=None) -> Dict[str, Any]:
    """
    Predice la demanda de un concierto.

    event puede traer solo las variables principales.
    Las variables derivadas se calculan en build_feature_dataframe.
    """
    if model is None:
        model = load_model()

    X = build_feature_dataframe(event)
    prediction = model.predict(X)[0]

    result: Dict[str, Any] = {
        "prediction": prediction,
        "input_features": X.iloc[0].to_dict(),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["probabilities"] = {
            class_name: float(probability)
            for class_name, probability in zip(model.classes_, probabilities)
        }

    return result


# ==============================
# 4. Predicción por lote
# ==============================
def predict_events(events: Iterable[Dict[str, Any]], model=None) -> pd.DataFrame:
    """Predice varios eventos y regresa un DataFrame con resultados."""
    if model is None:
        model = load_model()

    X = build_feature_dataframe(events)
    predictions = model.predict(X)

    results = X.copy()
    results["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for index, class_name in enumerate(model.classes_):
            results[f"prob_{class_name}"] = probabilities[:, index]

    return results


# ==============================
# 5. Ejemplo rápido desde terminal
# ==============================
if __name__ == "__main__":
    example_event = {
        "artist": "The Weeknd",
        "genre": "pop",
        "city": "Guadalajara",
        "venue_type": "arena",
        "capacity": 18000,
        "month": 8,
        "event_day": "Friday",
        "days_until_event": 90,
        "marketing_budget": 180000,
        "ticket_price": 1800,
        "artist_popularity": 88,
    }

    output = predict_event(example_event)

    print("Predicción:", output["prediction"])
    print("Probabilidades:")
    for class_name, probability in output.get("probabilities", {}).items():
        print(f"{class_name}: {probability:.4f}")
