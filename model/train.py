"""
train.py

Script de entrenamiento final para ConcertDemandAI.

¿Por qué este archivo?
- El notebook 02_train_model.ipynb sirve para experimentar y explicar.
- Este script sirve para repetir el entrenamiento desde terminal:
  python model/train.py

Modelo final:
- Logistic Regression
- Seleccionado porque fue el mejor modelo en F1 macro y accuracy en el notebook.

Métricas finales esperadas del notebook:
- Accuracy: 0.8500
- F1 macro: 0.8512
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .model import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        prepare_training_dataframe,
        validate_training_columns,
    )
except ImportError:
    from model import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        prepare_training_dataframe,
        validate_training_columns,
    )


# ==============================
# 1. Rutas del proyecto
# ==============================
# Se calculan de forma relativa para que funcione desde Colab, VS Code o terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_TXT_PATH = MODEL_DIR / "metrics.txt"
METRICS_JSON_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42


# ==============================
# 2. Carga y preparación del dataset
# ==============================
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset y prepara las variables usadas por el modelo."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    df = pd.read_csv(path)

    # Preparamos las variables históricas y derivadas.
    # Esto evita diferencias entre notebook, script y app.
    df = prepare_training_dataframe(df)

    validate_training_columns(df)

    return df


# ==============================
# 3. Construcción del pipeline
# ==============================
def build_pipeline() -> Pipeline:
    """
    Construye el pipeline final.

    ¿Por qué Pipeline?
    Porque guarda juntos:
    - preprocesamiento de variables categóricas y numéricas
    - modelo final Logistic Regression

    Así model.pkl puede transformar datos nuevos y predecir sin pasos manuales.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ==============================
# 4. Evaluación del modelo
# ==============================
def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Calcula métricas del modelo final.

    ¿Por qué F1 macro?
    Porque el problema tiene tres clases: baja, media y alta.
    F1 macro evalúa el desempeño promedio entre clases sin favorecer una clase específica.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classes": list(pipeline.classes_),
        "features": FEATURES,
        "excluded_columns": [
            "occupancy_pct",
            "tickets_sold",
        ],
        "note": (
            "occupancy_pct y tickets_sold no se usan como variables de entrada "
            "porque son información posterior al evento y causarían fuga de información."
        ),
    }

    return metrics


# ==============================
# 5. Guardado de modelo y métricas
# ==============================
def save_artifacts(pipeline: Pipeline, metrics: Dict[str, Any]) -> None:
    """Guarda model.pkl, metrics.txt y metrics.json en la carpeta model/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    metrics_text = f"""Modelo final: {metrics['model']}

Accuracy: {metrics['accuracy']:.4f}
Precision macro: {metrics['precision_macro']:.4f}
Recall macro: {metrics['recall_macro']:.4f}
F1 macro: {metrics['f1_macro']:.4f}
F1 weighted: {metrics['f1_weighted']:.4f}

Reporte de clasificación:
{metrics['classification_report']}

Matriz de confusión:
{metrics['confusion_matrix']}

Nota:
{metrics['note']}
"""

    METRICS_TXT_PATH.write_text(metrics_text, encoding="utf-8")

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)


# ==============================
# 6. Ejecución principal
# ==============================
def main() -> None:
    """Entrena el modelo final y guarda los artefactos."""
    print("==============================")
    print("Entrenamiento del modelo final")
    print("==============================")

    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    # Stratify mantiene la proporción de baja/media/alta en train y test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    save_artifacts(pipeline, metrics)

    print("Modelo entrenado y guardado correctamente.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Métricas TXT: {METRICS_TXT_PATH}")
    print(f"Métricas JSON: {METRICS_JSON_PATH}")
    print()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
