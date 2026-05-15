"""
app.py

Streamlit app for ConcertDemandAI.

Updates included:
- The app uses a music icon again in the browser tab and title.
- Form options are loaded dynamically from data/dataset.csv.
- City selection is outside st.form so the detected country updates immediately.
- Country is detected automatically from the selected city.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ==============================
# 1. Project paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model.predict import load_model, predict_event  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METRICS_JSON_PATH = PROJECT_ROOT / "model" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
LOGO_PATH = PROJECT_ROOT / "assets" / "concertdemandai_logo.png"


# ==============================
# 2. Streamlit page configuration
# ==============================
# Music icon restored for the browser tab.
st.set_page_config(
    page_title="ConcertDemandAI",
    page_icon="🎵",
    layout="wide",
)


# ==============================
# 3. Data loading helpers
# ==============================
@st.cache_data
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load data/dataset.csv to build dynamic form options.

    This avoids hardcoded artists, genres, cities, countries,
    venue types and event days.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Make sure data/dataset.csv exists in the repository."
        )

    df = pd.read_csv(path)

    required_columns = [
        "artist",
        "genre",
        "city",
        "country",
        "venue_type",
        "capacity",
        "month",
        "event_day",
        "days_until_event",
        "marketing_budget",
        "ticket_price",
        "artist_popularity",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    return df


def sorted_unique(df: pd.DataFrame, column: str) -> list[Any]:
    """Return sorted unique values from a dataset column."""
    values = df[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value).lower())


def numeric_default(df: pd.DataFrame, column: str, fallback: float = 0) -> float:
    """Return a robust default value using the median of a numeric column."""
    if column not in df.columns:
        return fallback

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return fallback

    return float(values.median())


def numeric_min(df: pd.DataFrame, column: str, fallback: float = 0) -> float:
    """Return a numeric minimum for a column."""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else fallback


def numeric_max(df: pd.DataFrame, column: str, fallback: float = 100) -> float:
    """Return a numeric maximum for a column."""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else fallback


@st.cache_data
def get_city_country_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a dynamic city-to-country map from the dataset.

    If a city appears more than once, the most frequent country is used.
    Values such as empty, unknown or Desconocido are ignored when possible.
    """
    mapping: dict[str, str] = {}

    invalid_values = {"", "unknown", "desconocido", "nan", "none", "null"}

    for city, group in df.groupby("city"):
        countries = (
            group["country"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        countries = countries[~countries.str.lower().isin(invalid_values)]

        if countries.empty:
            continue

        mode = countries.mode()
        mapping[str(city)] = str(mode.iloc[0] if not mode.empty else countries.iloc[0])

    # Fallbacks for known cities in case the dataset has missing values.
    fallback_mapping = {
        "Seul": "South Korea",
        "Seoul": "South Korea",
        "Seúl": "South Korea",
        "Los Angeles": "Estados Unidos",
        "LosAngeles": "Estados Unidos",
        "Bogota": "Colombia",
        "Bogotá": "Colombia",
        "Madrid": "España",
        "CDMX": "Mexico",
        "Guadalajara": "Mexico",
        "Monterrey": "Mexico",
        "Queretaro": "Mexico",
        "Querétaro": "Mexico",
        "Leon": "Mexico",
        "León": "Mexico",
        "Puebla": "Mexico",
        "Toluca": "Mexico",
    }

    for city, country in fallback_mapping.items():
        mapping.setdefault(city, country)

    return mapping


@st.cache_resource
def get_model():
    """Load the trained model once and reuse it across app interactions."""
    return load_model(MODEL_PATH)


def load_metrics() -> dict[str, Any] | None:
    """Load model metrics if metrics.json exists."""
    if METRICS_JSON_PATH.exists():
        with METRICS_JSON_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    return None


# ==============================
# 4. Load model and dataset
# ==============================
try:
    model = get_model()
except Exception as error:
    st.error("No fue posible cargar el modelo entrenado.")
    st.info(
        "Verifica que exista model/model.pkl y que requirements.txt use "
        "versiones compatibles con el modelo entrenado."
    )
    st.exception(error)
    st.stop()

try:
    dataset = load_dataset()
except Exception as error:
    st.error("No fue posible cargar el dataset.")
    st.info("Verifica que exista data/dataset.csv y que contenga las columnas requeridas.")
    st.exception(error)
    st.stop()

metrics = load_metrics()
city_country_map = get_city_country_map(dataset)


# ==============================
# 5. Header
# ==============================
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=320)

# Music icon restored in the title.
st.title("🎵 ConcertDemandAI")
st.caption("Predicción de demanda de conciertos usando Machine Learning")

if metrics:
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    metric_col2.metric("F1 macro", f"{metrics.get('f1_macro', 0):.4f}")
    metric_col3.metric("Modelo", metrics.get("model", "Logistic Regression"))

st.divider()


# ==============================
# 6. Dynamic form options from dataset
# ==============================
artist_options = sorted_unique(dataset, "artist")
genre_options = sorted_unique(dataset, "genre")
city_options = sorted_unique(dataset, "city")
venue_options = sorted_unique(dataset, "venue_type")
event_day_options = sorted_unique(dataset, "event_day")

if not artist_options or not genre_options or not city_options or not venue_options or not event_day_options:
    st.error("El dataset no tiene suficientes valores para construir el formulario.")
    st.stop()


# ==============================
# 7. City selector outside the form
# ==============================
# Important:
# Widgets inside st.form only update when the submit button is pressed.
# City is outside the form so the detected country changes immediately.
st.subheader("Datos del concierto")

city_col, country_col = st.columns(2)

with city_col:
    city = st.selectbox("Ciudad", city_options)

with country_col:
    country = city_country_map.get(city, "Desconocido")
    st.info(f"País detectado: {country}")


# ==============================
# 8. Form
# ==============================
with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        artist = st.selectbox("Artista", artist_options)
        genre = st.selectbox("Género", genre_options)
        venue_type = st.selectbox("Tipo de recinto", venue_options)

        capacity_min = max(1, int(numeric_min(dataset, "capacity", 1000)))
        capacity_max = max(capacity_min, int(numeric_max(dataset, "capacity", 100000)))
        capacity_value = min(max(int(numeric_default(dataset, "capacity", 18000)), capacity_min), capacity_max)

        capacity = st.number_input(
            "Capacidad del recinto",
            min_value=capacity_min,
            max_value=capacity_max,
            value=capacity_value,
            step=1000,
        )

    with right:
        month_min = max(1, int(numeric_min(dataset, "month", 1)))
        month_max = min(12, int(numeric_max(dataset, "month", 12)))
        month_value = min(max(int(numeric_default(dataset, "month", 8)), month_min), month_max)

        month = st.slider(
            "Mes del evento",
            min_value=month_min,
            max_value=month_max,
            value=month_value,
        )

        event_day = st.selectbox("Día del evento", event_day_options)

        days_min = max(1, int(numeric_min(dataset, "days_until_event", 1)))
        days_max = max(days_min, int(numeric_max(dataset, "days_until_event", 365)))
        days_value = min(max(int(numeric_default(dataset, "days_until_event", 90)), days_min), days_max)

        days_until_event = st.number_input(
            "Días para el evento",
            min_value=days_min,
            max_value=days_max,
            value=days_value,
            step=1,
        )

        marketing_min = max(0, int(numeric_min(dataset, "marketing_budget", 0)))
        marketing_max = max(marketing_min, int(numeric_max(dataset, "marketing_budget", 2_000_000)))
        marketing_value = min(max(int(numeric_default(dataset, "marketing_budget", 180000)), marketing_min), marketing_max)

        marketing_budget = st.number_input(
            "Presupuesto de marketing",
            min_value=marketing_min,
            max_value=marketing_max,
            value=marketing_value,
            step=10000,
        )

        price_min = max(1, int(numeric_min(dataset, "ticket_price", 100)))
        price_max = max(price_min, int(numeric_max(dataset, "ticket_price", 10000)))
        price_value = min(max(int(numeric_default(dataset, "ticket_price", 1800)), price_min), price_max)

        ticket_price = st.number_input(
            "Precio del boleto",
            min_value=price_min,
            max_value=price_max,
            value=price_value,
            step=100,
        )

        popularity_min = max(0, int(numeric_min(dataset, "artist_popularity", 0)))
        popularity_max = min(100, int(numeric_max(dataset, "artist_popularity", 100)))
        popularity_value = min(max(int(numeric_default(dataset, "artist_popularity", 80)), popularity_min), popularity_max)

        artist_popularity = st.slider(
            "Popularidad del artista",
            min_value=popularity_min,
            max_value=popularity_max,
            value=popularity_value,
        )

    submitted = st.form_submit_button("Predecir demanda")


# ==============================
# 9. Prediction
# ==============================
if submitted:
    event_data = {
        "artist": artist,
        "genre": genre,
        "city": city,
        "country": country,
        "venue_type": venue_type,
        "capacity": capacity,
        "month": month,
        "event_day": event_day,
        "days_until_event": days_until_event,
        "marketing_budget": marketing_budget,
        "ticket_price": ticket_price,
        "artist_popularity": artist_popularity,
    }

    result = predict_event(event_data, model=model)

    prediction = result["prediction"]
    probabilities = result.get("probabilities", {})

    st.divider()
    st.subheader("Resultado de predicción")

    if prediction == "alta":
        st.success("Demanda estimada: ALTA")
        recommendation = (
            "Recomendación: reforzar logística, disponibilidad de boletos, "
            "campañas digitales y operación del recinto."
        )
    elif prediction == "media":
        st.warning("Demanda estimada: MEDIA")
        recommendation = (
            "Recomendación: monitorear preventa, ajustar marketing y revisar precios "
            "para mejorar la ocupación esperada."
        )
    else:
        st.info("Demanda estimada: BAJA")
        recommendation = (
            "Recomendación: reducir riesgo operativo, optimizar presupuesto de marketing "
            "y considerar promociones o ajustes de recinto."
        )

    st.write(recommendation)

    if probabilities:
        probability_df = pd.DataFrame(
            {
                "Clase": list(probabilities.keys()),
                "Probabilidad": list(probabilities.values()),
            }
        ).sort_values("Probabilidad", ascending=False)

        st.write("Probabilidades por clase")
        st.dataframe(probability_df, use_container_width=True)
        st.bar_chart(probability_df.set_index("Clase"))

    with st.expander("Ver variables enviadas al modelo"):
        st.dataframe(pd.DataFrame([result["input_features"]]), use_container_width=True)


# ==============================
# 10. Methodological note
# ==============================
st.divider()
st.caption(
    "Nota: Las opciones del formulario se cargan dinámicamente desde data/dataset.csv. "
    "El sistema no usa occupancy_pct ni tickets_sold como entrada para evitar fuga de información."
)
