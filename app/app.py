"""
app.py

Interfaz Streamlit para ConcertDemandAI.

¿Por qué se actualizó?
- El modelo final ya no usa solamente variables básicas.
- Ahora necesita variables históricas y derivadas.
- La app debe capturar las variables principales y calcular el resto antes de predecir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ==============================
# 1. Configuración de rutas
# ==============================
# Streamlit ejecuta app/app.py desde la carpeta app.
# Agregamos la raíz del proyecto para poder importar model.predict.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model.predict import load_model, predict_event  # noqa: E402
from model.model import CITY_COUNTRY  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METRICS_JSON_PATH = PROJECT_ROOT / "model" / "metrics.json"


# ==============================
# 2. Configuración visual
# ==============================
st.set_page_config(
    page_title="ConcertDemandAI",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 ConcertDemandAI")
st.caption("Predicción de demanda de conciertos usando Machine Learning")


# ==============================
# 3. Carga del modelo
# ==============================
# Se usa cache para no cargar model.pkl en cada interacción.
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    model = get_model()
except Exception as error:
    st.error("No fue posible cargar el modelo entrenado.")
    st.info("Verifica que exista el archivo model/model.pkl.")
    st.exception(error)
    st.stop()


# ==============================
# 4. Carga de métricas
# ==============================
# Las métricas sirven para mostrar evidencia del desempeño del modelo final.
def load_metrics():
    if METRICS_JSON_PATH.exists():
        with METRICS_JSON_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return None


metrics = load_metrics()

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    col2.metric("F1 macro", f"{metrics.get('f1_macro', 0):.4f}")
    col3.metric("Modelo", metrics.get("model", "Logistic Regression"))


st.divider()


# ==============================
# 5. Formulario de entrada
# ==============================
# El usuario captura variables disponibles antes del evento.
# Las variables históricas y derivadas se calculan automáticamente en model.py.
st.subheader("Datos del concierto")

artist_options = [
    "BTS",
    "The Weeknd",
    "Ariana Grande",
    "Drake",
    "TWICE",
    "David Guetta",
    "Grupo Firme",
    "Los Tigres del Norte",
    "Peso Pluma",
]

genre_options = [
    "kpop",
    "pop",
    "rap",
    "edm",
    "regional_mexicano",
    "rock",
    "latin",
]

city_options = [
    "CDMX",
    "Guadalajara",
    "Monterrey",
    "Queretaro",
    "Leon",
    "Puebla",
    "Toluca",
    "Los Angeles",
    "Bogota",
    "Madrid",
]

venue_options = [
    "stadium",
    "arena",
    "theater",
    "club",
]

day_options = {
    "Lunes": "Monday",
    "Martes": "Tuesday",
    "Miércoles": "Wednesday",
    "Jueves": "Thursday",
    "Viernes": "Friday",
    "Sábado": "Saturday",
    "Domingo": "Sunday",
}


with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        artist = st.selectbox("Artista", artist_options)
        genre = st.selectbox("Género", genre_options)
        city = st.selectbox("Ciudad", city_options)
        country = CITY_COUNTRY.get(city, "Mexico")
        st.text_input("País", value=country, disabled=True)
        venue_type = st.selectbox("Tipo de recinto", venue_options)
        capacity = st.number_input("Capacidad del recinto", min_value=1000, max_value=100000, value=18000, step=1000)

    with right:
        month = st.slider("Mes del evento", min_value=1, max_value=12, value=8)
        event_day_label = st.selectbox("Día del evento", list(day_options.keys()), index=5)
        days_until_event = st.number_input("Días para el evento", min_value=1, max_value=365, value=90, step=1)
        marketing_budget = st.number_input("Presupuesto de marketing", min_value=0, max_value=2_000_000, value=180000, step=10000)
        ticket_price = st.number_input("Precio del boleto", min_value=100, max_value=10000, value=1800, step=100)
        artist_popularity = st.slider("Popularidad del artista", min_value=0, max_value=100, value=88)

    submitted = st.form_submit_button("Predecir demanda")


# ==============================
# 6. Predicción
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
        "event_day": day_options[event_day_label],
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
            "Recomendación: reforzar logística, inventario de boletos, campañas digitales "
            "y operación del recinto."
        )
    elif prediction == "media":
        st.warning("Demanda estimada: MEDIA")
        recommendation = (
            "Recomendación: monitorear preventa, ajustar marketing y revisar precios para "
            "evitar baja ocupación."
        )
    else:
        st.info("Demanda estimada: BAJA")
        recommendation = (
            "Recomendación: reducir riesgo operativo, optimizar presupuesto de marketing "
            "y considerar promociones."
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
# 7. Nota metodológica
# ==============================
st.divider()
st.caption(
    "Nota: Las variables históricas usadas por el modelo son simuladas para el MVP académico. "
    "El sistema no usa occupancy_pct ni tickets_sold como entrada para evitar fuga de información."
)
