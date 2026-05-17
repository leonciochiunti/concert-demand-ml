"""
app.py

ConcertDemandAI - Streamlit application.

Complete version:
- Music app icon.
- Dynamic form options loaded from data/dataset.csv.
- Automatic country detection when the selected city changes.
- Automatic genre detection when the selected artist changes.
- Demand prediction using the trained model.
- Dynamic business evaluation after prediction:
    * estimated occupancy
    * estimated tickets sold
    * estimated revenue
    * commercial risk
    * revenue / marketing ratio
- Basic content-based concert recommendations using similar events from the dataset.

Methodological note:
occupancy_pct and tickets_sold are NOT used as model inputs. They are only used
after prediction as historical/simulated reference values to estimate business
metrics from similar events.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# 1. Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model.predict import load_model, predict_event  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METRICS_JSON_PATH = PROJECT_ROOT / "model" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
LOGO_PATH = PROJECT_ROOT / "assets" / "concertdemandai_logo.png"


# ============================================================
# 2. Streamlit configuration
# ============================================================
st.set_page_config(
    page_title="ConcertDemandAI",
    page_icon="🎵",
    layout="wide",
)


# ============================================================
# 3. Dataset and model helpers
# ============================================================
@st.cache_data
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset used by the application.

    The dataset is used for:
    - Dynamic form options.
    - City-to-country detection.
    - Artist-to-genre detection.
    - Similar event recommendations.
    - Business metric estimation after prediction.
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


@st.cache_resource
def get_model():
    """Load the trained model once and reuse it during the app session."""
    return load_model(MODEL_PATH)


def load_metrics() -> dict[str, Any] | None:
    """Load model metrics from model/metrics.json if available."""
    if METRICS_JSON_PATH.exists():
        with METRICS_JSON_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    return None


def sorted_unique(df: pd.DataFrame, column: str) -> list[Any]:
    """Return sorted unique non-empty values from a dataset column."""
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]
    return sorted(values.unique().tolist(), key=lambda value: value.lower())


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric version of a column."""
    if column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[column], errors="coerce").dropna()


def numeric_default(df: pd.DataFrame, column: str, fallback: float = 0) -> float:
    """Return a robust default value using the median of a numeric column."""
    values = numeric_series(df, column)
    if values.empty:
        return fallback

    return float(values.median())


def numeric_min(df: pd.DataFrame, column: str, fallback: float = 0) -> float:
    """Return the minimum value of a numeric column."""
    values = numeric_series(df, column)
    if values.empty:
        return fallback

    return float(values.min())


def numeric_max(df: pd.DataFrame, column: str, fallback: float = 100) -> float:
    """Return the maximum value of a numeric column."""
    values = numeric_series(df, column)
    if values.empty:
        return fallback

    return float(values.max())


@st.cache_data
def get_city_country_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a dynamic city-to-country map from the dataset.

    If a city has multiple countries, the most frequent valid country is used.
    Unknown values are ignored when possible.
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

    fallback_mapping = {
        "Seul": "Corea del Sur",
        "Seoul": "Corea del Sur",
        "Seúl": "Corea del Sur",
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


@st.cache_data
def get_artist_genre_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a dynamic artist-to-genre map from the dataset.

    If an artist appears with multiple genres, the most frequent genre is used.
    This prevents incoherent combinations such as aespa + banda when
    the dataset associates aespa mainly with kpop.
    """
    mapping: dict[str, str] = {}
    invalid_values = {"", "unknown", "desconocido", "nan", "none", "null"}

    for artist, group in df.groupby("artist"):
        genres = (
            group["genre"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        genres = genres[~genres.str.lower().isin(invalid_values)]

        if genres.empty:
            continue

        mode = genres.mode()
        mapping[str(artist)] = str(mode.iloc[0] if not mode.empty else genres.iloc[0])

    return mapping


# ============================================================
# 4. Recommendation and business evaluation helpers
# ============================================================
def add_similarity_score(
    df: pd.DataFrame,
    event_data: dict[str, Any],
) -> pd.DataFrame:
    """
    Add a simple content-based similarity score to dataset events.

    Similarity is based on:
    - genre
    - city
    - country
    - venue type
    - artist popularity
    - ticket price
    - capacity
    """
    candidates = df.copy()
    candidates["similarity_score"] = 0

    if "genre" in candidates.columns:
        candidates.loc[candidates["genre"] == event_data["genre"], "similarity_score"] += 4

    if "city" in candidates.columns:
        candidates.loc[candidates["city"] == event_data["city"], "similarity_score"] += 3

    if "country" in candidates.columns:
        candidates.loc[candidates["country"] == event_data["country"], "similarity_score"] += 2

    if "venue_type" in candidates.columns:
        candidates.loc[candidates["venue_type"] == event_data["venue_type"], "similarity_score"] += 2

    if "artist_popularity" in candidates.columns:
        candidates["artist_popularity_numeric"] = pd.to_numeric(
            candidates["artist_popularity"],
            errors="coerce",
        )

        candidates["popularity_difference"] = (
            candidates["artist_popularity_numeric"] - float(event_data["artist_popularity"])
        ).abs()

        candidates.loc[candidates["popularity_difference"] <= 10, "similarity_score"] += 2
        candidates.loc[candidates["popularity_difference"] <= 5, "similarity_score"] += 1

    if "ticket_price" in candidates.columns:
        candidates["ticket_price_numeric"] = pd.to_numeric(
            candidates["ticket_price"],
            errors="coerce",
        )

        candidates["price_difference"] = (
            candidates["ticket_price_numeric"] - float(event_data["ticket_price"])
        ).abs()

        candidates.loc[candidates["price_difference"] <= 300, "similarity_score"] += 1
        candidates.loc[candidates["price_difference"] <= 150, "similarity_score"] += 1

    if "capacity" in candidates.columns:
        candidates["capacity_numeric"] = pd.to_numeric(
            candidates["capacity"],
            errors="coerce",
        )

        selected_capacity = max(float(event_data["capacity"]), 1)
        candidates["capacity_difference_pct"] = (
            (candidates["capacity_numeric"] - selected_capacity).abs() / selected_capacity
        )

        candidates.loc[candidates["capacity_difference_pct"] <= 0.25, "similarity_score"] += 1

    return candidates.sort_values("similarity_score", ascending=False)


def recommend_similar_events(
    df: pd.DataFrame,
    event_data: dict[str, Any],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend similar concerts using a basic content-based approach.

    This module supports the recommendation scope of the MVP:
    it suggests events similar to the one evaluated by the user.
    """
    recommendations = add_similarity_score(df, event_data)

    if "artist" in recommendations.columns:
        recommendations = recommendations[recommendations["artist"] != event_data["artist"]]

    columns_to_show = [
        "artist",
        "genre",
        "city",
        "country",
        "venue_type",
        "ticket_price",
        "artist_popularity",
        "demand_class",
        "similarity_score",
    ]

    available_columns = [
        column for column in columns_to_show
        if column in recommendations.columns
    ]

    subset_columns = [
        column for column in ["artist", "genre", "city", "venue_type"]
        if column in recommendations.columns
    ]

    if subset_columns:
        recommendations = recommendations.drop_duplicates(subset=subset_columns)

    return recommendations[available_columns].head(top_n)


def estimate_event_business_metrics(
    df: pd.DataFrame,
    event_data: dict[str, Any],
    predicted_demand: str,
) -> dict[str, Any]:
    """
    Estimate business metrics dynamically using similar events from the dataset.

    This does not use occupancy_pct or tickets_sold as model inputs.
    They are used only after the prediction as historical/simulated references.
    """
    candidates = df.copy()

    if "demand_class" in candidates.columns:
        same_class = candidates[candidates["demand_class"] == predicted_demand]
        if not same_class.empty:
            candidates = same_class.copy()

    candidates = add_similarity_score(candidates, event_data)
    reference_events = candidates.head(30)

    fallback_occupancy = {
        "baja": 0.45,
        "media": 0.68,
        "alta": 0.88,
    }

    estimated_occupancy = fallback_occupancy.get(predicted_demand, 0.65)

    if "occupancy_pct" in reference_events.columns:
        occupancy_values = pd.to_numeric(
            reference_events["occupancy_pct"],
            errors="coerce",
        ).dropna()

        if not occupancy_values.empty:
            estimated_occupancy = float(occupancy_values.median())

    if estimated_occupancy > 1:
        estimated_occupancy = estimated_occupancy / 100

    estimated_occupancy = max(0, min(estimated_occupancy, 1))

    capacity = int(event_data["capacity"])
    ticket_price = float(event_data["ticket_price"])
    marketing_budget = max(float(event_data["marketing_budget"]), 1)

    estimated_tickets = int(round(capacity * estimated_occupancy))
    estimated_revenue = float(estimated_tickets * ticket_price)
    revenue_per_marketing = estimated_revenue / marketing_budget

    if predicted_demand == "alta":
        risk_level = "Bajo"
        strategic_advice = (
            "La demanda estimada es alta. Conviene reforzar logística, disponibilidad "
            "de boletos, operación del recinto y campañas de conversión."
        )
    elif predicted_demand == "media":
        risk_level = "Moderado"
        strategic_advice = (
            "La demanda estimada es media. Conviene monitorear la preventa, ajustar "
            "el presupuesto de marketing y revisar si el precio o el recinto son adecuados."
        )
    else:
        risk_level = "Alto"
        strategic_advice = (
            "La demanda estimada es baja. Conviene reducir riesgo operativo, considerar "
            "promociones, revisar el precio o evaluar un recinto de menor capacidad."
        )

    return {
        "estimated_occupancy": estimated_occupancy,
        "estimated_tickets": estimated_tickets,
        "estimated_revenue": estimated_revenue,
        "revenue_per_marketing": revenue_per_marketing,
        "risk_level": risk_level,
        "strategic_advice": strategic_advice,
        "reference_events_used": len(reference_events),
    }


def format_currency(value: float) -> str:
    """Format a numeric value as currency."""
    return f"${value:,.0f}"


# ============================================================
# 5. Load resources
# ============================================================
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
artist_genre_map = get_artist_genre_map(dataset)


# ============================================================
# 6. Header
# ============================================================
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=320)

st.title("🎵 ConcertDemandAI")
st.caption(
    "Predicción y evaluación inteligente de demanda para conciertos usando Machine Learning."
)

if metrics:
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    metric_col2.metric("F1 macro", f"{metrics.get('f1_macro', 0):.4f}")
    metric_col3.metric("Modelo", metrics.get("model", "Logistic Regression"))

st.divider()


# ============================================================
# 7. Dynamic form options
# ============================================================
artist_options = sorted_unique(dataset, "artist")
city_options = sorted_unique(dataset, "city")
venue_options = sorted_unique(dataset, "venue_type")
event_day_options = sorted_unique(dataset, "event_day")

if not artist_options or not city_options or not venue_options or not event_day_options:
    st.error("El dataset no tiene suficientes valores para construir el formulario.")
    st.stop()


# ============================================================
# 8. Selectors outside the form
# ============================================================
# Widgets inside st.form only update after pressing the submit button.
# City and artist are outside the form so country and genre update immediately.
st.subheader("Datos del concierto")

selector_col1, selector_col2 = st.columns(2)

with selector_col1:
    city = st.selectbox("Ciudad", city_options)

with selector_col2:
    country = city_country_map.get(city, "Desconocido")
    st.info(f"País detectado: {country}")

artist_col, genre_col = st.columns(2)

with artist_col:
    artist = st.selectbox("Artista", artist_options)

with genre_col:
    genre = artist_genre_map.get(artist, "Desconocido")
    st.info(f"Género detectado: {genre}")


# ============================================================
# 9. Prediction form
# ============================================================
with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        venue_type = st.selectbox("Tipo de recinto", venue_options)

        capacity_min = max(1, int(numeric_min(dataset, "capacity", 1000)))
        capacity_max = max(capacity_min, int(numeric_max(dataset, "capacity", 100000)))
        capacity_value = min(
            max(int(numeric_default(dataset, "capacity", 18000)), capacity_min),
            capacity_max,
        )

        capacity = st.number_input(
            "Capacidad del recinto",
            min_value=capacity_min,
            max_value=capacity_max,
            value=capacity_value,
            step=1000,
        )

        price_min = max(1, int(numeric_min(dataset, "ticket_price", 100)))
        price_max = max(price_min, int(numeric_max(dataset, "ticket_price", 10000)))
        price_value = min(
            max(int(numeric_default(dataset, "ticket_price", 1800)), price_min),
            price_max,
        )

        ticket_price = st.number_input(
            "Precio del boleto",
            min_value=price_min,
            max_value=price_max,
            value=price_value,
            step=100,
        )

    with right:
        month_min = max(1, int(numeric_min(dataset, "month", 1)))
        month_max = min(12, int(numeric_max(dataset, "month", 12)))
        month_value = min(
            max(int(numeric_default(dataset, "month", 8)), month_min),
            month_max,
        )

        month = st.slider(
            "Mes del evento",
            min_value=month_min,
            max_value=month_max,
            value=month_value,
        )

        event_day = st.selectbox("Día del evento", event_day_options)

        days_min = max(1, int(numeric_min(dataset, "days_until_event", 1)))
        days_max = max(days_min, int(numeric_max(dataset, "days_until_event", 365)))
        days_value = min(
            max(int(numeric_default(dataset, "days_until_event", 90)), days_min),
            days_max,
        )

        days_until_event = st.number_input(
            "Días para el evento",
            min_value=days_min,
            max_value=days_max,
            value=days_value,
            step=1,
        )

        marketing_min = max(0, int(numeric_min(dataset, "marketing_budget", 0)))
        marketing_max = max(
            marketing_min,
            int(numeric_max(dataset, "marketing_budget", 2_000_000)),
        )
        marketing_value = min(
            max(int(numeric_default(dataset, "marketing_budget", 180000)), marketing_min),
            marketing_max,
        )

        marketing_budget = st.number_input(
            "Presupuesto de marketing",
            min_value=marketing_min,
            max_value=marketing_max,
            value=marketing_value,
            step=10000,
        )

        popularity_min = max(0, int(numeric_min(dataset, "artist_popularity", 0)))
        popularity_max = min(100, int(numeric_max(dataset, "artist_popularity", 100)))
        popularity_value = min(
            max(int(numeric_default(dataset, "artist_popularity", 80)), popularity_min),
            popularity_max,
        )

        artist_popularity = st.slider(
            "Popularidad del artista",
            min_value=popularity_min,
            max_value=popularity_max,
            value=popularity_value,
        )

    submitted = st.form_submit_button("Predecir demanda y evaluar evento")


# ============================================================
# 10. Prediction, business evaluation and recommendations
# ============================================================
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
    elif prediction == "media":
        st.warning("Demanda estimada: MEDIA")
    else:
        st.info("Demanda estimada: BAJA")

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

    business_metrics = estimate_event_business_metrics(
        dataset,
        event_data,
        prediction,
    )

    st.subheader("Evaluación estimada del evento")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)

    metric_a.metric(
        "Ocupación estimada",
        f"{business_metrics['estimated_occupancy']:.1%}",
    )

    metric_b.metric(
        "Boletos estimados",
        f"{business_metrics['estimated_tickets']:,}",
    )

    metric_c.metric(
        "Ingreso estimado",
        format_currency(business_metrics["estimated_revenue"]),
    )

    metric_d.metric(
        "Riesgo comercial",
        business_metrics["risk_level"],
    )

    st.write(business_metrics["strategic_advice"])

    with st.expander("Detalles de la evaluación estimada"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.metric(
                "Ingreso por cada peso invertido en marketing",
                f"{business_metrics['revenue_per_marketing']:.2f}x",
            )

        with detail_col2:
            st.metric(
                "Eventos usados como referencia",
                business_metrics["reference_events_used"],
            )

        st.caption(
            "La evaluación se calcula de forma dinámica usando eventos similares del dataset. "
            "Las variables occupancy_pct y tickets_sold no se usan como entradas del modelo; "
            "solo se utilizan después de la predicción como referencia histórica/simulada."
        )

    st.subheader("Conciertos similares recomendados")

    similar_events = recommend_similar_events(
        dataset,
        event_data,
        top_n=5,
    )

    if similar_events.empty:
        st.info("No se encontraron conciertos similares en el dataset.")
    else:
        st.write(
            "Estas recomendaciones se generan comparando género, ciudad, país, "
            "tipo de recinto, precio, capacidad y popularidad del artista."
        )
        st.dataframe(similar_events, use_container_width=True)

    with st.expander("Ver variables enviadas al modelo"):
        input_features = result.get("input_features", event_data)
        st.dataframe(pd.DataFrame([input_features]), use_container_width=True)


# ============================================================
# 11. Methodological note
# ============================================================
st.divider()
st.caption(
    "Nota metodológica: las opciones del formulario se cargan dinámicamente desde "
    "data/dataset.csv. El modelo predice demanda baja, media o alta. La evaluación "
    "de ocupación, boletos e ingresos se calcula después de la predicción usando "
    "eventos similares como referencia. Esto evita fuga de información porque "
    "occupancy_pct y tickets_sold no son entradas del modelo."
)
