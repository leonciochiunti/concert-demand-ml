"""
model.py

Funciones y constantes compartidas para ConcertDemandAI.

¿Por qué existe este archivo?
- Para que train.py, predict.py y app.py usen las mismas columnas.
- Para evitar que cada archivo calcule las variables de forma diferente.
- Para mantener en un solo lugar los mapas de artista, ciudad, país y variables derivadas.

Nota importante:
Las variables históricas usadas aquí son simuladas para el MVP académico.
No representan afirmaciones reales sobre artistas o mercados.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


# ==============================
# 1. Columnas del modelo
# ==============================
# Estas columnas son las mismas que se usaron en el notebook 02_train_model.
# Se excluyen occupancy_pct y tickets_sold porque son variables posteriores al evento.
# Usarlas como entrada sería fuga de información.
FEATURES: List[str] = [
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
    "previous_attendance_rate",
    "pre_sale_interest",
    "artist_market_score",
    "is_weekend",
    "marketing_per_capacity",
    "price_per_capacity",
    "popularity_marketing",
]

TARGET = "demand_class"

CATEGORICAL_FEATURES: List[str] = [
    "artist",
    "genre",
    "city",
    "country",
    "venue_type",
    "event_day",
]

NUMERIC_FEATURES: List[str] = [
    "capacity",
    "month",
    "days_until_event",
    "marketing_budget",
    "ticket_price",
    "artist_popularity",
    "previous_attendance_rate",
    "pre_sale_interest",
    "artist_market_score",
    "is_weekend",
    "marketing_per_capacity",
    "price_per_capacity",
    "popularity_marketing",
]


# ==============================
# 2. Mapas usados para enriquecer datos
# ==============================
# Estos valores se basan en el dataset simulado del proyecto.
# Sirven para estimar señales previas al evento cuando el usuario no las captura manualmente.
ARTIST_ATTENDANCE_RATE: Dict[str, float] = {
    "BTS": 0.95,
    "The Weeknd": 0.90,
    "Ariana Grande": 0.88,
    "Drake": 0.86,
    "TWICE": 0.84,
    "David Guetta": 0.82,
    "Grupo Firme": 0.78,
    "Los Tigres del Norte": 0.74,
    "Peso Pluma": 0.58,
}

CITY_MARKET_SCORE: Dict[str, float] = {
    "CDMX": 1.00,
    "Guadalajara": 0.92,
    "Monterrey": 0.92,
    "Queretaro": 0.82,
    "Querétaro": 0.82,
    "Leon": 0.78,
    "León": 0.78,
    "Puebla": 0.76,
    "Toluca": 0.72,
    "Los Angeles": 0.95,
    "Bogota": 0.88,
    "Bogotá": 0.88,
    "Madrid": 0.90,
}

CITY_COUNTRY: Dict[str, str] = {
    "CDMX": "Mexico",
    "Guadalajara": "Mexico",
    "Monterrey": "Mexico",
    "Queretaro": "Mexico",
    "Querétaro": "Mexico",
    "Leon": "Mexico",
    "León": "Mexico",
    "Puebla": "Mexico",
    "Toluca": "Mexico",
    "Los Angeles": "Estados Unidos",
    "Bogota": "Colombia",
    "Bogotá": "Colombia",
    "Madrid": "España",
}

DAY_NORMALIZATION: Dict[str, str] = {
    "monday": "Monday",
    "lunes": "Monday",
    "tuesday": "Tuesday",
    "martes": "Tuesday",
    "wednesday": "Wednesday",
    "miercoles": "Wednesday",
    "miércoles": "Wednesday",
    "thursday": "Thursday",
    "jueves": "Thursday",
    "friday": "Friday",
    "viernes": "Friday",
    "saturday": "Saturday",
    "sabado": "Saturday",
    "sábado": "Saturday",
    "sunday": "Sunday",
    "domingo": "Sunday",
}

WEEKEND_DAYS = {"Friday", "Saturday", "Sunday"}


# ==============================
# 3. Funciones de normalización
# ==============================
# Estas funciones hacen que la app pueda usar español en la interfaz,
# pero el modelo reciba valores compatibles con el entrenamiento.
def normalize_event_day(event_day: Any) -> str:
    """Convierte días en español o inglés a la forma usada por el modelo."""
    if event_day is None:
        return "Saturday"

    value = str(event_day).strip()
    return DAY_NORMALIZATION.get(value.lower(), value)


def infer_country(city: Any, country: Any = None) -> str:
    """Obtiene el país a partir de la ciudad si el usuario no lo proporciona."""
    if country is not None and str(country).strip():
        return str(country).strip()

    city_value = str(city).strip()
    return CITY_COUNTRY.get(city_value, "Mexico")


def infer_previous_attendance_rate(artist: Any, value: Any = None) -> float:
    """Obtiene el historial previo del artista si no viene en la entrada."""
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass

    artist_value = str(artist).strip()
    return float(ARTIST_ATTENDANCE_RATE.get(artist_value, 0.70))


def infer_artist_market_score(city: Any, value: Any = None) -> float:
    """Obtiene el score de mercado de la ciudad si no viene en la entrada."""
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass

    city_value = str(city).strip()
    return float(CITY_MARKET_SCORE.get(city_value, 0.80))


def estimate_pre_sale_interest(
    capacity: float,
    previous_attendance_rate: float,
    artist_market_score: float,
    artist_popularity: float,
    value: Any = None,
) -> float:
    """
    Estima interés de preventa cuando no se captura manualmente.

    ¿Por qué se calcula?
    El modelo final usa pre_sale_interest porque representa una señal previa al evento.
    En producción podría venir de búsquedas, registros, preventas o visitas.
    En el MVP académico se estima de manera determinista para que la app funcione.
    """
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass

    estimated = (
        capacity
        * previous_attendance_rate
        * artist_market_score
        * (artist_popularity / 100)
        * 0.85
    )

    # No permitimos valores negativos ni mayores a la capacidad del recinto.
    return float(max(0, min(capacity, round(estimated))))


# ==============================
# 4. Feature engineering
# ==============================
# Aquí se calculan variables derivadas que el modelo aprendió durante el entrenamiento.
def build_feature_row(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye una fila completa con todas las variables que espera el modelo.

    Entrada esperada mínima:
    artist, genre, city, venue_type, capacity, month, event_day,
    days_until_event, marketing_budget, ticket_price, artist_popularity.

    Salida:
    Diccionario con las 19 columnas de FEATURES.
    """
    capacity = float(event.get("capacity", 0))
    marketing_budget = float(event.get("marketing_budget", 0))
    ticket_price = float(event.get("ticket_price", 0))
    artist_popularity = float(event.get("artist_popularity", 0))

    artist = event.get("artist", "Unknown")
    genre = event.get("genre", "Unknown")
    city = event.get("city", "CDMX")
    venue_type = event.get("venue_type", "arena")
    event_day = normalize_event_day(event.get("event_day", "Saturday"))

    country = infer_country(city, event.get("country"))
    previous_attendance_rate = infer_previous_attendance_rate(
        artist,
        event.get("previous_attendance_rate"),
    )
    artist_market_score = infer_artist_market_score(
        city,
        event.get("artist_market_score"),
    )
    pre_sale_interest = estimate_pre_sale_interest(
        capacity=capacity,
        previous_attendance_rate=previous_attendance_rate,
        artist_market_score=artist_market_score,
        artist_popularity=artist_popularity,
        value=event.get("pre_sale_interest"),
    )

    is_weekend = 1 if event_day in WEEKEND_DAYS else 0

    # Evitamos división entre cero por seguridad.
    marketing_per_capacity = marketing_budget / capacity if capacity else 0
    price_per_capacity = ticket_price / capacity if capacity else 0
    popularity_marketing = artist_popularity * marketing_budget

    row = {
        "artist": artist,
        "genre": genre,
        "city": city,
        "country": country,
        "venue_type": venue_type,
        "capacity": capacity,
        "month": int(event.get("month", 1)),
        "event_day": event_day,
        "days_until_event": int(event.get("days_until_event", 0)),
        "marketing_budget": marketing_budget,
        "ticket_price": ticket_price,
        "artist_popularity": artist_popularity,
        "previous_attendance_rate": previous_attendance_rate,
        "pre_sale_interest": pre_sale_interest,
        "artist_market_score": artist_market_score,
        "is_weekend": is_weekend,
        "marketing_per_capacity": marketing_per_capacity,
        "price_per_capacity": price_per_capacity,
        "popularity_marketing": popularity_marketing,
    }

    return row


def build_feature_dataframe(events: Dict[str, Any] | Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convierte uno o varios eventos a DataFrame con el orden correcto de columnas."""
    if isinstance(events, dict):
        rows = [build_feature_row(events)]
    else:
        rows = [build_feature_row(event) for event in events]

    df = pd.DataFrame(rows)

    # Reordenamos las columnas para que coincidan exactamente con el entrenamiento.
    return df[FEATURES]


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset para entrenamiento.

    ¿Por qué existe?
    Si el dataset ya trae las variables históricas, las respeta.
    Si faltan algunas variables derivadas, las calcula para evitar errores.
    """
    prepared_rows = []

    for _, row in df.iterrows():
        event = row.to_dict()
        feature_row = build_feature_row(event)

        # Conservamos el target si existe.
        if TARGET in row:
            feature_row[TARGET] = row[TARGET]

        prepared_rows.append(feature_row)

    return pd.DataFrame(prepared_rows)


def validate_training_columns(df: pd.DataFrame) -> None:
    """Valida que el dataset tenga el target y las columnas necesarias."""
    if TARGET not in df.columns:
        raise ValueError(f"El dataset debe contener la columna target: {TARGET}")

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para entrenar: {missing}")
