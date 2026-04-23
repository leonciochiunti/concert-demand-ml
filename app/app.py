from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

st.set_page_config(page_title="Concert Demand ML", page_icon="🎤", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def simple_recommendations(df: pd.DataFrame, genre: str, city: str, artist: str):
    recs = df[
        (df["genre"] == genre) &
        (df["city"] == city) &
        (df["artist"] != artist)
    ][["artist", "genre", "city", "ticket_price", "demand_class"]].head(5)
    if recs.empty:
        recs = df[df["genre"] == genre][["artist", "genre", "city", "ticket_price", "demand_class"]].head(5)
    return recs

st.title("🎤 Predicción de Demanda de Conciertos")
st.write("MVP académico: clasificación de demanda **baja / media / alta**.")

df = load_data()

if not MODEL_PATH.exists():
    st.warning("No se encontró el modelo entrenado. Ejecuta: `python model/train.py`")
    st.stop()

model = load_model()

artists = sorted(df["artist"].unique())
genres = sorted(df["genre"].unique())
cities = sorted(df["city"].unique())
venues = sorted(df["venue_type"].unique())
days = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"]

with st.form("prediction_form"):
    artist = st.selectbox("Artista", artists)
    genre = st.selectbox("Género", genres)
    city = st.selectbox("Ciudad", cities)
    venue_type = st.selectbox("Tipo de venue", venues)
    capacity = st.slider("Capacidad", 3000, 60000, 12000, step=1000)
    month = st.slider("Mes", 1, 12, 7)
    event_day = st.selectbox("Día del evento", days)
    days_until_event = st.slider("Días hasta el evento", 7, 180, 45)
    marketing_budget = st.slider("Presupuesto de marketing", 20000, 250000, 80000, step=5000)
    ticket_price = st.slider("Precio del boleto", 350, 4500, 1500, step=50)
    artist_popularity = st.slider("Popularidad del artista", 60, 100, 85)

    submitted = st.form_submit_button("Predecir demanda")

if submitted:
    input_data = pd.DataFrame([{
        "artist": artist,
        "genre": genre,
        "city": city,
        "venue_type": venue_type,
        "capacity": capacity,
        "month": month,
        "event_day": event_day,
        "days_until_event": days_until_event,
        "marketing_budget": marketing_budget,
        "ticket_price": ticket_price,
        "artist_popularity": artist_popularity,
    }])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    classes = list(model.classes_)

    st.subheader("Resultado")
    st.success(f"Demanda predicha: **{prediction.upper()}**")

    st.write("Probabilidades por clase:")
    proba_df = pd.DataFrame({"Clase": classes, "Probabilidad": proba}).sort_values("Probabilidad", ascending=False)
    st.dataframe(proba_df, use_container_width=True)

    st.subheader("Recomendación simple")
    recs = simple_recommendations(df, genre=genre, city=city, artist=artist)
    st.dataframe(recs, use_container_width=True)

st.markdown("---")
st.caption("Proyecto MVP de maestría: predicción de demanda + recomendación ligera.")
