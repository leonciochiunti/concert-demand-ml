from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    features = [
        "artist", "genre", "city", "venue_type", "capacity", "month",
        "event_day", "days_until_event", "marketing_budget",
        "ticket_price", "artist_popularity"
    ]
    target = "demand_class"

    X = df[features]
    y = df[target]

    categorical_features = ["artist", "genre", "city", "venue_type", "event_day"]
    numeric_features = [
        "capacity", "month", "days_until_event", "marketing_budget",
        "ticket_price", "artist_popularity"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
