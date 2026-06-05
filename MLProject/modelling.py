import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# Tracking MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_AlyaSekarDwinurama")

# Autolog
mlflow.sklearn.autolog()

def train_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        BASE_DIR,
        "heart_preprocessing.csv"
    )

    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Sedang melatih model...")

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()
