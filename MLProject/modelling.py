import os
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# Mengatur alamat MLflow Tracking UI ke lokalhost
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_AlyaSekarDwinurama")

mlflow.sklearn.autolog()

def train_model():
    # Tentukan path dataset bersih
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
    BASE_DIR, "heart_preprocessing.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    # Memuat data bersih
    df = pd.read_csv(data_path)
    
    # Pisahkan Fitur (X) dan Target (y)
    # Kolom 'target' 0 = Sehat, 1 = Sakit
    X = df.drop(columns=['target'])
    y = df['target']
    
    #Split data menjadi Training dan Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # mulai proses training di MLflow
    with mlflow.start_run(nested=True):
        print("Sedang melatih model")
        
        # Inisialisasi model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model")

        #prediksi model
        y_pred = model.predict(X_test)
        
        # evaluasi score sederhana
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Training Selesai!")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Accurancy Score: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()
