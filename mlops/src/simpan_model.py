import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Membuat data dan melatih model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Konfigurasi URI MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Menggunakan alamat server MLflow yang baru
mlflow.set_experiment("MLOps Experiment")  # Buat eksperimen baru

# Log model ke MLflow
with mlflow.start_run(run_name="Register Random Forest Model"):
    mlflow.sklearn.log_model(model, "random_forest_model", registered_model_name="test_model")  # Simpan ke registry
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

print("Model berhasil didaftarkan ke MLflow Model Registry.")
