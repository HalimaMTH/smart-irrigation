# ==============================
# MLOPS TRAINING PIPELINE
# SMART IRRIGATION
# ==============================

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Créer les dossiers nécessaires
os.makedirs("../models", exist_ok=True)
os.makedirs("../reports", exist_ok=True)

# 2. Charger les données traitées
X_train = pd.read_csv("../data/processed/X_train.csv")
X_test = pd.read_csv("../data/processed/X_test.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

# éviter le warning sklearn : y doit être 1D
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Données chargées avec succès")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 3. Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("intelligent_irrigation_training")

# 4. Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

# 5. Variables pour sauvegarder le meilleur modèle
best_model = None
best_model_name = None
best_f1_macro = 0

# 6. Entraînement et tracking
for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        print(f"\nEntraînement du modèle : {model_name}")

        # Training
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_weighted", precision_weighted)
        mlflow.log_metric("recall_weighted", recall_weighted)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        report_path = f"../reports/{model_name}_classification_report.csv"
        report_df.to_csv(report_path, index=True)

        mlflow.log_artifact(report_path)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = f"../reports/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(cm_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(model, name=model_name)

        # Sauvegarder le meilleur modèle selon f1_macro
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model = model
            best_model_name = model_name

        print(f"{model_name} terminé")
        print("Accuracy:", accuracy)
        print("F1 macro:", f1_macro)
        print("-" * 40)

# 7. Sauvegarde du meilleur modèle localement
joblib.dump(best_model, "../models/best_model.pkl")

# 8. Sauvegarde des infos du meilleur modèle
with open("../models/best_model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Best F1 macro: {best_f1_macro}\n")

print("\nMLOps training terminé")
print("Meilleur modèle :", best_model_name)
print("Meilleur F1 macro :", best_f1_macro)
print("Modèle sauvegardé dans ../models/best_model.pkl")