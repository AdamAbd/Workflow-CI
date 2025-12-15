import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="namadataset_preprocessing/titanic_preprocessing.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=3000)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    if "Survived" not in df.columns:
        raise ValueError("Kolom target 'Survived' tidak ditemukan.")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    mlflow.set_experiment("CI-Training-MLflow-Project")

    # Autolog boleh untuk Kriteria 3 Basic (yang penting MLflow Project + CI jalan)
    mlflow.sklearn.autolog(log_models=True)

    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Tambahan manual metrics (aman)
    with mlflow.start_run(run_name="ci_eval_metrics"):
        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))
        mlflow.log_metric("test_auc", float(auc))

    print("✅ Training selesai")
    print("Accuracy:", acc)
    print("F1:", f1)
    print("AUC:", auc)


if __name__ == "__main__":
    main()
