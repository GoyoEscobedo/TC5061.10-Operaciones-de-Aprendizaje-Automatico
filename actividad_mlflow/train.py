"""Regresión logística del notebook base, parametrizada y registrada en MLflow."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import warnings

# Fijar concurrencia antes de importar las bibliotecas numéricas.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "Bank Marketing - logistica parametrizada prellamada"
EXCLUDED = ["contact", "day", "month", "duration", "campaign", "poutcome"]
NUMERIC = ["age", "balance", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan"]
PACKAGES = ["mlflow", "scikit-learn", "pandas", "numpy", "matplotlib", "cloudpickle", "threadpoolctl"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/bank-full.csv", help="CSV de UCI separado por ;")
    parser.add_argument("--C", type=float, default=1.0, help="Inverso de la fuerza de regularización")
    parser.add_argument("--penalty", choices=["l1", "l2"], default="l2", help="Tipo de regularización")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Pesos de las clases")
    parser.add_argument("--max-iter", type=int, default=1000, help="Máximo de iteraciones del optimizador")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del modelo y de las particiones")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--run-name", default="01_baseline")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluar prueba solo después de seleccionar la configuración")
    args = parser.parse_args()
    if not np.isfinite(args.C) or args.C <= 0 or args.max_iter < 1 or not 0 <= args.seed < 2**32 - 1:
        parser.error("C debe ser finito y positivo; max-iter positivo; seed entre 0 y 2**32-2")
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero: python prepare_data.py")
    frame = pd.read_csv(path, sep=";")
    required = set(NUMERIC + CATEGORICAL + EXCLUDED + ["y"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any() or set(frame.y.unique()) != {"yes", "no"}:
        raise ValueError("Se esperan datos sin nulos y objetivo con ambas categorías yes/no")
    X = frame.drop(columns=EXCLUDED + ["y"])
    return X, frame.y.eq("yes").astype(int)


def split_data(X, y, seed):
    # Igual que en el notebook: 20 % de prueba; validación = 25 % del 80 % restante.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed + 1, stratify=y_train)
    return {"train": (X_fit, y_fit), "validation": (X_val, y_val), "test": (X_test, y_test),
            "train_full": (X_train, y_train)}


def build_model(args):
    preprocessing = ColumnTransformer([
        ("numericas", StandardScaler(), NUMERIC),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    # sklearn 1.9 expresa L1/L2 mediante l1_ratio; evita el argumento penalty obsoleto.
    classifier = LogisticRegression(C=args.C, l1_ratio=1.0 if args.penalty == "l1" else 0.0,
                                    class_weight=None if args.class_weight == "none" else "balanced",
                                    max_iter=args.max_iter, solver="liblinear", random_state=args.seed)
    return Pipeline([("preparacion", preprocessing), ("clasificador", classifier)])


def evaluate(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    count = max(1, int(len(y) * 0.10))
    # Orden estable para que los empates tengan una resolución determinista.
    top = np.argsort(-probabilities, kind="stable")[:count]
    rate = float(y.iloc[top].mean())
    metrics = {"accuracy": accuracy_score(y, predictions),
               "precision_yes": precision_score(y, predictions, zero_division=0),
               "recall_yes": recall_score(y, predictions, zero_division=0),
               "f1_yes": f1_score(y, predictions, zero_division=0),
               "AP": average_precision_score(y, probabilities),
               "tasa_top10": rate, "lift_top10": rate / float(y.mean())}
    return {key: float(value) for key, value in metrics.items()}, predictions, probabilities


def save_evaluation(directory, subset, y, predictions, probabilities, metrics):
    (directory / f"{subset}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"row_id": y.index, "y_true": y.to_numpy(), "y_pred": predictions,
                  "probability_yes": probabilities}).to_csv(directory / f"{subset}_predictions.csv", index=False, float_format="%.17g")
    ConfusionMatrixDisplay.from_predictions(y, predictions, display_labels=["no", "yes"], cmap="Blues", values_format="d")
    plt.title(f"Regresión logística — {subset}, umbral 0.5")
    plt.tight_layout()
    plt.savefig(directory / f"{subset}_confusion_matrix.png", dpi=160)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y, probabilities, name="Regresión logística")
    plt.title(f"Curva precisión-recall — {subset}")
    plt.tight_layout()
    plt.savefig(directory / f"{subset}_precision_recall.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    X, y = load_data(args.data)
    splits = split_data(X, y, args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(args.experiment)
    versions = {package: importlib.metadata.version(package) for package in PACKAGES}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    with mlflow.start_run(run_name=args.run_name) as run, threadpool_limits(limits=1):
        directory = ROOT / "outputs" / run.info.run_id
        directory.mkdir(parents=True)
        params = {"C": args.C, "penalty": args.penalty, "class_weight": args.class_weight,
                  "max_iter": args.max_iter, "seed": args.seed, "validation_seed": args.seed + 1,
                  "solver": "liblinear", "test_size": 0.20, "validation_size_training": 0.25,
                  "threshold": 0.5, "top_fraction": 0.10, "excluded_columns": ",".join(EXCLUDED),
                  "dataset_sha256": sha256(args.data), "train_py_sha256": sha256(__file__),
                  "selection_metric": "validation_AP", "evaluate_test": args.evaluate_test}
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.set_tags({"phase": "final_test" if args.evaluate_test else "comparison",
                         "git_commit": git.stdout.strip() if git.returncode == 0 else "unavailable",
                         "code_version": params["train_py_sha256"], "python_version": platform.python_version()})
        for name in ("train", "validation", "test"):
            indices = splits[name][0].index.to_numpy(dtype=np.int64)
            mlflow.log_param(f"{name}_rows", len(indices))
            mlflow.log_param(f"{name}_split_sha256", hashlib.sha256(indices.tobytes()).hexdigest())
        environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
                       "numeric_threads": 1}
        (directory / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        (directory / "configuration.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        model = build_model(args)
        # Elegir los datos antes de entrenar evita un ajuste que sería descartado.
        training_subset = "train_full" if args.evaluate_test else "train"
        X_fit, y_fit = splits[training_subset]
        mlflow.log_param("model_training_rows", len(X_fit))
        model.fit(X_fit, y_fit)
        subsets = ["test" if args.evaluate_test else "validation"]
        all_metrics = {}
        for subset in subsets:
            X_eval, y_eval = splits[subset]
            metrics, predictions, probabilities = evaluate(model, X_eval, y_eval)
            save_evaluation(directory, subset, y_eval, predictions, probabilities, metrics)
            for key, value in metrics.items():
                all_metrics[f"{subset}_{key}"] = value
                mlflow.log_metric(f"{subset}_{key}", value)
        for file in directory.iterdir():
            mlflow.log_artifact(str(file), artifact_path="evaluation")
        mlflow.log_artifact(__file__, artifact_path="source")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="source")
        model_info = mlflow.sklearn.log_model(model, name="modelo_logistico", serialization_format="cloudpickle",
                                             signature=infer_signature(X_fit.iloc[:5], model.predict(X_fit.iloc[:5])),
                                             input_example=X_fit.iloc[:5],
                                             pip_requirements=[f"{p}=={v}" for p, v in versions.items()])
        result = {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
                  "run_name": args.run_name, "model_uri": model_info.model_uri,
                  "params": params, "metrics": all_metrics}
        (directory / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(directory / "run.json"), artifact_path="evaluation")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
