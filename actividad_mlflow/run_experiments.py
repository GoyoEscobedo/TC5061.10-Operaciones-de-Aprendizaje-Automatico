"""Ejecuta cinco configuraciones, repite el baseline y evalúa al ganador en prueba."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "Bank Marketing - logistica parametrizada prellamada"
CONFIGURATIONS = [
    ("01_baseline", []),
    ("02_C_001", ["--C", "0.01"]),
    ("03_C_100", ["--C", "100"]),
    ("04_regularizacion_l1", ["--penalty", "l1"]),
    ("05_clases_balanceadas", ["--class-weight", "balanced"]),
    ("06_baseline_repetido", []),
]


def execute(name, arguments):
    command = [sys.executable, str(ROOT / "train.py"), *arguments]
    # La primera corrida se ejecuta literalmente sin argumentos.
    if name != "01_baseline":
        command.extend(["--run-name", name])
    print("Ejecutando:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    logs = ROOT / "evidencias" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Falló {name}. Consulta {logs / (name + '.log')}")
    return json.loads(result.stdout)


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    runs = [execute(name, arguments) for name, arguments in CONFIGURATIONS]
    evidence = ROOT / "evidencias"
    evidence.mkdir(exist_ok=True)
    rows = [{"run_name": r["run_name"], "run_id": r["run_id"],
             **{p: r["params"][p] for p in ("C", "penalty", "class_weight", "max_iter", "seed")},
             **r["metrics"]} for r in runs]
    table = pd.DataFrame(rows)
    table.to_csv(evidence / "comparacion.csv", index=False)
    a, b = runs[0], runs[-1]
    prediction_a = ROOT / "outputs" / a["run_id"] / "validation_predictions.csv"
    prediction_b = ROOT / "outputs" / b["run_id"] / "validation_predictions.csv"
    reproducible = a["metrics"] == b["metrics"] and prediction_a.read_bytes() == prediction_b.read_bytes()
    (evidence / "reproducibilidad.json").write_text(json.dumps({
        "run_original": a["run_id"], "run_repeated": b["run_id"],
        "metrics_identical": a["metrics"] == b["metrics"],
        "prediction_files_identical": prediction_a.read_bytes() == prediction_b.read_bytes(),
        "passed": reproducible,
    }, indent=2), encoding="utf-8")
    if not reproducible:
        raise RuntimeError("Las dos ejecuciones del baseline no coinciden")
    # Seleccionar solo entre las cinco configuraciones distintas, usando validación.
    winner = max(runs[:5], key=lambda r: r["metrics"]["validation_AP"])
    client = MlflowClient()
    client.set_tag(winner["run_id"], "selected_by", "highest validation_AP among five configurations")
    arguments = ["--C", str(winner["params"]["C"]), "--penalty", winner["params"]["penalty"],
                 "--class-weight", winner["params"]["class_weight"], "--evaluate-test"]
    final = execute("07_ganador_prueba", arguments)
    client.set_tag(final["run_id"], "selected_from_run", winner["run_id"])
    (evidence / "manifest.json").write_text(json.dumps({"comparison_runs": runs, "winner": winner,
                                                        "final_test": final}, indent=2), encoding="utf-8")
    print(table[["run_name", "validation_AP", "validation_f1_yes", "validation_recall_yes"]].to_string(index=False))
    print("Reproducibilidad: métricas y archivos de predicciones idénticos")
    print("Ganador por AP de validación:", winner["run_name"])
    print("Prueba final:", final["metrics"])


if __name__ == "__main__":
    main()
