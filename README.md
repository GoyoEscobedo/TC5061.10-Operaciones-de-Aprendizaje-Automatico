# TC5061 Experimentación con MLflow

Actividad individual de Gregorio Escobedo: migración del modelo base de Bank Marketing a un script parametrizado, registro de experimentos y comparación de configuraciones.

## Ejecutar

Utiliza Python 3.13. Desde la raíz del repositorio:

```bash
cd actividad_mlflow
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python prepare_data.py
python train.py
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

El script funciona sin argumentos. También permite cambiar hiperparámetros y semilla:

```bash
python train.py --C 1.0 --penalty l1 --class-weight none --max-iter 1000 --seed 42
python train.py --help
```

## Reproducir la comparación

```bash
python run_experiments.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Abre http://127.0.0.1:5001 y selecciona el experimento **Bank Marketing - logistica parametrizada prellamada**. El programa ejecuta cinco configuraciones distintas, repite el baseline para comprobar reproducibilidad y evalúa al ganador en prueba. Cada ejecución adicional crea nuevas corridas.

## Archivos de la actividad

- [Script de entrenamiento](actividad_mlflow/train.py), con carga, preparación, entrenamiento, evaluación y MLflow Tracking.
- [Ejecución de experimentos](actividad_mlflow/run_experiments.py).
- [Notebook de exploración](actividad_mlflow/exploracion_bank_marketing.ipynb).
- [Dependencias directas](actividad_mlflow/requirements.txt) y [entorno con versiones fijas](actividad_mlflow/requirements-lock.txt).
- [Instrucciones detalladas](actividad_mlflow/README.md).
- [Análisis de resultados](actividad_mlflow/analisis_resultados.txt).
- [Capturas y evidencia](actividad_mlflow/evidencias), incluyendo comparación, detalle de artefactos, métricas y comprobación de reproducibilidad.

La copia del notebook anterior está en `actividad_mlflow/referencias/` como antecedente. El notebook de trabajo de esta actividad se dedica exclusivamente al EDA.

Los registros de MLflow, los modelos y las predicciones se generan localmente al ejecutar los scripts. Se incluyen las evidencias del experimento realizado; las rutas locales de los manifiestos documentan aquella ejecución y no son rutas portables de descarga.

## Datos

Se incluye `bank-full.csv`, del conjunto [Bank Marketing de UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing). Si falta, `prepare_data.py` lo descarga desde la fuente oficial.

Moro, S., Rita, P. y Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. DOI: 10.24432/C5K306. Licencia CC BY 4.0.
