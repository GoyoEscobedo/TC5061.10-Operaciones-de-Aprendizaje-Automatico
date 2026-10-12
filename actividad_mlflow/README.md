# Actividad individual: experimentación con MLflow

Migración de `TC5061_Notebook_Baseline_EscobedoGregorio.ipynb`, de Gregorio Escobedo.
Se conserva la regresión logística base, la estandarización, la codificación one-hot,
las categorías `unknown` y los valores extremos. Se excluyen `contact`, `day`,
`month`, `duration`, `campaign` y `poutcome`, como en el notebook entregado.

## Ejecutar desde la línea de comandos

Utiliza Python 3.13 y ejecuta los comandos dentro de esta carpeta:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python prepare_data.py
python train.py
```

El CSV se incluye en `data/bank-full.csv`; el comando de descarga conserva el
archivo existente. `train.py` no requiere argumentos y no necesita un servidor
MLflow en ejecución: escribe directamente en la base SQLite local.

Ejemplo con argumentos:

```bash
python train.py --C 0.01 --penalty l1 --class-weight balanced --max-iter 1000 --seed 42 --run-name ejemplo
python train.py --help
```

Los hiperparámetros disponibles son `C` (inverso de la fuerza de regularización),
`penalty` (L1/L2), `class-weight` y `max-iter`. Se fija `solver=liblinear`.
En scikit-learn 1.9 el argumento `penalty` del script se traduce a `l1_ratio`:
1 para L1 y 0 para L2, evitando el parámetro obsoleto de la biblioteca.

## Diseño de la comparación

```bash
python run_experiments.py
```

Se ejecutan cinco configuraciones distintas: baseline, C=0.01, C=100, L1 y pesos
balanceados. Cada variante cambia un factor respecto al baseline. Una sexta
corrida repite el baseline y comprueba igualdad exacta de todas las métricas y
del CSV de predicciones. Los nombres y argumentos de las seis corridas están
en `run_experiments.py`; las salidas completas se guardan en `evidencias/logs/`.
Cada nueva ejecución crea corridas adicionales, sin sobrescribir las anteriores.

Todas usan particiones estratificadas idénticas: 60 % ajuste, 20 % validación y
20 % prueba. Se fija semilla 42 para la primera división y 43 para la segunda.
La preparación se ajusta exclusivamente con los datos de ajuste. Se elige por
AP de validación porque el objetivo es priorizar llamadas con una clase positiva
minoritaria. F1, precisión y recall se calculan con umbral 0.5.

Solo el ganador se reajusta con el 80 % de entrenamiento y se evalúa en el 20 %
de prueba, mediante una séptima corrida. La opción `--evaluate-test` está destinada
a esa evaluación final. Los resultados del baseline en validación no son
directamente comparables con los resultados de prueba del notebook anterior.

## Interfaz de MLflow y evidencia

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Abre http://127.0.0.1:5001 y el experimento **Bank Marketing - logistica
parametrizada prellamada**. En **Runs**, selecciona las cinco configuraciones y
pulsa **Compare**. Contrasta `C`, `penalty`, `class_weight`, `validation_AP`,
`validation_f1_yes` y `validation_recall_yes`. La corrida repetida verifica la
reproducibilidad; la corrida final de prueba se interpreta por separado.

La entrega contiene:

- `train.py`: carga, particiones, preparación, entrenamiento y evaluación en funciones.
- `exploracion_bank_marketing.ipynb`: notebook dedicado únicamente al EDA, sin resultados antiguos.
- `referencias/notebook_baseline_original.ipynb`: copia de referencia del trabajo anterior;
  el notebook de trabajo para esta actividad es únicamente el de exploración.
- `requirements.txt`: versiones exactas de las dependencias directas.
- `requirements-lock.txt`: versiones de todas las dependencias del entorno utilizado.
- `analisis_resultados.txt`: análisis breve de los resultados obtenidos.
- `evidencias/`: capturas, tabla CSV, comprobación de reproducibilidad y manifiesto de corridas.
- `mlflow.db` y `mlartifacts/`: registros y artefactos persistentes de MLflow,
  generados localmente al ejecutar los scripts; no se versionan en Git.
- `outputs/`: copia local de métricas, predicciones, gráficas y configuración por corrida,
  generada al ejecutar los scripts; no se versiona en Git.

## Reproducibilidad y trazabilidad

Se fija la semilla en Python, NumPy, el modelo y las particiones; las bibliotecas
numéricas usan un hilo. Los empates del ranking se resuelven con un orden estable.
La comprobación realizada se refiere a los mismos datos, argumentos, código y
entorno; una semilla aislada no garantiza igualdad en otras versiones o equipos.
Cada corrida registra SHA-256 del CSV, del script y de las particiones, además
del entorno y una copia del código. El commit de Git es información adicional;
la huella y el script guardado identifican también cambios sin commit.
El parámetro `model_training_rows` identifica cuántos registros se utilizaron
realmente para entrenar el modelo: 27,126 durante la comparación y 36,168 en la
evaluación final. Cada ejecución ajusta el modelo una sola vez.

Se registra el pipeline completo con `mlflow.sklearn.log_model`, incluyendo la
firma, ejemplo de entrada y dependencias. Sus artefactos incluyen matriz de
confusión, curva precisión-recall, métricas JSON y predicciones CSV. El modelo
queda vinculado a una corrida con URI propia; esta actividad no exige crear
versiones en Model Registry.

Para mover la entrega a otro equipo, copia el código y los datos y vuelve a
ejecutar las corridas. Las ubicaciones de artefactos existentes en la base SQLite
son absolutas; copiar solo la base no traslada los modelos ni corrige esas rutas.

## Fuentes

- [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing),
  Moro, S., Rita, P. y Cortez, P. (2014), DOI: 10.24432/C5K306, licencia CC BY 4.0.
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).
- [LogisticRegression, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
