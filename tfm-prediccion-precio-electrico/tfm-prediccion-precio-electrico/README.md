# Predicción del precio horario del mercado eléctrico español

Código del Trabajo de Fin de Máster *"Predicción del precio horario del
mercado eléctrico español: análisis comparativo de modelos de Machine
Learning y Deep Learning aplicados al mercado SPOT diario de OMIE"*
(Yassin Ettijani El Barni).

El trabajo compara seis familias de modelos para predecir el precio
horario del mercado diario español a 24 horas vista (*day-ahead
forecasting*), sobre un conjunto de datos real y continuo de 2018 a 2026
construido a partir de las APIs públicas de ESIOS y AEMET. La aportación
central es un **modelo híbrido adaptativo** de elaboración propia que
combina LightGBM con un baseline de persistencia mediante un peso dinámico
basado en la volatilidad reciente del mercado, diseñado para no degradarse
ante cambios de régimen (como el apagón ibérico del 28 de abril de 2025)
de la misma forma que los modelos puramente estadísticos o de aprendizaje
automático.

La memoria completa está en la raíz de este repositorio
(`TFM_Yassin_Ettijani_El_Barni.pdf`).

## Resultado principal

Sobre el conjunto de test (2025-2026, régimen posterior al apagón, nunca
visto en entrenamiento), el modelo híbrido obtiene el mejor MAE de los
ocho modelos comparados (45,01 €/MWh), un 5,7 % mejor que el mejor
baseline naive y un 21,7 % mejor que LightGBM puro (57,50 €/MWh) — pese a
que LightGBM es netamente superior en la partición de validación. El
capítulo 7 de la memoria detalla esta comparativa con significación
estadística (contraste de Diebold-Mariano) y un análisis de explicabilidad
SHAP.

## Estructura del repositorio

```
src/
├── descarga_datos.py         Descarga ESIOS + AEMET (Fase 1 del pipeline)
├── calidad_datos.py          Corrección del problema de agregación de ESIOS (sección 6.1)
├── features.py                Ingeniería de variables (sección 5.3)
├── evaluacion.py               Métricas, partición temporal y contraste de Diebold-Mariano
├── modelos_baseline.py         Baselines naive
├── modelos_entrenamiento.py    XGBoost, LightGBM, SARIMAX
├── modelos_lstm.py             Red LSTM
├── modelo_hibrido.py           El modelo híbrido adaptativo (aportación central del TFM)
├── explicabilidad_shap.py      Análisis SHAP
├── generar_figuras.py          Generación de las figuras del capítulo 7
└── pipeline_completo.py        Orquesta todo lo anterior de principio a fin
data/
└── README.md                  Cómo obtener y reconstruir los datos (no incluidos en el repo)
```

## Uso rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # y rellena tus tokens de ESIOS y AEMET
export $(grep -v '^#' .env | xargs)

python -m src.descarga_datos --inicio 2018-01-01
# (integra los ficheros de data/raw/ en data/processed/dataset_maestro.parquet — ver data/README.md)

python -m src.pipeline_completo
```

Esto entrena los ocho modelos, calcula las métricas y el contraste de
Diebold-Mariano de la tabla 7.1 y la figura 7.6, y regenera la figura 7.9.
El entrenamiento completo (SARIMAX walk-forward incluido) tarda del orden
de 15-30 minutos en un portátil sin GPU; con GPU, la LSTM se acelera
notablemente.

## Sobre los datos

Este repositorio **no** incluye datos ni modelos entrenados: ESIOS y AEMET
requieren credenciales personales gratuitas y sus términos de uso
desaconsejan redistribuir el histórico completo. `data/README.md` explica
cómo obtener las credenciales y reconstruir el dataset maestro desde cero.

## Requisitos

Python ≥ 3.10. Ver `requirements.txt`. LightGBM y XGBoost usan CPU por
defecto; PyTorch detecta GPU automáticamente si está disponible.

## Licencia

Código publicado bajo licencia MIT (ver `LICENSE`). La memoria (PDF) se
publica con todos los derechos reservados salvo mención expresa.
