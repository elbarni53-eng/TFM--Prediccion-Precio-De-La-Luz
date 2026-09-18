"""Entrenamiento de los modelos de gradient boosting y del modelo
estadistico clasico (secciones 6.2.2 y 6.2.3 de la memoria).

La red LSTM se entrena por separado en `modelos_lstm.py` porque su pipeline
de datos (ventanas deslizantes, normalizacion, Dataset de PyTorch) es lo
bastante distinto como para no compartir funciones utiles con los modelos
tabulares de este modulo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import FEATURES_MODELO, TARGET

# Hiperparametros de gradient boosting (seccion 6.2.3). Se comparten entre
# XGBoost y LightGBM para que la comparativa entre ambos aisle, en la
# medida de lo posible, la diferencia entre implementaciones y no una
# diferencia de esfuerzo de ajuste de hiperparametros.
GB_PARAMS = dict(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
)
PARADA_TEMPRANA_RONDAS = 50

# Especificacion SARIMAX (seccion 6.2.2): orden (2,0,1) x (1,0,1)_24.
SARIMAX_ORDER = (2, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 1, 24)
SARIMAX_MAX_ITER = 50
SARIMAX_VENTANA_ENTRENAMIENTO_HORAS = 17_520  # ultimos dos anios de train

# Variables exogenas de SARIMAX (subconjunto reducido respecto a
# FEATURES_MODELO: un modelo estadistico clasico con 23 exogenas no
# converge en un tiempo razonable sobre series de esta longitud).
SARIMAX_EXOG = [
    "hora_sin", "hora_cos", "dow_sin", "dow_cos", "festivo",
    "demanda_prevista_rel", "temperatura_media", "gas_ttf", "co2_eua",
]


def entrenar_xgboost(train: pd.DataFrame, val: pd.DataFrame) -> xgb.XGBRegressor:
    modelo = xgb.XGBRegressor(
        **GB_PARAMS,
        early_stopping_rounds=PARADA_TEMPRANA_RONDAS,
        eval_metric="mae",
    )
    modelo.fit(
        train[FEATURES_MODELO], train[TARGET],
        eval_set=[(val[FEATURES_MODELO], val[TARGET])],
        verbose=False,
    )
    print(f"XGBoost: parada en la iteracion {modelo.best_iteration} de {GB_PARAMS['n_estimators']}")
    return modelo


def entrenar_lightgbm(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMRegressor:
    modelo = lgb.LGBMRegressor(**GB_PARAMS, verbose=-1)
    modelo.fit(
        train[FEATURES_MODELO], train[TARGET],
        eval_set=[(val[FEATURES_MODELO], val[TARGET])],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(PARADA_TEMPRANA_RONDAS, verbose=False)],
    )
    print(f"LightGBM: parada en la iteracion {modelo.best_iteration_} de {GB_PARAMS['n_estimators']}")
    return modelo


def entrenar_y_predecir_sarimax(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Entrena SARIMAX sobre los ultimos dos anios de `train` y evalua sobre
    `test` mediante un esquema walk-forward de bloques mensuales: cada
    bloque de 30 dias se predice de forma dinamica a partir del estado
    actual del modelo, y a continuacion ese estado se actualiza con los
    valores reales del bloque (sin reoptimizar los parametros) mediante
    `extend`, mucho mas eficiente en memoria que `append`.

    Nota (seccion 6.2.2): el optimizador de maxima verosimilitud no siempre
    converge por completo dentro de `SARIMAX_MAX_ITER` iteraciones en series
    de esta longitud con un termino estacional de periodo 24; se documenta
    aqui con transparencia en lugar de subir el limite artificialmente
    hasta ocultar el aviso de convergencia.
    """
    ventana_train = train.iloc[-SARIMAX_VENTANA_ENTRENAMIENTO_HORAS:]

    modelo = SARIMAX(
        ventana_train[TARGET],
        exog=ventana_train[SARIMAX_EXOG],
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    resultado = modelo.fit(maxiter=SARIMAX_MAX_ITER, disp=False)

    predicciones = []
    bloques = pd.date_range(test["datetime"].min(), test["datetime"].max(), freq="30D")
    for inicio_bloque in bloques:
        fin_bloque = min(inicio_bloque + pd.Timedelta(days=30), test["datetime"].max())
        bloque = test[(test["datetime"] >= inicio_bloque) & (test["datetime"] < fin_bloque)]
        if bloque.empty:
            continue

        pred_bloque = resultado.get_forecast(
            steps=len(bloque), exog=bloque[SARIMAX_EXOG]
        ).predicted_mean
        predicciones.append(pd.Series(pred_bloque.values, index=bloque.index))

        # Actualiza el estado del filtro con los valores reales observados,
        # sin reoptimizar los parametros del modelo.
        resultado = resultado.extend(bloque[TARGET], exog=bloque[SARIMAX_EXOG])

    return pd.concat(predicciones).sort_index().values
