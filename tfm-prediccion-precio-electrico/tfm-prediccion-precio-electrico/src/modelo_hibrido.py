"""Modelo hibrido adaptativo (secciones 2.4.5 y 6.2.5 de la memoria).

Esta es la aportacion original de este TFM, no un modelo de referencia
tomado de la literatura. Combina, hora a hora, las predicciones ya
calculadas de LightGBM (el modelo estatico mas preciso) y del baseline
naive de 24 horas, mediante un peso dinamico lambda(t) que depende
unicamente de la volatilidad reciente observada del propio precio:

    sigma(t)   = desviacion tipica movil de 168h del precio, desplazada 24h
    lambda(t)  = 1 / (1 + (sigma(t) / sigma_ref) ** k)
    yhat(t)    = lambda(t) * yhat_LightGBM(t) + (1 - lambda(t)) * yhat_naive24(t)

Cuando la volatilidad reciente es baja (mercado en un regimen parecido al
de entrenamiento), lambda(t) tiende a 1 y el hibrido confia en LightGBM.
Cuando la volatilidad reciente se dispara muy por encima de lo visto en
entrenamiento (un cambio de regimen, como el apagon iberico de abril de
2025), lambda(t) cae hacia 0 y el hibrido se repliega hacia el baseline
robusto, que no tiene ninguna relacion aprendida que pueda dejar de ser
valida. No requiere entrenar ningun parametro adicional: sigma_ref y k se
fijan por diseno, antes de observar el conjunto de test (tabla 6.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VENTANA_VOLATILIDAD_HORAS = 168
DESPLAZAMIENTO_HORAS = 24
PERCENTIL_SIGMA_REF = 90
K_EXPONENTE = 2.5  # fijado por diseno; tabla 7.2 confirma robustez en [1.0, 4.0]


def calcular_sigma_ref(precio_train: pd.Series) -> float:
    """Calcula sigma_ref como el percentil 90 de la volatilidad movil,
    usando EXCLUSIVAMENTE la particion de entrenamiento (nunca validacion
    ni test), para no filtrar informacion futura hacia el diseno del
    mecanismo adaptativo.
    """
    sigma_train = precio_train.rolling(VENTANA_VOLATILIDAD_HORAS).std()
    return float(np.nanpercentile(sigma_train, PERCENTIL_SIGMA_REF))


def calcular_lambda(
    precio_particion: pd.Series, sigma_ref: float, k: float = K_EXPONENTE
) -> np.ndarray:
    """Calcula lambda(t) para una particion (validacion o test).

    El desplazamiento de 24 horas en sigma(t) es la misma restriccion de
    informacion disponible que rige el resto de variables anticipadas de
    la seccion 5.3.2: en el momento de predecir la hora t, solo se conoce
    la volatilidad hasta t-24h.
    """
    sigma_t = precio_particion.shift(DESPLAZAMIENTO_HORAS).rolling(VENTANA_VOLATILIDAD_HORAS).std()
    # Relleno hacia atras para las primeras horas de la particion, donde la
    # ventana movil aun no esta completa; equivale a asumir volatilidad de
    # referencia hasta que haya suficiente historial dentro de la propia
    # particion.
    sigma_t = sigma_t.bfill().fillna(sigma_ref)
    return 1.0 / (1.0 + (sigma_t.values / sigma_ref) ** k)


def predecir_hibrido(
    precio_particion: pd.Series,
    pred_lightgbm: np.ndarray,
    pred_naive_24h: np.ndarray,
    sigma_ref: float,
    k: float = K_EXPONENTE,
) -> tuple[np.ndarray, np.ndarray]:
    """Combina las predicciones ya calculadas de LightGBM y del baseline
    naive de 24h con el peso dinamico lambda(t).

    Devuelve (prediccion_hibrida, lambda_t) para poder inspeccionar y
    graficar lambda(t) por separado (figura 7.3).
    """
    lam = calcular_lambda(precio_particion, sigma_ref, k)
    prediccion = lam * pred_lightgbm + (1 - lam) * pred_naive_24h
    return prediccion, lam


def analisis_sensibilidad_k(
    precio_particion: pd.Series,
    pred_lightgbm: np.ndarray,
    pred_naive_24h: np.ndarray,
    y_real: np.ndarray,
    sigma_ref: float,
    valores_k: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
) -> pd.DataFrame:
    """Reproduce la tabla 7.2: MAE y RMSE de test para distintos valores de
    k, calculada a posteriori con fines de robustez, no para seleccionar k
    (que se fija por diseno antes de observar el conjunto de test).
    """
    from .evaluacion import mae, rmse  # import local para evitar ciclo

    filas = []
    for k in valores_k:
        pred, _ = predecir_hibrido(precio_particion, pred_lightgbm, pred_naive_24h, sigma_ref, k)
        filas.append({"k": k, "MAE": mae(y_real, pred), "RMSE": rmse(y_real, pred)})
    return pd.DataFrame(filas)
