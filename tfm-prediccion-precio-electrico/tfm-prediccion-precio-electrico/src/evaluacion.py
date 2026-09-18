"""Metricas, particion temporal y contraste de Diebold-Mariano (seccion 5.4,
5.6 y 5.6.1 de la memoria).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Esquema de validacion temporal (seccion 5.4 / tabla 5.1): entrenamiento
# 2018-2023, validacion 2024, test 2025-2026 (regimen post-apagon, nunca
# visto en entrenamiento ni en validacion).
TRAIN_INICIO, TRAIN_FIN = "2018-01-08", "2023-12-31 23:00"
VAL_INICIO, VAL_FIN = "2024-01-01", "2024-12-31 23:00"
TEST_INICIO, TEST_FIN = "2025-01-01", "2026-07-31 23:00"


def particionar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train / validacion / test segun el esquema
    cronologico fijo de la memoria. `df` debe tener una columna `datetime`.
    """
    indexado = df.set_index("datetime")
    train = indexado.loc[TRAIN_INICIO:TRAIN_FIN]
    val = indexado.loc[VAL_INICIO:VAL_FIN]
    test = indexado.loc[TEST_INICIO:TEST_FIN]
    return train.reset_index(), val.reset_index(), test.reset_index()


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """sMAPE, preferible al MAPE clasico cuando existen valores cercanos a
    cero o negativos, como las horas de precio negativo de este mercado.
    """
    denominador = (np.abs(y) + np.abs(yhat)) / 2
    denominador = np.where(denominador < 1e-6, 1e-6, denominador)
    return float(100 * np.mean(np.abs(y - yhat) / denominador))


def resumen_metricas(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "sMAPE": smape(y, yhat)}


def metricas_por_grupo(
    df_eval: pd.DataFrame, columna_grupo: str, y_col: str = "y_real", yhat_col: str = "y_pred"
) -> pd.DataFrame:
    """Metricas agregadas por hora del dia, mes, o cualquier otra columna
    categorica (usado en las figuras 7.5 y 7.9, MAE por hora/mes).
    """
    filas = []
    for grupo, subconjunto in df_eval.groupby(columna_grupo):
        m = resumen_metricas(subconjunto[y_col].values, subconjunto[yhat_col].values)
        m[columna_grupo] = grupo
        m["n"] = len(subconjunto)
        filas.append(m)
    return pd.DataFrame(filas).set_index(columna_grupo)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """Contraste de Diebold-Mariano sobre la perdida cuadratica de dos
    modelos (seccion 5.6.1).

    e1, e2: arrays de errores de prediccion (y_real - y_pred) de los
    modelos 1 y 2, alineados en el tiempo.

    Devuelve (estadistico_DM, p_valor). Un estadistico negativo indica que
    el modelo 1 tiene, en promedio, menor perdida cuadratica que el modelo
    2; un p-valor por debajo de 0.05 indica que esa diferencia es
    estadisticamente significativa al 95% de confianza.

    Nota metodologica (seccion 7.6 de la memoria): esta implementacion usa
    la varianza muestral simple de la serie de diferencias `d` como
    estimador de su varianza de largo plazo, sin correccion de
    autocorrelacion (HAC/Newey-West), valida bajo el supuesto h=1 (horizonte
    de prediccion de un paso) de errores serialmente no correlacionados.
    Con una muestra de test grande (13 848 observaciones horarias en este
    trabajo), el contraste gana potencia estadistica muy rapidamente, de
    forma que incluso diferencias de precision modestas en terminos
    absolutos resultan significativas en terminos estadisticos: esta es la
    explicacion mas probable de que todas las comparaciones por pares de la
    figura 7.6 resulten significativas pese a que algunos modelos tengan
    MAE de test relativamente proximos entre si. Quien reutilice esta
    funcion sobre una serie con dependencia temporal fuerte en `d` (por
    ejemplo, errores que se agrupan en rachas por dia) deberia sustituir
    `gamma0` por un estimador HAC (ver `statsmodels.stats.stattools` o
    `arch.bootstrap`) para no subestimar la varianza y sobre-rechazar la
    hipotesis nula.
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_media = d.mean()
    gamma0 = np.var(d, ddof=0)
    estadistico_dm = d_media / np.sqrt(gamma0 / n)
    p_valor = 2 * (1 - norm.cdf(np.abs(estadistico_dm)))
    return float(estadistico_dm), float(p_valor)


def matriz_diebold_mariano(predicciones: dict[str, np.ndarray], y_real: np.ndarray) -> pd.DataFrame:
    """Calcula la matriz completa de p-valores de Diebold-Mariano entre
    todos los pares de modelos en `predicciones` (usada para la figura 7.6).

    predicciones: diccionario {nombre_modelo: array_de_predicciones}.
    """
    nombres = list(predicciones.keys())
    errores = {nombre: y_real - pred for nombre, pred in predicciones.items()}
    matriz = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
    for i in nombres:
        for j in nombres:
            if i == j:
                matriz.loc[i, j] = np.nan
                continue
            _, p_valor = diebold_mariano(errores[i], errores[j])
            matriz.loc[i, j] = p_valor
    return matriz
