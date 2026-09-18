"""Ingenieria de variables (seccion 5.3 de la memoria).

Construye, a partir del conjunto de datos maestro horario ya integrado y
corregido (ver `calidad_datos.py`), las veintitres variables explicativas
que alimentan a todos los modelos no lineales, organizadas en las cinco
familias de la tabla 5.1: lags y medias moviles del precio, codificacion
ciclica temporal, calendario, exogenas energeticas anticipadas y
meteorologia/materias primas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Filas descartadas al principio de la serie: sin lag de 168h valido
# (seccion 5.3.4). Con datos horarios equivale a exactamente una semana.
FILAS_CALENTAMIENTO = 168

FEATURES_MODELO = [
    # Codificacion ciclica temporal (seccion 5.3.3)
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "dow_sin", "dow_cos",
    # Calendario (seccion 2.2.5)
    "festivo", "finde",
    # Lags y medias moviles del precio (seccion 5.3.1, justificados por la
    # estructura de autocorrelacion de la seccion 4.4)
    "precio_lag24", "precio_lag48", "precio_lag168",
    "precio_ma24", "precio_ma168", "precio_std24",
    # Exogenas energeticas anticipadas (seccion 5.3.2)
    "demanda_prevista_rel", "eolica_prevista_proxy_rel", "solar_prevista_proxy_rel",
    "nuclear_medida_rel",
    # Meteorologia y materias primas (seccion 2.2.3 y 2.2.4)
    "temperatura_media", "temperatura_min", "temperatura_max",
    "viento_medio_ms", "precipitacion_mm",
    "gas_ttf", "co2_eua",
]

TARGET = "precio_spot"


def _regimen(fecha: pd.Timestamp) -> str:
    """Etiqueta de regimen segun los tres periodos caracterizados en el
    capitulo 4: pre-crisis, crisis energetica, y post-apagon iberico.
    """
    if fecha < pd.Timestamp("2021-07-01", tz="UTC"):
        return "1_precrisis"
    if fecha < pd.Timestamp("2025-01-01", tz="UTC"):
        return "2_crisis"
    return "3_postapagon"


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el conjunto completo de features sobre el dataset maestro.

    Se asume que `df` ya ha pasado por `calidad_datos.corregir_dataframe`
    (columnas `_rel` disponibles) y contiene, como minimo, las columnas
    crudas listadas en el pipeline de adquisicion.

    Devuelve un DataFrame ordenado por fecha, con las `FILAS_CALENTAMIENTO`
    primeras filas eliminadas (lag de 168h aun no disponible) y sin huecos
    (NaN) en `FEATURES_MODELO + [TARGET]`.
    """
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    df["hora"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["mes"] = df["datetime"].dt.month

    # Codificacion ciclica: preferible a one-hot porque usa solo dos
    # columnas por variable y preserva la adyacencia natural entre valores
    # consecutivos (la hora 23 y la hora 0 quedan geometricamente proximas).
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Lags y medias moviles del precio. `.shift(1)` antes del `.rolling(...)`
    # de las medias/desviacion asegura que la ventana de la hora t no
    # incluye el propio precio de la hora t.
    df["precio_lag24"] = df[TARGET].shift(24)
    df["precio_lag48"] = df[TARGET].shift(48)
    df["precio_lag168"] = df[TARGET].shift(168)
    df["precio_ma24"] = df[TARGET].shift(1).rolling(24).mean()
    df["precio_ma168"] = df[TARGET].shift(1).rolling(168).mean()
    df["precio_std24"] = df[TARGET].shift(1).rolling(24).std()

    # Exogenas energeticas anticipadas. Para eolica y solar se usa el lag de
    # 24h de la generacion MEDIDA como proxy de la generacion prevista, en
    # lugar de la prevision que tambien publica ESIOS: decision explicita,
    # documentada como limitacion en el capitulo 8, no un descuido.
    df["eolica_prevista_proxy_rel"] = df["eolica_medida_rel"].shift(24)
    df["solar_prevista_proxy_rel"] = df["solar_medida_rel"].shift(24)

    df["festivo"] = df["festivo"].astype(int)
    df["finde"] = (df["dow"] >= 5).astype(int)
    df["regimen"] = df["datetime"].apply(_regimen)

    df = df.iloc[FILAS_CALENTAMIENTO:].reset_index(drop=True)

    faltantes = df[FEATURES_MODELO + [TARGET]].isna().sum()
    faltantes = faltantes[faltantes > 0]
    if not faltantes.empty:
        raise ValueError(
            "Quedan NaN tras construir_features en columnas: "
            f"{faltantes.to_dict()}. Revisa el merge del dataset maestro."
        )

    return df
