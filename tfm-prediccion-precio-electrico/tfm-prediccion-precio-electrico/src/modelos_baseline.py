"""Modelos baseline naive (seccion 6.2.1). Sin parametros que ajustar: se
limitan a copiar un valor ya observado del propio precio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predecir_naive_24h(df: pd.DataFrame) -> np.ndarray:
    """Repite el precio observado 24 horas antes (estacionalidad diaria)."""
    return df["precio_lag24"].values


def predecir_naive_168h(df: pd.DataFrame) -> np.ndarray:
    """Repite el precio observado 168 horas antes (estacionalidad semanal)."""
    return df["precio_lag168"].values


def predecir_naive_media_movil_168h(df: pd.DataFrame) -> np.ndarray:
    """Usa la media movil de las 168 horas previas como prediccion."""
    return df["precio_ma168"].values
