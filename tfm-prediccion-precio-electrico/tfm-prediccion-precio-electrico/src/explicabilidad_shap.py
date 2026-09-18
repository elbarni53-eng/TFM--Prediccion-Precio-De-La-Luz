"""Explicabilidad con valores SHAP (secciones 5.7 y 7.9 de la memoria).

Se aplica sobre los mejores modelos no lineales (LightGBM y, como
contraste de coherencia, XGBoost) para producir tres analisis
complementarios: importancia global de variables, un summary plot que
combina importancia y direccion del efecto, y graficos de cascada
(waterfall) para casos individuales paradigmaticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

N_MUESTRAS_SHAP = 3000  # tamano de muestra de test usado en la memoria
SEMILLA = 42


def calcular_valores_shap(modelo, X: pd.DataFrame, n_muestras: int = N_MUESTRAS_SHAP):
    """Calcula valores SHAP sobre una muestra aleatoria de `X` (por coste
    computacional; ver seccion 7.9). Devuelve el objeto `shap.Explanation`
    completo para poder generar cualquiera de los graficos estandar de la
    libreria (summary, waterfall, bar) sin recalcular.
    """
    muestra = X.sample(n=min(n_muestras, len(X)), random_state=SEMILLA)
    explicador = shap.TreeExplainer(modelo)
    return explicador(muestra)


def importancia_global(valores_shap: shap.Explanation) -> pd.Series:
    """Media del valor absoluto de la contribucion SHAP por variable
    (tabla 7.5): la version numerica del summary plot.
    """
    importancia = np.abs(valores_shap.values).mean(axis=0)
    return pd.Series(importancia, index=valores_shap.feature_names).sort_values(ascending=False)


def comparar_coherencia_entre_modelos(
    importancia_a: pd.Series, importancia_b: pd.Series, top_n: int = 8
) -> float:
    """Cuantifica cuantas variables del top-`top_n` coinciden entre dos
    modelos (seccion 7.9.2, coherencia SHAP comparado): una forma simple y
    directa de contrastar si la importancia de variables es un artefacto
    de un modelo concreto o una senal robusta compartida entre familias.

    Devuelve la fraccion de coincidencia (0 a 1).
    """
    top_a = set(importancia_a.head(top_n).index)
    top_b = set(importancia_b.head(top_n).index)
    return len(top_a & top_b) / top_n


def seleccionar_casos_paradigmaticos(
    df_test: pd.DataFrame, columna_precio: str = "precio_spot"
) -> dict[str, int]:
    """Selecciona tres indices (posiciones en `df_test`) representativos
    para los graficos de cascada de la seccion 7.9.4: una hora de precio
    negativo, un pico vespertino de invierno, y una hora de precio tipico
    (la mas cercana a la mediana de toda la particion).
    """
    precio = df_test[columna_precio]

    idx_negativo = precio.idxmin() if (precio < 0).any() else precio.nsmallest(1).index[0]

    es_invierno = df_test["datetime"].dt.month.isin([12, 1, 2])
    es_vespertino = df_test["datetime"].dt.hour.between(18, 21)
    candidatos_pico = df_test[es_invierno & es_vespertino]
    idx_pico = (
        candidatos_pico[columna_precio].idxmax()
        if not candidatos_pico.empty
        else precio.idxmax()
    )

    mediana = precio.median()
    idx_tipico = (precio - mediana).abs().idxmin()

    return {
        "precio_negativo": df_test.index.get_loc(idx_negativo),
        "pico_vespertino_invierno": df_test.index.get_loc(idx_pico),
        "hora_tipica": df_test.index.get_loc(idx_tipico),
    }
