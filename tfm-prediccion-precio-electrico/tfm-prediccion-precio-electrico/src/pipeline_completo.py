"""Orquesta el pipeline completo de principio a fin: carga el dataset
maestro ya descargado, entrena los ocho modelos comparados en el capitulo
7, evalua sobre las tres particiones, calcula el contraste de
Diebold-Mariano y genera las figuras principales.

Requiere haber ejecutado antes `descarga_datos.py` (o disponer ya de
`data/processed/dataset_maestro.parquet`, ver `data/README.md`).

Uso
---
    python -m src.pipeline_completo
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import generar_figuras
from .calidad_datos import corregir_dataframe
from .evaluacion import (
    matriz_diebold_mariano,
    particionar,
    resumen_metricas,
)
from .features import FEATURES_MODELO, TARGET, construir_features
from .modelo_hibrido import calcular_sigma_ref, predecir_hibrido
from .modelos_baseline import predecir_naive_24h, predecir_naive_168h, predecir_naive_media_movil_168h
from .modelos_entrenamiento import entrenar_lightgbm, entrenar_xgboost, entrenar_y_predecir_sarimax

DATASET_MAESTRO = Path("data/processed/dataset_maestro.parquet")
DIR_MODELOS = Path("data/modelos_entrenados")
DIR_FIGURAS = Path("figures")


def cargar_dataset() -> pd.DataFrame:
    if not DATASET_MAESTRO.exists():
        raise FileNotFoundError(
            f"No se encuentra {DATASET_MAESTRO}. Ejecuta primero "
            "`python -m src.descarga_datos` y el notebook de integracion "
            "(ver data/README.md)."
        )
    df = pd.read_parquet(DATASET_MAESTRO)
    df = corregir_dataframe(df)
    return construir_features(df)


def main() -> None:
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)

    print("Cargando y preparando dataset...")
    df = cargar_dataset()
    train, val, test = particionar(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)} filas")

    resultados = {}
    predicciones_test = {}

    print("\nBaselines naive...")
    predicciones_test["Naive 24h"] = predecir_naive_24h(test)
    predicciones_test["Naive 168h"] = predecir_naive_168h(test)
    predicciones_test["Naive MA-168h"] = predecir_naive_media_movil_168h(test)

    print("\nEntrenando XGBoost...")
    modelo_xgb = entrenar_xgboost(train, val)
    predicciones_test["XGBoost"] = modelo_xgb.predict(test[FEATURES_MODELO])
    joblib.dump(modelo_xgb, DIR_MODELOS / "xgboost.joblib")

    print("\nEntrenando LightGBM...")
    modelo_lgb = entrenar_lightgbm(train, val)
    pred_lgb_test = modelo_lgb.predict(test[FEATURES_MODELO])
    predicciones_test["LightGBM"] = pred_lgb_test
    modelo_lgb.booster_.save_model(str(DIR_MODELOS / "lightgbm.txt"))

    print("\nEntrenando y evaluando SARIMAX (walk-forward, puede tardar varios minutos)...")
    predicciones_test["SARIMAX"] = entrenar_y_predecir_sarimax(train, test)

    print("\nModelo hibrido adaptativo...")
    sigma_ref = calcular_sigma_ref(train[TARGET])
    pred_hibrido, lam_test = predecir_hibrido(
        test[TARGET], pred_lgb_test, predicciones_test["Naive 24h"], sigma_ref
    )
    predicciones_test["Hibrido adaptativo"] = pred_hibrido
    print(f"sigma_ref (percentil 90 en train) = {sigma_ref:.2f} €/MWh")

    print("\n--- Metricas de test (tabla 7.1) ---")
    y_test = test[TARGET].values
    for nombre, pred in predicciones_test.items():
        m = resumen_metricas(y_test, pred)
        resultados[nombre] = m
        print(f"  {nombre:22s}  MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  sMAPE={m['sMAPE']:6.2f}%")

    tabla_resultados = pd.DataFrame(resultados).T.round(2)
    tabla_resultados.to_csv(DIR_MODELOS / "metricas_test.csv")

    print("\nCalculando matriz de Diebold-Mariano (figura 7.6)...")
    matriz_dm = matriz_diebold_mariano(predicciones_test, y_test)
    matriz_dm.to_csv(DIR_MODELOS / "diebold_mariano_pvalores.csv")
    generar_figuras.figura_matriz_diebold_mariano(matriz_dm, str(DIR_FIGURAS / "fig_7_6.png"))

    print("\nGenerando figura 7.9 (LightGBM puro vs. híbrido, media diaria en test)...")
    diario = pd.DataFrame({
        "y_true": y_test, "LightGBM": pred_lgb_test, "Hybrid": pred_hibrido,
    }, index=test["datetime"]).resample("D").mean()
    generar_figuras.figura_prediccion_test_hibrido(diario, ruta_salida=str(DIR_FIGURAS / "fig_7_9.png"))

    print("\nListo. Resultados en", DIR_MODELOS, "y figuras en", DIR_FIGURAS)


if __name__ == "__main__":
    main()
