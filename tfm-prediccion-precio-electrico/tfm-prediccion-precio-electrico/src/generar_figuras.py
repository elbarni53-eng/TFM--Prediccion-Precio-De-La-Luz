"""Generacion de las figuras del capitulo 7.

Paleta de color: las primeras versiones de la figura 7.9 usaban dos tonos
de verde casi identicos para "LightGBM puro" y "hibrido adaptativo", lo que
las hacia dificiles de distinguir a simple vista. Este modulo usa una
paleta de tres colores con buen contraste entre si (negro / naranja /
azul), consistente en todas las figuras que comparan varias series de
prediccion frente al precio real.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_PRECIO_REAL = "#000000"
COLOR_LIGHTGBM = "#e67e22"     # naranja
COLOR_HIBRIDO = "#1a5276"      # azul
COLOR_NAIVE = "#7f8c8d"
Z_95 = 1.96


def figura_prediccion_test_hibrido(
    daily: pd.DataFrame,
    columna_real: str = "y_true",
    columna_lightgbm: str = "LightGBM",
    columna_hibrido: str = "Hybrid",
    ruta_salida: str = "figures/fig_7_9.png",
) -> None:
    """Figura 7.9: precio real, LightGBM puro e hibrido adaptativo,
    agregados a media diaria sobre el conjunto de test, con la banda de
    confianza del 95% del hibrido.

    `daily` debe estar indexado por fecha y contener las tres columnas
    (ya agregadas a media diaria con `.resample("D").mean()`).
    """
    residuo = daily[columna_real] - daily[columna_hibrido]
    rmse_tramo = np.sqrt((residuo ** 2).mean())

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily[columna_real], color=COLOR_PRECIO_REAL, lw=1.0,
            label="Precio real (media diaria)")
    ax.plot(daily.index, daily[columna_lightgbm], color=COLOR_LIGHTGBM, lw=1.0, ls="--",
            label="LightGBM puro")
    ax.plot(daily.index, daily[columna_hibrido], color=COLOR_HIBRIDO, lw=1.3,
            label="Híbrido adaptativo")
    ax.fill_between(
        daily.index,
        daily[columna_hibrido] - Z_95 * rmse_tramo,
        daily[columna_hibrido] + Z_95 * rmse_tramo,
        color=COLOR_HIBRIDO, alpha=0.13,
        label=f"Intervalo 95% del híbrido (±{Z_95 * rmse_tramo:.1f} €/MWh)",
    )
    ax.set_ylabel("Precio medio diario (€/MWh)")
    ax.set_title(
        "Predicción agregada a media diaria en test (2025-2026, régimen "
        "post-apagón): LightGBM puro frente a híbrido"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def figura_lambda_dinamico(
    test: pd.DataFrame, lam: np.ndarray, precio_real: pd.Series, pred_hibrido: np.ndarray,
    ruta_salida: str = "figures/fig_7_3.png",
) -> None:
    """Figura 7.3: panel superior con el peso dinamico lambda(t), panel
    inferior con precio real frente a prediccion del hibrido.
    """
    fig, (ax_lambda, ax_precio) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 2]
    )
    ax_lambda.plot(test.index, lam, color=COLOR_HIBRIDO, lw=0.9)
    ax_lambda.set_ylabel(r"$\lambda(t)$")
    ax_lambda.set_ylim(-0.05, 1.05)
    ax_lambda.axhline(0.5, color="grey", lw=0.6, ls=":")

    ax_precio.plot(test.index, precio_real, color=COLOR_PRECIO_REAL, lw=0.8, label="Precio real")
    ax_precio.plot(test.index, pred_hibrido, color=COLOR_HIBRIDO, lw=0.8, label="Híbrido adaptativo")
    ax_precio.set_ylabel("Precio (€/MWh)")
    ax_precio.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        r"Peso dinámico $\lambda(t)$ y predicción del híbrido, conjunto de test completo"
    )
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def figura_matriz_diebold_mariano(
    matriz_p_valores: pd.DataFrame, ruta_salida: str = "figures/fig_7_6.png",
) -> None:
    """Figura 7.6: mapa de calor de p-valores del contraste de
    Diebold-Mariano entre todos los pares de modelos.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    datos = matriz_p_valores.astype(float)
    im = ax.imshow(datos, cmap="viridis_r", vmin=0, vmax=0.10)
    ax.set_xticks(range(len(datos.columns)))
    ax.set_xticklabels(datos.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(datos.index)))
    ax.set_yticklabels(datos.index)
    for i in range(len(datos.index)):
        for j in range(len(datos.columns)):
            valor = datos.iloc[i, j]
            if not np.isnan(valor):
                ax.text(j, i, f"{valor:.3f}", ha="center", va="center",
                        color="white" if valor < 0.05 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="p-valor")
    ax.set_title("Matriz de p-valores, contraste de Diebold-Mariano (test)")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
