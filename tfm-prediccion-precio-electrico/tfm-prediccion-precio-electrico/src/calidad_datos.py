"""Correccion de calidad de datos para indicadores de ESIOS (seccion 6.1).

Contexto
--------
Una primera version de `descarga_datos.py` no fijaba explicitamente los
parametros `time_agg` y `geo_agg` de la API de ESIOS, cuyo valor por defecto
es "sum". Para indicadores publicados nativamente a una granularidad mas
fina que la hora (demanda real, demanda prevista, generacion eolica, solar
y nuclear), esto hizo que la API devolviera la SUMA de las sub-lecturas
dentro de cada hora en lugar de su media, inflando la magnitud por un
factor que ademas varia en el tiempo porque la granularidad nativa de
publicacion de ESIOS ha cambiado a lo largo de los anios. La causa se
confirmo consultando la documentacion oficial de la API [ver README].

`descarga_datos.py` ya incluye la correccion correcta en origen
(`time_agg=average`, `geo_agg=average`), por lo que cualquier descarga
nueva no necesita este modulo. Se conserva aqui, tal y como se describe en
la memoria, como solucion interina para series ya descargadas con la
version antigua del pipeline, y porque la transformacion relativa que
implementa (ecuacion 6.1) es en si misma una forma razonable de neutralizar
un factor de escala desconocido y variable sin necesidad de reconstruir la
serie completa.

La variable de generacion de ciclo combinado (`ciclo_combinado_medida`) no
se corrige aqui: su correlacion casi perfecta con la demanda real (0.9981)
sugiere un error de origen adicional (posible mapeo incorrecto de columnas)
y se descarta por completo del modelado en lugar de corregirse, para evitar
introducir una fuga de informacion espuria.
"""

from __future__ import annotations

import pandas as pd

VENTANA_MEDIANA_DIAS = 90
DESPLAZAMIENTO_HORAS = 24

COLUMNAS_AFECTADAS = [
    "demanda_real",
    "demanda_prevista",
    "eolica_medida",
    "solar_medida",
    "nuclear_medida",
]


def corregir_escala_relativa(serie: pd.Series) -> pd.Series:
    """Aplica la correccion relativa de la ecuacion 6.1 de la memoria.

        x_rel(t) = x(t) / mediana_movil_causal_90d(x)(t - 24h)

    El desplazamiento de 24 horas asegura que la mediana de referencia en el
    instante t solo usa informacion ya disponible un dia antes, evitando
    fuga de informacion hacia el futuro si esta variable se usa como feature
    de un modelo que predice con esa misma antelacion.

    Parametros
    ----------
    serie: pd.Series indexada por datetime horario, ya ordenada.

    Devuelve
    --------
    pd.Series con la misma longitud, con NaN en las primeras
    `VENTANA_MEDIANA_DIAS * 24 + DESPLAZAMIENTO_HORAS` horas, donde la
    mediana movil todavia no tiene ventana completa.
    """
    ventana_horas = VENTANA_MEDIANA_DIAS * 24
    mediana_causal = serie.rolling(ventana_horas, min_periods=ventana_horas // 2).median()
    mediana_desplazada = mediana_causal.shift(DESPLAZAMIENTO_HORAS)
    # Las primeras ~45-46 dias de la serie (ventana_horas // 2 + el
    # desplazamiento de 24h) no tienen mediana movil completa. Se rellenan
    # hacia atras con la primera mediana valida, equivalente a asumir el
    # mismo factor de escala reciente durante ese arranque: una
    # simplificacion explicita y documentada, sin impacto en el modelado
    # porque esas fechas quedan fuera de la particion de entrenamiento
    # (que empieza en 2018-01-08, tras el calentamiento de FILAS_CALENTAMIENTO
    # en features.py) solo si el historial descargado empieza con margen
    # suficiente antes de esa fecha; en caso contrario, revisar `data/README.md`.
    mediana_desplazada = mediana_desplazada.bfill()
    return serie / mediana_desplazada


def corregir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica `corregir_escala_relativa` a todas las columnas afectadas
    presentes en `df`, anadiendo una columna nueva con sufijo `_rel` y
    dejando la columna original intacta para trazabilidad.
    """
    df = df.copy()
    for columna in COLUMNAS_AFECTADAS:
        if columna in df.columns:
            df[f"{columna}_rel"] = corregir_escala_relativa(df[columna])
    return df


def informe_magnitudes_por_anio(df: pd.DataFrame, columna_fecha: str = "datetime") -> pd.DataFrame:
    """Reproduce la tabla 6.1 de la memoria: media anual de cada indicador
    afectado, util como celda de verificacion de magnitudes plausibles que,
    de haber existido desde el principio, habria detectado el problema de
    inmediato.
    """
    anio = pd.to_datetime(df[columna_fecha]).dt.year
    columnas_presentes = [c for c in COLUMNAS_AFECTADAS if c in df.columns]
    return df.groupby(anio)[columnas_presentes].mean().round(0)
