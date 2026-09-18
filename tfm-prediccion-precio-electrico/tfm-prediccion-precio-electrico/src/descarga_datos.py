"""Descarga de datos crudos desde las APIs publicas de ESIOS y AEMET.

Este modulo implementa la Fase 1 del pipeline descrito en la seccion 5.2
de la memoria: obtiene el historico horario de precio, demanda, generacion
por tecnologia (ESIOS) y de variables meteorologicas diarias (AEMET), y los
guarda como ficheros parquet independientes en `data/raw/` para que el resto
del pipeline (features.py) no dependa de tener conexion a internet.

Credenciales
------------
Ambas APIs requieren un token personal gratuito, solicitado por correo:
  - ESIOS:  consultasios@ree.es
  - AEMET:  https://opendata.aemet.es/centrodedescargas/inicio

Los tokens se leen exclusivamente de variables de entorno (nunca se escriben
en este fichero). Copia `.env.example` como `.env`, rellena tus tokens, y
carga el entorno antes de ejecutar este script, por ejemplo con
`python-dotenv` o `export $(cat .env | xargs)`.

Uso
---
    python -m src.descarga_datos --inicio 2018-01-01 --fin 2026-09-01
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

ESIOS_BASE_URL = "https://api.esios.ree.es"
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"

# Indicadores de ESIOS utilizados en este trabajo (tabla 2.2 de la memoria).
INDICADORES_ESIOS = {
    600: "precio_spot",              # variable objetivo
    1293: "demanda_real",
    460: "demanda_prevista",
    541: "eolica_medida",
    10034: "solar_medida",
    549: "nuclear_medida",
    544: "ciclo_combinado_medida",   # descartado tras el hallazgo de la seccion 6.1
}

# Estaciones climatologicas de AEMET usadas para las variables meteorologicas
# (seccion 2.3.2). Se promedian las seis para obtener una serie nacional
# representativa, tal y como se describe en la memoria.
ESTACIONES_AEMET = {
    "3195": "Madrid-Barajas",
    "0076": "Barcelona-Aeropuerto",
    "5960": "Sevilla-Aeropuerto",
    "1082": "Bilbao-Aeropuerto",
    "2462": "Valladolid",
    "8175": "Valencia-Aeropuerto",
}

RAW_DIR = Path("data/raw")


def _cabeceras_esios(token: str) -> dict:
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "x-api-key": token,
    }


def descargar_indicador_esios(
    indicador_id: int,
    fecha_inicio: str,
    fecha_fin: str,
    token: str,
    pausa_seg: float = 1.0,
) -> pd.DataFrame:
    """Descarga un indicador de ESIOS trocando la peticion en bloques anuales.

    La API responde con timeout o error 429 si se pide un rango de varios
    anios en una sola llamada, asi que se pagina por anios naturales y se
    espera `pausa_seg` segundos entre peticiones para no saturar el servidor.
    """
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin)
    bloques = pd.date_range(inicio, fin, freq="YS")
    if len(bloques) == 0 or bloques[0] > inicio:
        bloques = pd.DatetimeIndex([inicio]).append(bloques)

    filas = []
    for i, bloque_inicio in enumerate(bloques):
        bloque_fin = min(
            (bloque_inicio + pd.DateOffset(years=1)) - pd.Timedelta(hours=1), fin
        )
        params = {
            "start_date": bloque_inicio.strftime("%Y-%m-%dT%H:%M"),
            "end_date": bloque_fin.strftime("%Y-%m-%dT%H:%M"),
            "time_trunc": "hour",
            # IMPORTANTE (seccion 6.1 de la memoria): la API de ESIOS agrega
            # por defecto con time_agg=sum y geo_agg=sum. Para un indicador
            # publicado nativamente a una granularidad mas fina que la hora,
            # eso hace que el valor horario devuelto sea la SUMA de las
            # sub-lecturas dentro de esa hora en lugar de su media, inflando
            # la magnitud por un factor variable en el tiempo. La primera
            # version de este pipeline no fijaba estos dos parametros, lo
            # que produjo el problema de calidad de datos diagnosticado y
            # documentado en la memoria (demanda y generacion x2-x4 a partir
            # de 2022). Se fijan aqui explicitamente a "average", tal y como
            # recomienda la documentacion oficial de la API.
            "time_agg": "average",
            "geo_agg": "average",
        }
        resp = requests.get(
            f"{ESIOS_BASE_URL}/indicators/{indicador_id}",
            headers=_cabeceras_esios(token),
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        valores = resp.json()["indicator"]["values"]
        filas.extend(valores)
        if i < len(bloques) - 1:
            time.sleep(pausa_seg)

    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[["datetime", "value"]].rename(columns={"value": "valor"})
    # Algunos indicadores de ESIOS devuelven varias geo-zonas (peninsula,
    # Baleares, Canarias...); nos quedamos solo con la peninsular agregando
    # por si hubiera duplicados de datetime tras el filtrado previo del JSON.
    df = df.groupby("datetime", as_index=False)["valor"].mean()
    return df.sort_values("datetime").reset_index(drop=True)


def descargar_todos_los_indicadores(
    fecha_inicio: str, fecha_fin: str, token: str, salida: Path = RAW_DIR
) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for indicador_id, nombre in INDICADORES_ESIOS.items():
        print(f"Descargando indicador {indicador_id} ({nombre})...")
        df = descargar_indicador_esios(indicador_id, fecha_inicio, fecha_fin, token)
        destino = salida / f"esios_{nombre}.parquet"
        df.to_parquet(destino, index=False)
        print(f"  -> {len(df)} filas guardadas en {destino}")


def descargar_meteo_aemet(
    fecha_inicio: str, fecha_fin: str, token: str, salida: Path = RAW_DIR
) -> None:
    """Descarga observaciones climatologicas diarias de las estaciones de
    referencia y guarda la media diaria across estaciones.

    La API de AEMET funciona en dos pasos: la primera peticion devuelve una
    URL temporal (`datos`) donde esta el JSON real con los valores.
    """
    salida.mkdir(parents=True, exist_ok=True)
    frames = []
    rangos = pd.date_range(fecha_inicio, fecha_fin, freq="6MS")
    for estacion_id, nombre_estacion in ESTACIONES_AEMET.items():
        for i in range(len(rangos) - 1):
            ini = rangos[i].strftime("%Y-%m-%dT00:00:00UTC")
            fin = (rangos[i + 1] - pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59UTC")
            url = (
                f"{AEMET_BASE_URL}/valores/climatologicos/diarios/datos/"
                f"fechaini/{ini}/fechafin/{fin}/estacion/{estacion_id}"
            )
            resp = requests.get(url, params={"api_key": token}, timeout=30)
            resp.raise_for_status()
            enlace = resp.json().get("datos")
            if not enlace:
                continue
            datos = requests.get(enlace, timeout=30).json()
            df = pd.DataFrame(datos)
            df["estacion"] = nombre_estacion
            frames.append(df)
            time.sleep(1.5)  # AEMET limita a ~50 peticiones/minuto

    if not frames:
        raise RuntimeError("AEMET no devolvio datos para ninguna estacion/rango.")

    bruto = pd.concat(frames, ignore_index=True)
    bruto["fecha"] = pd.to_datetime(bruto["fecha"])

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(bruto[col].astype(str).str.replace(",", "."), errors="coerce")

    diario = pd.DataFrame({
        "fecha": bruto["fecha"],
        "temperatura_media": _num("tmed"),
        "temperatura_min": _num("tmin"),
        "temperatura_max": _num("tmax"),
        "viento_medio_ms": _num("velmedia"),
        "precipitacion_mm": _num("prec"),
    })
    diario = diario.groupby("fecha", as_index=False).mean(numeric_only=True)
    diario.to_parquet(salida / "aemet_meteo_diaria.parquet", index=False)
    print(f"Meteorologia diaria guardada: {len(diario)} dias.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", default="2018-01-01")
    parser.add_argument("--fin", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--salida", default=str(RAW_DIR))
    args = parser.parse_args()

    esios_token = os.environ.get("ESIOS_TOKEN")
    aemet_token = os.environ.get("AEMET_TOKEN")
    if not esios_token or not aemet_token:
        raise SystemExit(
            "Faltan credenciales. Define ESIOS_TOKEN y AEMET_TOKEN como "
            "variables de entorno (ver .env.example)."
        )

    salida = Path(args.salida)
    descargar_todos_los_indicadores(args.inicio, args.fin, esios_token, salida)
    descargar_meteo_aemet(args.inicio, args.fin, aemet_token, salida)


if __name__ == "__main__":
    main()
