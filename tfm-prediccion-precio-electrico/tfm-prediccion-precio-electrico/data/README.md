# Datos

Este repositorio no incluye los datos crudos ni el dataset maestro: son
datos de acceso publico pero sujetos a los terminos de uso de ESIOS y
AEMET, y el histórico completo (2018-presente, resolución horaria) pesa
varios cientos de MB.

Para reconstruir el dataset maestro:

1. Solicita credenciales gratuitas:
   - ESIOS: escribe a `consultasios@ree.es`.
   - AEMET OpenData: regístrate en <https://opendata.aemet.es/centrodedescargas/inicio>.
2. Copia `.env.example` a `.env` y rellena `ESIOS_TOKEN` y `AEMET_TOKEN`.
3. Ejecuta la descarga:

   ```bash
   export $(grep -v '^#' .env | xargs)
   python -m src.descarga_datos --inicio 2018-01-01
   ```

   Esto crea `data/raw/esios_*.parquet` y `data/raw/aemet_meteo_diaria.parquet`.
4. Une los ficheros crudos en un único dataset horario maestro (join por
   `datetime`, con la meteorología diaria replicada a las 24 horas del día
   correspondiente) y guarda el resultado en
   `data/processed/dataset_maestro.parquet`. Este paso de integración es
   deliberadamente sencillo (un `merge` por fecha) y no se incluye como
   script separado para no atar el repositorio a un esquema de columnas
   crudas que puede cambiar si ESIOS o AEMET actualizan su API.
5. Los precios de gas TTF y derecho de emisión CO2 (EUA) no proceden de
   ESIOS ni AEMET; la memoria (sección 2.2.3) usa series públicas de
   Trading Economics y Ember respectivamente. Añádelas como dos columnas
   más (`gas_ttf`, `co2_eua`) del dataset maestro, con el mismo índice
   `datetime`.

A partir de ahí, `python -m src.pipeline_completo` reproduce el
entrenamiento y la evaluación de los ocho modelos comparados en el
capítulo 7.
