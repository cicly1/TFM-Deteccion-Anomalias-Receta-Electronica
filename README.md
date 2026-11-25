# Detección de anomalías en receta electrónica mediante Machine Learning

Trabajo Fin de Máster en Inteligencia Artificial — UNIR.

## Objetivo

Comparar Isolation Forest, Local Outlier Factor (LOF) y One-Class SVM para priorizar registros de prescripción y dispensación que puedan requerir revisión por profesionales sanitarios. El sistema no constituye una herramienta de diagnóstico ni de decisión clínica automática.

## Datos y metodología

- 2.640 registros completamente sintéticos: 2.400 normales y 240 anomalías (9,09 %).
- Seis tipologías: dosis extrema, frecuencia alta, duración excesiva, duplicidad temporal, especialidad atípica y dispensación atípica.
- Generación reproducible mediante Python y semilla `20261117`.
- Cinco particiones externas agrupadas por paciente (70/30 aproximadamente por grupos), con semillas `11`, `22`, `33`, `44` y `55` y validación interna agrupada para seleccionar hiperparámetros.
- Dos escenarios: entrenamiento solo con registros normales y entrenamiento contaminado; ambos con y sin frecuencia contextual especialidad–ATC.
- Preprocesamiento mediante OneHotEncoder y StandardScaler, ajustados sobre los datos de entrenamiento. Las etiquetas no se emplean como predictores.
- Métrica principal: PR AUC (average precision); métricas complementarias: ROC AUC, precisión, recall, F1, precision@50, recall@50, R-precision y tiempos.
- Umbral principal P95 de las puntuaciones de entrenamiento, sin utilizar la prevalencia real de anomalías de la prueba externa. P90 y P97 se consideran en el análisis de sensibilidad.

## Estructura propuesta

```text
README.md
requirements.txt
codigo/
  generador_tfm_definitivo.py
  validacion_umbral95_tfm.py
  validacion_if_completa.py
datos/
  dataset_tfm_definitivo.csv
resultados/
  resultados_validacion_umbral95.csv
  resultados_tipologias_umbral95.csv
  resultados_validacion_tfm_if_ampliado.csv
  resultados_sensibilidad_p90_p95_p97.csv
```

## Instalación

Desde la raíz del repositorio, con Python 3.11 o 3.12 (versiones propuestas por compatibilidad; comprobar la versión del entorno experimental original):

```bash
python -m venv .venv
```

Activación en Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Activación en Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Ejecutar desde la raíz del repositorio, con las carpetas `datos` y `resultados` creadas.

### 1. Generación del conjunto de datos

```bash
python codigo/generador_tfm_definitivo.py --seed 20261117 --output datos/dataset_tfm_definitivo.csv
```

El generador también produce `datos/dataset_tfm_definitivo.metadata.json`, con la semilla, recuentos, columnas y SHA-256 del CSV. El SHA-256 del CSV de referencia disponible al preparar este README es:

```text
6dc52c24264411574d0dac265dee839dd55c10b8418b4ff21cd7398c4ac057aa
```

### 2. Evaluación principal P95

**Adaptación necesaria:** los scripts originales escriben resultados en `/mnt/data/`. Antes de ejecutarlos localmente, sustituir las rutas de salida `/mnt/data/resultados_...csv` por `resultados/resultados_...csv`. El argumento `--input` permite especificar el CSV de entrada sin modificar su ruta predeterminada.

```bash
python codigo/validacion_umbral95_tfm.py --input datos/dataset_tfm_definitivo.csv
```

Se generan `resultados_validacion_umbral95.csv` y `resultados_tipologias_umbral95.csv`. Conservar una copia de los resultados originales antes de ejecutar el script.

### 3. Evaluación ampliada de Isolation Forest

```bash
python codigo/validacion_if_completa.py --input datos/dataset_tfm_definitivo.csv
```

Este script genera los CSV de la rejilla ampliada de Isolation Forest. No sustituye por sí solo la evaluación principal P95: comprobar que la rejilla y el criterio de umbral coinciden con la configuración definitiva de la memoria antes de integrar resultados.

### 4. Comprobación de resultados

Contrastar los CSV obtenidos con los archivos de referencia. Los tiempos pueden variar entre equipos. El CSV de sensibilidad P90/P95/P97 y el CSV integrado de la rejilla ampliada se incluyen como resultados de referencia; los comandos anteriores no regeneran por sí solos todos esos archivos. Para una reproducción integral, publicar también el procedimiento o script exacto de integración y sensibilidad.

## Interpretación y limitaciones

LOF y One-Class SVM presentan resultados próximos en la comparación por PR AUC. La mejora asociada a la frecuencia especialidad–ATC se concentra principalmente en una tipología relacionada con su regla de generación sintética. El rendimiento sobre datos sintéticos no demuestra validez clínica ni garantiza generalización a datos reales.

## Privacidad y uso de IA

Los datos son sintéticos. No deben incorporarse datos personales, credenciales ni información corporativa confidencial. Se utilizó ChatGPT (OpenAI) como apoyo a la generación de datos, el desarrollo y revisión del código y la elaboración del trabajo, bajo supervisión del autor.

## Referencia

Trabajo Fin de Máster: *Detección de anomalías en receta electrónica mediante Machine Learning*, UNIR, 2026. Para citar el código, utilizar la URL y el identificador de la versión definitiva publicada en GitHub.
