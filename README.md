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

## Estructura del repositorio

- `codigo/`: generador y script unificado de validación y sensibilidad.
- `datos/`: CSV sintético y metadatos de generación.
- `resultados/`: archivos de referencia conservados de los experimentos originales.
- `resultados_reproducidos/`: nuevas salidas, sin sobrescribir las referencias.

## Instalación

Python **3.13.5** es la versión de referencia del entorno comprobado. Las dependencias están fijadas en `requirements.txt`. La instalación desde un entorno vacío todavía no se ha verificado.

```bash
python --version
python -m venv .venv
```

Activación en Windows (PowerShell): ` .\.venv\Scripts\Activate.ps1 `.

Activación en Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
```

## Ejecución desde la raíz

```bash
python codigo/generador_tfm_definitivo.py --seed 20261117 --output datos/dataset_tfm_definitivo.csv
python codigo/validacion_definitiva_tfm.py --input datos/dataset_tfm_definitivo.csv --output-dir resultados_reproducidos
```

No es necesario editar rutas en los scripts. El script unificado conserva las particiones, preprocesamiento y rejillas originales, utilizando la rejilla ampliada de Isolation Forest (100, 150, 250 y 400 árboles). Selecciona hiperparámetros mediante PR AUC y etiquetas de validación interna; por tanto, el ajuste no es estrictamente no supervisado. Las etiquetas también permiten seleccionar registros normales en ese escenario. Las etiquetas externas se reservan para evaluar los modelos.

El script genera:

| Archivo | Filas esperadas |
|---|---:|
| `resultados_validacion_definitiva.csv` | 60 (5 semillas × 2 escenarios × 2 contextos × 3 modelos) |
| `resultados_tipologias_definitiva.csv` | 360 (60 × 6 tipologías, P95) |
| `resultados_sensibilidad_p90_p95_p97.csv` | 180 (60 × 3 percentiles) |

Los tres percentiles se calculan sobre las puntuaciones del mismo ajuste final, sin reentrenar ni optimizar sobre la prueba externa. P95 es el criterio principal predefinido, no necesariamente el mejor F1 en todos los escenarios.

## Resultados de referencia

PR AUC media de cinco particiones, procedente de `resultados/resultados_validacion_tfm_if_ampliado.csv`:

| Modelo | Solo normales | Contaminado | Solo normales con contexto | Contaminado con contexto |
|---|---:|---:|---:|---:|
| Isolation Forest | 0,144 | 0,468 | 0,139 | 0,554 |
| LOF | 0,733 | 0,689 | 0,853 | 0,818 |
| One-Class SVM | 0,722 | 0,686 | 0,841 | 0,796 |

**Estado de verificación:** el generador ha reproducido exactamente el CSV. El script unificado es una adaptación nueva, comprobada sintácticamente, pendiente de ejecución integral. Los CSV originales se conservan como referencia, no como resultados de esa nueva ejecución. El CSV `resultados_validacion_umbral95.csv` utiliza una rejilla reducida de Isolation Forest; no debe confundirse con el integrado de rejilla ampliada. Los recuentos de tipologías de referencia de IF ampliado están en `resultados_tipologias_if_rejilla_ampliada.csv`. El CSV histórico de sensibilidad requiere reconciliación con las nuevas salidas; no se garantiza igualdad de sus métricas P95.

## Integridad y comprobación

```bash
python verificar_integridad.py
```

Este comando comprueba SHA-256 de los archivos de referencia y, si existen las nuevas salidas, sus recuentos, claves y coherencia P95, además de comparar las métricas principales con la referencia ampliada (tolerancia absoluta 1e-8, excluyendo tiempos). Las discrepancias se notifican; no deben ocultarse ni sustituirse resultados de la memoria sin revisión.

SHA-256 esperado del dataset: `6dc52c24264411574d0dac265dee839dd55c10b8418b4ff21cd7398c4ac057aa`.

## Interpretación y limitaciones

LOF y One-Class SVM presentan resultados próximos en la comparación por PR AUC. La mejora asociada a la frecuencia especialidad–ATC se concentra principalmente en una tipología relacionada con su regla de generación sintética. El rendimiento sobre datos sintéticos no demuestra validez clínica ni garantiza generalización a datos reales.

## Privacidad y uso de IA

Los datos son sintéticos. No deben incorporarse datos personales, credenciales ni información corporativa confidencial. Se utilizó ChatGPT (OpenAI) como apoyo a la generación de datos, el desarrollo y revisión del código y la elaboración del trabajo, bajo supervisión del autor.

## Referencia

Trabajo Fin de Máster: *Detección de anomalías en receta electrónica mediante Machine Learning*, UNIR, 2026. Para citar el código, utilizar la URL y el identificador de la versión definitiva publicada en GitHub.

- URL definitiva: pendiente de publicación.
- Versión propuesta: `v1.0-TFM` (pendiente de crear y verificar).
