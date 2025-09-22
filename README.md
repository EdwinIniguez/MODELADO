
# MODELADO - Proyecto de Modelado Predictivo

Este repositorio forma parte del **Portafolio de Análisis** y documenta el desarrollo completo de un proyecto de modelado predictivo, desde la exploración y limpieza de datos hasta la validación de modelos y experimentos de regresión. Aquí se presentan evidencias alineadas con los indicadores de la rúbrica, con referencias cruzadas para facilitar la evaluación.

---

## Evidencias para el Portafolio

### Construcción de Modelos (SMA0101A)
- **Construcción manual de modelos:**
  - `1/Exploracion_Datos.ipynb`, `1/UniendoDatos.ipynb`, `1/Modelo_Binario.ipynb`
  - Selección y justificación de variables en análisis exploratorio
- **Explicación de variables:**
  - `1/Diccionario_Simple_VisualizacionAonline.xlsx` y notebooks asociados
- **Validación y funcionamiento del modelo:**
  - `3/actividad_curvas_aprendizaje_validacion.ipynb`, `3/ejemplos_tema2_regresion_lineal.ipynb`
  - `Problema de regresion/problema_de_regresion.ipynb` (validación de supuestos)

### Relación con el Portafolio
Este repositorio es referenciado en el [Portafolio de Análisis](../TC3006C.101-Portafolio-Analisis/README.md) como evidencia principal para los indicadores de construcción manual, explicación de variables y validación de supuestos.

---

---

## Estructura del repositorio

- **1/**  
  Exploración, limpieza y unión de datos.  
  - `Exploracion_Datos.ipynb`: Análisis exploratorio de los datos originales.
  - `UniendoDatos.ipynb`: Integración de diferentes fuentes de datos.
  - `Modelo_Binario.ipynb`: Modelado de clasificación binaria.
  - `Funciones_Cuota.py`: Funciones auxiliares para el procesamiento de cuotas.
  - `Diccionario_Simple_VisualizacionAonline.xlsx`: Diccionario de datos simplificado.
  - **BD/**: Archivos de datos originales en formato `.txt` (no incluidos en el repo por su tamaño).
  - **Resultado/**: Resultados y objetos serializados del modelado.

- **2/**  
  Modelado avanzado y experimentos con CatBoost.
  - `Sesion_2_model.ipynb`: Modelos avanzados y experimentos.
  - `data_clean.csv`: Dataset limpio para modelado.
  - **catboost_info/**: Archivos de logs y resultados de CatBoost.

- **3/**  
  Validación y curvas de aprendizaje.
  - `actividad_curvas_aprendizaje_validacion.ipynb`/`.html`: Ejercicios de validación y aprendizaje.
  - `ejemplos_tema2_regresion_lineal.ipynb`: Ejemplos de regresión lineal.

- **Problema de regresion/**  
  Experimentos de regresión con el dataset de Parkinson.
  - `problema_de_regresion.ipynb`: Notebook principal del problema de regresión.
  - **parkinsons+telemonitoring/**: Datos y descripciones del dataset de Parkinson.

---

## Datasets

> **Nota:**  
> Los archivos de datos grandes (`VisualizacionAonline.txt`, `LlamadasyWhatsap.txt`) **no están incluidos** en el repositorio por restricciones de tamaño de GitHub.  
> Si necesitas estos archivos, es necesario extraerlos desde el archivo compreso `BD`.

---

## Tecnologías utilizadas

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- CatBoost
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Cómo usar este repositorio

1. Clona el repositorio:
   ```bash
   git clone https://github.com/EdwinIniguez/MODELADO.git
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abre los notebooks en Jupyter o VS Code para explorar los análisis y modelos.

---

## Notas para Evaluación
- Este README y los notebooks están organizados para facilitar la localización de evidencias por indicador.
- La documentación y los análisis están pensados para ser fácilmente referenciables desde el portafolio.

---
## Licencia

Este proyecto es solo para fines educativos y de investigación.

---
