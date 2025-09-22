# Proyecto de Modelado Predictivo

Este repositorio contiene el desarrollo completo de un proyecto de modelado predictivo, desde la exploración y limpieza de datos hasta la validación de modelos y experimentos de regresión. El proyecto está organizado en varias carpetas temáticas, cada una con notebooks, scripts y datos relevantes.

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
> Si necesitas estos archivos, solicítalos al autor o consulta la documentación interna.

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

## Licencia

Este proyecto es solo para fines educativos y de investigación.

---
