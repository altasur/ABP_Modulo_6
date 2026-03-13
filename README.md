# Predicción de Gasto en E-commerce - Módulo 6 ABP

# Descripción del Proyecto
Este proyecto desarrolla un sistema de Aprendizaje Supervisado de Regresión para predecir el monto promedio de compra de los usuarios en una plataforma de e-commerce. El objetivo principal es estimar el valor potencial de un cliente para optimizar la asignación de presupuestos de marketing.

# Tecnologías Utilizadas
* Python 3.x
* Pandas & NumPy: Manipulación de datos.
* Scikit-Learn: Construcción de Pipelines, preprocesamiento y modelos de ML.
* Matplotlib & Seaborn: Visualización de datos y análisis de residuos.

# Metodología
1. Preprocesamiento: Implementación de un `ColumnTransformer` para imputación de nulos (mediana), escalado de variables numéricas (`StandardScaler`) y codificación de categóricas (`OneHotEncoder`).
2. Modelado: Comparativa entre un modelo base de Regresión Lineal y un ensamble avanzado de Gradient Boosting.
3. Optimización: Ajuste de hiperparámetros mediante `GridSearchCV`.

# Resultados Finales
Tras las pruebas, la Regresión Lineal demostró ser el modelo más eficiente para este conjunto de datos:

| Métrica | Regresión Lineal | Gradient Boosting |
| :--- | :--- | :--- |
| MAE      | 41.85  | 47.33  |
| RMSE     | 52.31  | 60.38  |
| R² Score | 0.9769 | 0.9692 |

Conclusión: La relación entre el tiempo en el sitio y el gasto es predominantemente lineal, lo que permite obtener una precisión del 97.69% con un modelo de baja complejidad computacional.

# Cómo ejecutar
1. Clona el repositorio.
2. Asegúrate de tener instaladas las librerías: `pip install pandas scikit-learn seaborn matplotlib`.
3. Ejecuta el script: `P6.py`.