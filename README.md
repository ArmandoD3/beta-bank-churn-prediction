# 📉 Predicción de Fuga de Clientes (Customer Churn) - Beta Bank
Modelo de Machine Learning para predecir la fuga de clientes (Churn) en Beta Bank. Enfocado en optimizar la métrica F1 y manejar el desequilibrio de clases.


## 🎯 Descripción del Problema

Beta Bank enfrenta una tasa de abandono de clientes (Churn) que impacta negativamente su rentabilidad, siendo más costoso atraer nuevos clientes que retener a los existentes.

El objetivo de este proyecto es **desarrollar un modelo de Machine Learning** capaz de predecir con precisión qué clientes tienen una alta probabilidad de abandonar el banco en el futuro cercano.

---

## ✅ Solución Propuesta y Logro Principal

Se implementó un pipeline de Machine Learning que incluyó el manejo del desequilibrio de clases mediante el ajuste de pesos (`class_weight='balanced'`) y el *oversampling*. Se exploraron diferentes modelos (Regresión Logística, Árbol de Decisión y **Bosque Aleatorio**).

El modelo final seleccionado es un **Random Forest Classifier** optimizado, el cual superó el requisito de rendimiento clave.

### Requisito vs. Resultado
| Métrica | Objetivo Mínimo | Resultado Alcanzado (Random Forest) |
| :--- | :--- | :--- |
| **F1-Score (Conjunto de Prueba)** | **0.59** | **0.6558** (Superado) |
| **AUC-ROC (Conjunto de Prueba)** | N/A | **0.8703** |

---

## 🥇 Resultados Principales

El mejor rendimiento se obtuvo con el modelo **Random Forest Classifier** ajustado con `n_estimators=100` y `max_depth=10`, utilizando el ajuste de pesos de clase.

### Evaluación en el Conjunto de Prueba
| Métrica | Valor |
| :--- | :--- |
| **F1-Score** | **0.6558** |
| **AUC-ROC** | **0.8703** |
| Precision | 0.6596 |
| Recall | 0.6521 |

> **Interpretación:** El alto **F1-Score** (0.6558) indica un buen equilibrio entre precisión y exhaustividad, crucial para problemas de clasificación con desequilibrio. Un valor **AUC-ROC** de 0.8703 confirma que el modelo tiene una excelente capacidad de discriminación, siendo 0.5 un clasificador aleatorio y 1.0 un clasificador perfecto.

---

## 🛠️ Metodología

El proyecto siguió un enfoque estructurado en tres fases:

### 1. Preparación de Datos
* **Imputación:** Se rellenaron los 909 valores ausentes en la columna `Tenure` con la **mediana** (5.0) para mantener la robustez ante valores atípicos.
* **Codificación:** Las variables categóricas (`Geography` y `Gender`) se convirtieron a formato numérico utilizando **One-Hot Encoding** (`pd.get_dummies`). Se eliminaron las columnas redundantes (`RowNumber`, `CustomerId`, `Surname`).
* **División:** Los datos se dividieron en conjuntos de **Entrenamiento (60%)**, **Validación (20%)** y **Prueba (20%)** utilizando `stratify` para preservar la proporción de clases en cada subconjunto.
* **Escalado:** Las características numéricas (`CreditScore`, `Age`, `Balance`, etc.) fueron **estandarizadas** (`StandardScaler`) basándose únicamente en el conjunto de entrenamiento.

### 2. Manejo de Desequilibrio de Clases
La clase objetivo (`Exited`) mostró un **desequilibrio significativo** (aproximadamente **80%** de clientes retenidos (0) vs. **20%** de clientes que abandonaron (1)).

Se compararon dos enfoques principales:
1.  **Ajuste de Hiperparámetros (`class_weight='balanced'`):** Se utiliza para penalizar errores en la clase minoritaria durante el entrenamiento.
2.  **Sobremuestreo (*Upsampling*):** Se aumentó la clase minoritaria en el conjunto de entrenamiento (factor de repetición de 4) para crear un conjunto de entrenamiento artificialmente balanceado.

### 3. Entrenamiento y Optimización de Modelos
Se entrenaron modelos de Regresión Logística, Árbol de Decisión y Bosque Aleatorio:

| Modelo | Técnica de Balanceo | Mejor F1-Score (Validación) | AUC-ROC (Validación) |
| :--- | :--- | :--- | :--- |
| **Regresión Logística** | Sin Balanceo | 0.3077 | 0.7937 |
| **Árbol de Decisión** | `class_weight='balanced'` | 0.5759 (max\_depth=6) | 0.8233 |
| **Random Forest** | `class_weight='balanced'` | **0.6490** (n\_est=100, max\_depth=10) | **0.8711** |

El modelo **Random Forest** con ajuste de pesos de clase demostró ser el más robusto y eficaz, siendo seleccionado para la prueba final.

---

## 💻 Tecnologías Utilizadas

* **Lenguaje:** Python
* **Librerías Principales:**
    * **Pandas & NumPy:** Manipulación y cálculo de datos.
    * **Scikit-learn:** Modelado, preprocesamiento y evaluación (Modelos: `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`; Preprocesamiento: `StandardScaler`, `train_test_split`).
    * **Matplotlib & Seaborn:** Visualización de la distribución de clases.
* **Entorno:** Jupyter Notebook (`.ipynb`).

---

## 📂 Estructura del Proyecto
/
├── datasets/
│   └── Churn.csv                  # Conjunto de datos original.
├── notebooks/
│   └── proyecto_churn_final.ipynb # Notebook de Jupyter con todo el proceso de análisis, preprocesamiento, modelado y prueba final.
├── README.md                      # Documentación principal del proyecto.



## 🏃‍♀️ Cómo Ejecutar el Proyecto

Para replicar el análisis y los resultados, sigue los siguientes pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/tu_usuario/beta-bank-churn-prediction.git](https://github.com/tu_usuario/beta-bank-churn-prediction.git)
    cd beta-bank-churn-prediction
    ```
2.  **Instalar Dependencias:**
    Asegúrate de tener un entorno Python configurado e instalar las librerías necesarias:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Ejecutar el Notebook:**
    Abre el archivo `notebooks/proyecto_churn_final.ipynb` en tu entorno Jupyter (o VS Code) y ejecuta las celdas secuencialmente para replicar el preprocesamiento, entrenamiento y prueba final del modelo.

---

## 💡 Conclusiones y Aprendizajes

1.  **Impacto del Desequilibrio:** La métrica **F1-Score** pasó de un valor base muy bajo (0.3077) a uno competitivo (0.6490) tan pronto como se aplicó el manejo del desequilibrio de clases, demostrando la necesidad crítica de abordar este problema.
2.  **Elección del Algoritmo:** El cambio a **Random Forest** fue decisivo, proporcionando una mejora significativa sobre el Árbol de Decisión y la Regresión Logística, lo que subraya la importancia de probar múltiples arquitecturas de modelos.
3.  **Compensación de Complejidad:** El Random Forest es más lento de entrenar, pero su superioridad en métricas de rendimiento (F1 y AUC-ROC) justificó su elección, especialmente en un conjunto de datos de este tamaño.
4.  **Ajuste de Umbral:** Si bien el ajuste de umbral en el Árbol de Decisión mejoró el F1-Score en validación (a 0.6105), el rendimiento en el conjunto de prueba fue inferior al del Random Forest sin ajuste de umbral, lo que resalta el riesgo de **sobreajuste** al optimizar demasiado un modelo más simple.
