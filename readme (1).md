# Clasificación Binaria con Regresión Logística y Descenso de Gradiente desde Cero

Implementación transparente de **Regresión Logística binaria con *Batch Gradient Descent* y regularización L2** programada desde cero en **Python + NumPy**, aplicada al dataset real **Breast Cancer Wisconsin (Diagnostic)**.

El proyecto está diseñado para **exponer la mecánica interna del entrenamiento** (suma lineal, función sigmoide, *Binary Cross-Entropy / Log-Loss*, magnitud del gradiente y evolución de la frontera de decisión) y **justificar empíricamente** cada decisión de preprocesamiento y selección de hiperparámetros, validando los resultados frente a `scikit-learn`.

## Dataset y preprocesamiento

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) (`sklearn.datasets.load_breast_cancer`)
- **Tamaño:** 569 muestras, 30 características numéricas continuas, 0 valores faltantes
- **Clases:** `0` = Maligno (212 casos) | `1` = Benigno (357 casos)
- **Partición estratificada (60 / 20 / 20):**
  - Entrenamiento (`train`): 341 muestras (60%)
  - Validación (`val`): 114 muestras (20%) — utilizada exclusivamente para elegir $\eta$ y $\lambda$
  - Prueba (`test`): 114 muestras (20%) — reservada para la evaluación final
- **Escalado sin fuga de datos (*data leakage*):** `StandardScaler` ajustado únicamente con `X_train` (`fit_transform`) y aplicado a `X_val` y `X_test` (`transform`), verificando media `0.0000` y desviación estándar `1.0000` en entrenamiento.

## Mecánica interna implementada

En cada época del ciclo de entrenamiento sobre $m$ muestras:

1. **Combinación lineal:** $z = Xw + b$
2. **Función Sigmoide:** $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$ (con recorte numérico en $[-500, 500]$)
3. **Función de pérdida (*Binary Cross-Entropy* + regularización L2):**
   $$J(w, b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \ln(\hat{p}_i) + (1 - y_i)\ln(1 - \hat{p}_i)\Big] + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2$$
4. **Gradientes promedio y actualización:**
   $$\frac{\partial J}{\partial w} = \frac{1}{m}X^T(\hat{p} - y) + \frac{\lambda}{m}w, \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{p}_i - y_i)$$
   $$w \leftarrow w - \eta \frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial J}{\partial b}$$

## Resultados empíricos de hiperparámetros

### Experimento 1: Tasa de aprendizaje ($\eta$) — 300 épocas, $\lambda = 0.0$

Malla en escala logarítmica para evaluar convergencia lenta, óptima y pasos agresivos:

| Tasa de aprendizaje ($\eta$) | Train Log-Loss | Val Accuracy | Observación |
| :---: | :---: | :---: | :--- |
| `0.005` | 0.2053 | 95.61% | Convergencia demasiado lenta en 300 épocas |
| `0.05` | 0.0860 | 97.37% | Buena convergencia, aún margen de mejora |
| **`0.5`** | **0.0538** | **98.25%** | **Óptimo elegido (`Val Loss = 0.0591`)** |
| `5.0` | 0.0402 | 97.37% | Pasos agresivos; empeora generalización en validación |

### Experimento 2: Regularización L2 ($\lambda$) — 300 épocas, $\eta = 0.5$

| Penalización ($\lambda$) | Train Log-Loss | Val Log-Loss | Norma de pesos $\|w\|_2$ | Observación |
| :---: | :---: | :---: | :---: | :--- |
| **`0.0`** | **0.0538** | **0.0591** | **3.467** | **Óptimo elegido (brecha Train/Val mínima)** |
| `0.01` | 0.0538 | 0.0591 | 3.461 | Empate técnico con $\lambda = 0.0$ |
| `0.1` | 0.0542 | 0.0596 | 3.402 | Leve encogimiento de pesos |
| `1.0` | 0.0583 | 0.0650 | 2.932 | Sube el error en validación |
| `10.0` | 0.0901 | 0.1070 | 1.657 | Penalización excesiva (*underfitting*) |
| `50.0` | 0.1485 | 0.1737 | 0.965 | Fuerte *underfitting* |

### Evaluación final en conjunto de prueba (`Test`)

> Actualiza esta tabla con la salida de consola tras correr el bloque de evaluación final en `Test`.

| Modelo | Accuracy (Test) | Recall Maligno (Clase 0) |
| :--- | :---: | :---: |
| Regresión Logística NumPy (desde cero) | `[Accuracy Test]%` | `[Recall Maligno]%` |
| Referencia `scikit-learn` (`LogisticRegression`) | `[Accuracy Sklearn]%` | `[Recall Sklearn]%` |

## Gráficas generadas

Al ejecutar el script se generan automáticamente cuatro visualizaciones en `graficas_regresion_logistica/`:

| Archivo | Qué muestra |
| :--- | :--- |
| `01_convergencia_learning_rate.png` | Curvas de *Log-Loss* en validación frente a la línea base del azar ($\ln 2 \approx 0.693$) y decaimiento logarítmico de la norma del gradiente $\|\nabla L\|_2$. |
| `02_regularizacion_l2.png` | Comparación *Train vs. Val Log-Loss* y encogimiento de la norma de pesos $\|w\|_2$ al incrementar $\lambda$. |
| `03_evolucion_frontera_decision.png` | Instantáneas en épocas 1, 5, 25 y 200 sobre `worst radius` y `worst concave points` mostrando cómo el Descenso de Gradiente ajusta la frontera $\hat{p} = 0.5$. |
| `04_evaluacion_test_y_pesos.png` | Matriz de confusión sobre `Test` y Top 10 de pesos aprendidos $w_j$ para interpretabilidad clínica. |

## Requisitos e instalación

- Python 3.10 o superior
- Librerías: `numpy`, `matplotlib`, `scikit-learn`

```bash
pip install numpy matplotlib scikit-learn
```

## Ejecución

```bash
python regresion_logistica_cancer_mama.py
```

*(Si usas tu versión personalizada paso a paso, ejecuta `python programa1.py`.)*

## Estructura del repositorio

```text
.
├── regresion_logistica_cancer_mama.py   # Script completo con experimentos y 4 gráficas
├── programa1.py                         # Implementación guiada paso a paso
├── graficas_regresion_logistica/        # Gráficas PNG generadas por el script
├── guion_video_regresion_logistica.md   # Guion estructurado para la presentación en video
└── README.md                            # Documentación del proyecto
```
