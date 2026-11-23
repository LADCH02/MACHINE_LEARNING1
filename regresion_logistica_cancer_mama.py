"""
Clasificación Binaria con Regresión Logística y Descenso de Gradiente desde Cero
Dataset: Breast Cancer Wisconsin (Diagnostic) - scikit-learn

Este script implementa la Regresión Logística paso a paso en NumPy para exponer
su mecánica interna (suma lineal, sigmoide, Binary Cross-Entropy / Log-Loss,
gradientes y actualización de pesos) y justifica empíricamente:
  1. El escalado de variables (StandardScaler sin fuga de datos).
  2. La tasa de aprendizaje (learning_rate / eta).
  3. La regularización L2 (lambda).
  4. La evolución de la frontera de decisión época por época.
  5. La evaluación final en prueba y validación contra scikit-learn.

Uso:
    python regresion_logistica_cancer_mama.py
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Cambia a "TkAgg" o elimina esta línea si quieres ventanas interactivas
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. IMPLEMENTACIÓN TRANSPARENTE DE REGRESIÓN LOGÍSTICA (NUMPY DESDE CERO)
# ---------------------------------------------------------------------------

class RegresionLogisticaGD:
    """Regresión Logística binaria entrenada con Batch Gradient Descent y regularización L2."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 300, reg_lambda: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = 0.0
        self.history = {}
        self.snapshots = {}

    @staticmethod
    def sigmoide(z: np.ndarray) -> np.ndarray:
        """Transforma cualquier puntaje real z en una probabilidad en (0, 1)."""
        z_seguro = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_seguro))

    def calcular_log_loss(self, y: np.ndarray, p_hat: np.ndarray, incluir_reg: bool = True) -> float:
        """Binary Cross-Entropy (Log-Loss) promedio más penalización L2 opcional."""
        eps = 1e-12
        p_clip = np.clip(p_hat, eps, 1.0 - eps)
        loss_datos = -np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
        if incluir_reg and self.reg_lambda > 0.0:
            penalizacion_l2 = (self.reg_lambda / (2.0 * len(y))) * float(np.sum(self.w ** 2))
            return float(loss_datos + penalizacion_l2)
        return float(loss_datos)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad estimada de pertenecer a la clase 1 (Benigno)."""
        z = X @ self.w + self.b
        return self.sigmoide(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convierte probabilidades en clases binarias usando el umbral indicado."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        snapshot_epochs: tuple[int, ...] = (),
    ):
        """Entrena los pesos w y sesgo b registrando toda la mecánica interna por época."""
        m, n_features = X_train.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "grad_norm": [],
            "weight_norm": [],
        }
        self.snapshots = {}

        for epoch in range(1, self.epochs + 1):
            # Paso 1: Suma lineal
            z = X_train @ self.w + self.b

            # Paso 2: Probabilidades con Sigmoide
            p_hat = self.sigmoide(z)

            # Paso 3: Error por paciente (p_hat - y) y gradientes promedio
            error = p_hat - y_train
            dw = (X_train.T @ error) / m + (self.reg_lambda / m) * self.w
            db = float(np.mean(error))

            # Paso 4: Actualización por Descenso de Gradiente
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Registro del estado interno después de la actualización
            p_train_actual = self.predict_proba(X_train)
            self.history["train_loss"].append(self.calcular_log_loss(y_train, p_train_actual, incluir_reg=False))
            self.history["train_acc"].append(float(np.mean((p_train_actual >= 0.5) == y_train)))
            self.history["grad_norm"].append(float(np.linalg.norm(np.append(dw, db))))
            self.history["weight_norm"].append(float(np.linalg.norm(self.w)))

            if X_val is not None and y_val is not None:
                p_val = self.predict_proba(X_val)
                self.history["val_loss"].append(self.calcular_log_loss(y_val, p_val, incluir_reg=False))
                self.history["val_acc"].append(float(np.mean((p_val >= 0.5) == y_val)))

            if epoch in snapshot_epochs:
                self.snapshots[epoch] = {
                    "w": self.w.copy(),
                    "b": float(self.b),
                    "loss": self.history["train_loss"][-1],
                    "acc": self.history["train_acc"][-1],
                }

        return self


# ---------------------------------------------------------------------------
# 2. FUNCIONES DE VISUALIZACIÓN PARA EXPONER EL PROCESO INTERNO EN EL VIDEO
# ---------------------------------------------------------------------------

def configurar_estilo():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 10,
    })


def graficar_experimento_lr(resultados_lr: dict, out_dir: Path):
    """Gráfica 1: Comparación de tasas de aprendizaje (Log-Loss y norma del gradiente)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colores = ["#8e44ad", "#2980b9", "#27ae60", "#e74c3c"]

    for (lr, hist), color in zip(resultados_lr.items(), colores):
        epocas = np.arange(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epocas, hist["val_loss"], label=f"η = {lr}", color=color, linewidth=2)
        axes[1].plot(epocas, hist["grad_norm"], label=f"η = {lr}", color=color, linewidth=2)

    axes[0].axhline(np.log(2), color="#7f8c8d", linestyle=":", linewidth=1.4, label="Azar (ln 2 ≈ 0.693)")
    axes[0].set_title("Convergencia de Log-Loss (Validación) según η")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary Cross-Entropy (Log-Loss)")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=True)

    axes[1].set_title("Magnitud del Gradiente ||∇L||₂ durante el entrenamiento")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Norma L2 del gradiente")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=True)

    fig.suptitle("Experimento 1: Justificación empírica de la tasa de aprendizaje (η)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = out_dir / "01_convergencia_learning_rate.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_experimento_lambda(lambdas: list[float], metricas_reg: dict, mejor_lambda: float, out_dir: Path):
    """Gráfica 2: Impacto de la regularización L2 en pérdida (Train vs Val) y norma de pesos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    etiquetas_x = [str(l) for l in lambdas]
    pos = np.arange(len(lambdas))

    axes[0].plot(pos, metricas_reg["train_loss"], marker="o", linewidth=2, color="#2980b9", label="Log-Loss Train")
    axes[0].plot(pos, metricas_reg["val_loss"], marker="s", linewidth=2, color="#e67e22", label="Log-Loss Validación")
    idx_mejor = lambdas.index(mejor_lambda)
    axes[0].axvline(idx_mejor, color="#27ae60", linestyle="--", linewidth=1.5, label=f"Elegido λ = {mejor_lambda}")
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(etiquetas_x)
    axes[0].set_title("Log-Loss en Entrenamiento vs. Validación")
    axes[0].set_xlabel("Penalización L2 (λ)")
    axes[0].set_ylabel("Log-Loss final")
    axes[0].legend(frameon=True)

    axes[1].bar(pos, metricas_reg["weight_norm"], color="#34495e", alpha=0.85, width=0.55)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(etiquetas_x)
    axes[1].set_title("Encogimiento de pesos ||w||₂ por regularización L2")
    axes[1].set_xlabel("Penalización L2 (λ)")
    axes[1].set_ylabel("Norma L2 de los pesos ||w||₂")

    fig.suptitle("Experimento 2: Control de sobreajuste con regularización L2 (λ)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = out_dir / "02_regularizacion_l2.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_evolucion_frontera_2d(
    modelo_2d: RegresionLogisticaGD,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    nombres_2d: tuple[str, str],
    epocas_mostrar: tuple[int, ...],
    out_dir: Path,
):
    """Gráfica 3: Evolución paso a paso de la frontera de decisión en 2D."""
    fig, axes = plt.subplots(1, len(epocas_mostrar), figsize=(16, 4.2), sharex=True, sharey=True)

    x_min, x_max = X_train_2d[:, 0].min() - 0.6, X_train_2d[:, 0].max() + 0.6
    y_min, y_max = X_train_2d[:, 1].min() - 0.6, X_train_2d[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    malla = np.c_[xx.ravel(), yy.ravel()]

    for ax, epoca in zip(axes, epocas_mostrar):
        snap = modelo_2d.snapshots[epoca]
        z_malla = malla @ snap["w"] + snap["b"]
        probs = RegresionLogisticaGD.sigmoide(z_malla).reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.45)
        ax.contour(xx, yy, probs, levels=[0.5], colors=["#111111"], linewidths=2.2, linestyles="--")

        ax.scatter(
            X_train_2d[y_train == 0, 0],
            X_train_2d[y_train == 0, 1],
            c="#c0392b",
            edgecolors="white",
            linewidth=0.5,
            s=26,
            label="Maligno (0)",
        )
        ax.scatter(
            X_train_2d[y_train == 1, 0],
            X_train_2d[y_train == 1, 1],
            c="#2471a3",
            edgecolors="white",
            linewidth=0.5,
            s=26,
            label="Benigno (1)",
        )
        ax.set_title(f"Época {epoca}\nLoss={snap['loss']:.3f} | Acc={snap['acc']:.1%}", fontsize=10.5)
        ax.set_xlabel(f"{nombres_2d[0]} (estandarizado)")

    axes[0].set_ylabel(f"{nombres_2d[1]} (estandarizado)")
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.suptitle("Mecánica interna: cómo se ajusta la frontera de decisión (p̂ = 0.5) con el Descenso de Gradiente", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = out_dir / "03_evolucion_frontera_decision.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def graficar_evaluacion_y_pesos(cm: np.ndarray, pesos: np.ndarray, feature_names: np.ndarray, out_dir: Path):
    """Gráfica 4: Matriz de confusión en Test y Top 10 variables más influyentes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Matriz de confusión
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred: Maligno (0)", "Pred: Benigno (1)"])
    axes[0].set_yticklabels(["Real: Maligno (0)", "Real: Benigno (1)"])
    axes[0].set_title("Matriz de Confusión (Conjunto de Prueba)")
    axes[0].grid(False)

    etiquetas_celda = [["VN / Det. Maligno", "Falso Benigno (Riesgo)"], ["Falso Maligno", "VP / Det. Benigno"]]
    for i in range(2):
        for j in range(2):
            color_txt = "white" if cm[i, j] > cm.max() * 0.55 else "#111111"
            axes[0].text(
                j,
                i,
                f"{cm[i, j]}\n({etiquetas_celda[i][j]})",
                ha="center",
                va="center",
                color=color_txt,
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Top 10 pesos por magnitud absoluta
    idx_top = np.argsort(np.abs(pesos))[-10:]
    valores = pesos[idx_top]
    nombres = feature_names[idx_top]
    colores_bar = ["#c0392b" if v < 0 else "#2471a3" for v in valores]

    axes[1].barh(np.arange(10), valores, color=colores_bar, alpha=0.88)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_yticks(np.arange(10))
    axes[1].set_yticklabels(nombres)
    axes[1].set_title("Top 10 pesos aprendidos (w) — Interpretabilidad")
    axes[1].set_xlabel("Peso w_j (negativo → empuja a Maligno | positivo → Benigno)")

    fig.suptitle("Evaluación en Prueba (Test) e Interpretación de Pesos del Modelo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta = out_dir / "04_evaluacion_test_y_pesos.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 3. FLUJO PRINCIPAL DE EXPERIMENTOS
# ---------------------------------------------------------------------------

def main():
    configurar_estilo()
    out_dir = Path(__file__).resolve().parent / "graficas_regresion_logistica"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carga y exploración básica del dataset
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = Maligno, 1 = Benigno
    feature_names = data.feature_names

    print("=" * 72)
    print("DATASET: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Muestras: {X.shape[0]} | Características: {X.shape[1]} | Nulos: {np.isnan(X).sum()}")
    print(f"Distribución de clases -> Maligno (0): {(y == 0).sum()} | Benigno (1): {(y == 1).sum()}")
    print("=" * 72)

    # 2) División estratificada Train (60%) / Val (20%) / Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # 3) Escalado (ajustado SOLO en train para evitar data leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4) Experimento 1: Comparación de tasas de aprendizaje (η)
    tasas = [0.005, 0.05, 0.5, 5.0]
    resultados_lr = {}
    print("\n[Experimento 1] Comparación de tasa de aprendizaje (300 épocas, λ=0):")
    for lr in tasas:
        modelo = RegresionLogisticaGD(learning_rate=lr, epochs=300, reg_lambda=0.0)
        modelo.fit(X_train_s, y_train, X_val_s, y_val)
        resultados_lr[lr] = modelo.history
        print(
            f"  η = {lr:<5} | Val Log-Loss final = {modelo.history['val_loss'][-1]:.4f} "
            f"| Val Acc = {modelo.history['val_acc'][-1]:.2%} "
            f"| ||∇L||₂ final = {modelo.history['grad_norm'][-1]:.5f}"
        )

    mejor_lr = min(resultados_lr, key=lambda k: resultados_lr[k]["val_loss"][-1])
    ruta_g1 = graficar_experimento_lr(resultados_lr, out_dir)

    # 5) Experimento 2: Selección de regularización L2 (λ) usando el mejor η
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0]
    metricas_reg = {"train_loss": [], "val_loss": [], "val_acc": [], "weight_norm": []}
    print(f"\n[Experimento 2] Regularización L2 con η = {mejor_lr} (300 épocas):")
    for lam in lambdas:
        modelo = RegresionLogisticaGD(learning_rate=mejor_lr, epochs=300, reg_lambda=lam)
        modelo.fit(X_train_s, y_train, X_val_s, y_val)
        metricas_reg["train_loss"].append(modelo.history["train_loss"][-1])
        metricas_reg["val_loss"].append(modelo.history["val_loss"][-1])
        metricas_reg["val_acc"].append(modelo.history["val_acc"][-1])
        metricas_reg["weight_norm"].append(modelo.history["weight_norm"][-1])
        print(
            f"  λ = {lam:<5} | Train Loss = {modelo.history['train_loss'][-1]:.4f} "
            f"| Val Loss = {modelo.history['val_loss'][-1]:.4f} "
            f"| Val Acc = {modelo.history['val_acc'][-1]:.2%} "
            f"| ||w||₂ = {modelo.history['weight_norm'][-1]:.3f}"
        )

    mejor_lambda = lambdas[int(np.argmin(metricas_reg["val_loss"]))]
    ruta_g2 = graficar_experimento_lambda(lambdas, metricas_reg, mejor_lambda, out_dir)

    # 6) Entrenamiento final en Train+Val con los mejores hiperparámetros y evaluación en Test
    X_trainval_s = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])

    modelo_final = RegresionLogisticaGD(learning_rate=mejor_lr, epochs=400, reg_lambda=mejor_lambda)
    modelo_final.fit(X_trainval_s, y_trainval)

    p_test = modelo_final.predict_proba(X_test_s)
    y_pred_test = (p_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_test)

    # Validación cruzada de implementación contra scikit-learn
    c_sklearn = 1.0 / mejor_lambda if mejor_lambda > 0 else 1e6
    sk_model = LogisticRegression(C=c_sklearn, max_iter=1000, random_state=42)
    sk_model.fit(X_trainval_s, y_trainval)
    y_pred_sk = sk_model.predict(X_test_s)

    print("\n" + "=" * 72)
    print(f"EVALUACIÓN FINAL EN TEST (η={mejor_lr}, λ={mejor_lambda}, 400 épocas)")
    print(f"  Nuestro modelo (NumPy GD) -> Acc: {accuracy_score(y_test, y_pred_test):.2%} | "
          f"Recall Maligno (0): {recall_score(y_test, y_pred_test, pos_label=0):.2%} | "
          f"F1 (Benigno): {f1_score(y_test, y_pred_test):.4f} | "
          f"ROC-AUC: {roc_auc_score(y_test, p_test):.4f}")
    print(f"  Referencia scikit-learn   -> Acc: {accuracy_score(y_test, y_pred_sk):.2%} | "
          f"Recall Maligno (0): {recall_score(y_test, y_pred_sk, pos_label=0):.2%} | "
          f"F1 (Benigno): {f1_score(y_test, y_pred_sk):.4f}")
    print(f"  Matriz de confusión Test:\n{cm}")
    print("=" * 72)

    # 7) Experimento 3: Visualización de la frontera de decisión en 2D (proceso interno)
    # Elegimos las 2 variables con mayor peso absoluto en el modelo entrenado (o clásicas clínicas)
    idx_f1 = list(feature_names).index("worst radius")
    idx_f2 = list(feature_names).index("worst concave points")
    X_train_2d = X_train_s[:, [idx_f1, idx_f2]]
    epocas_mostrar = (1, 5, 25, 200)

    modelo_2d = RegresionLogisticaGD(learning_rate=0.5, epochs=200, reg_lambda=0.1)
    modelo_2d.fit(X_train_2d, y_train, snapshot_epochs=epocas_mostrar)
    ruta_g3 = graficar_evolucion_frontera_2d(
        modelo_2d,
        X_train_2d,
        y_train,
        (feature_names[idx_f1], feature_names[idx_f2]),
        epocas_mostrar,
        out_dir,
    )

    ruta_g4 = graficar_evaluacion_y_pesos(cm, modelo_final.w, feature_names, out_dir)

    print("\nGráficas generadas correctamente:")
    for r in (ruta_g1, ruta_g2, ruta_g3, ruta_g4):
        print(f"  - {r}")


if __name__ == "__main__":
    main()