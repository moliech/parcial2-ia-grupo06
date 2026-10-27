"""Modulo de Perceptron binario con funcion de activacion sigmoide y descenso
del gradiente implementado exclusivamente con NumPy desde cero.
Diseñado para la clasificacion de consumo excesivo de agua potable en Cartago.
"""

import numpy as np
import matplotlib.pyplot as plt


class Estandarizador:
    """Escalador de caracteristicas basado en media (media=0) y desviacion (std=1)."""

    def __init__(self):
        self.media = None
        self.desviacion = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Calcula media y desviacion estandar por columna y estandariza los datos."""
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        self.desviacion = np.where(self.desviacion == 0, 1.0, self.desviacion)
        return (X - self.media) / self.desviacion

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la estandarizacion usando los parametros previamente calculados."""
        if self.media is None or self.desviacion is None:
            raise ValueError("El estandarizador debe ajustarse con fit_transform antes de transformar.")
        return (X - self.media) / self.desviacion


class PerceptronSigmoide:
    """Perceptron con activacion sigmoide y optimizacion por descenso del gradiente."""

    def __init__(self, tasa_aprendizaje: float = 0.1, epocas: int = 500, semilla: int = 42):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.semilla = semilla
        self.pesos = None
        self.sesgo = 0.0
        self.historial_perdida = []
        self.historial_exactitud = []
        self.pesos_iniciales = None
        self.sesgo_inicial = None
        self.perdida_inicial = None

    @staticmethod
    def sigmoide(z: np.ndarray) -> np.ndarray:
        """Funcion de activacion sigmoide con acotamiento numerico para evitar desbordamiento."""
        z_acotado = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_acotado))

    @staticmethod
    def perdida_logaritmica(y_real: np.ndarray, y_prob: np.ndarray) -> float:
        """Calcula la perdida de entropia cruzada binaria (Binary Cross-Entropy Loss)."""
        epsilon = 1e-15
        p_segura = np.clip(y_prob, epsilon, 1.0 - epsilon)
        return float(-np.mean(y_real * np.log(p_segura) + (1.0 - y_real) * np.log(1.0 - p_segura)))

    @staticmethod
    def exactitud(y_real: np.ndarray, y_prob: np.ndarray, umbral: float = 0.5) -> float:
        """Calcula el porcentaje de aciertos clasificando con un umbral de decision."""
        predicciones = (y_prob >= umbral).astype(int)
        return float(np.mean(predicciones == y_real))

    def predecir_probabilidad(self, X: np.ndarray) -> np.ndarray:
        """Calcula la probabilidad P(y=1|X) mediante la combinacion lineal y la sigmoide."""
        combinacion_lineal = X @ self.pesos + self.sesgo
        return self.sigmoide(combinacion_lineal)

    def predecir(self, X: np.ndarray, umbral: float = 0.5) -> np.ndarray:
        """Retorna la clase predicha (0 o 1)."""
        probabilidades = self.predecir_probabilidad(X)
        return (probabilidades >= umbral).astype(int)

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Entrena el perceptron utilizando descenso del gradiente sobre la perdida logaritmica."""
        n_muestras, n_caracteristicas = X.shape

        np.random.seed(self.semilla)
        self.pesos = np.random.randn(n_caracteristicas) * 0.01
        self.sesgo = 0.0

        self.pesos_iniciales = self.pesos.copy()
        self.sesgo_inicial = float(self.sesgo)

        prob_inicial = self.predecir_probabilidad(X)
        self.perdida_inicial = self.perdida_logaritmica(y, prob_inicial)
        exactitud_inicial = self.exactitud(y, prob_inicial)

        self.historial_perdida = []
        self.historial_exactitud = []

        for epoca in range(self.epocas):
            probabilidades = self.predecir_probabilidad(X)

            perdida = self.perdida_logaritmica(y, probabilidades)
            acc = self.exactitud(y, probabilidades)
            self.historial_perdida.append(perdida)
            self.historial_exactitud.append(acc)

            # Gradientes de la perdida
            error = probabilidades - y
            gradiente_w = (X.T @ error) / n_muestras
            gradiente_b = float(np.mean(error))

            self.pesos -= self.tasa_aprendizaje * gradiente_w
            self.sesgo -= self.tasa_aprendizaje * gradiente_b

        prob_final = self.predecir_probabilidad(X)
        perdida_final = self.perdida_logaritmica(y, prob_final)
        exactitud_final = self.exactitud(y, prob_final)

        return {
            "pesos_iniciales": self.pesos_iniciales,
            "sesgo_inicial": self.sesgo_inicial,
            "perdida_inicial": self.perdida_inicial,
            "exactitud_inicial": exactitud_inicial,
            "pesos_finales": self.pesos,
            "sesgo_final": self.sesgo,
            "perdida_final": perdida_final,
            "exactitud_final": exactitud_final,
        }


def graficar_curva_aprendizaje(historial_perdida: list, ruta_guardado: str = "data/curva_aprendizaje.png") -> None:
    """Genera y guarda el grafico de convergencia de la perdida logaritmica."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historial_perdida) + 1), historial_perdida, color="#0284c7", linewidth=2)
    plt.title("Curva de Aprendizaje: Pérdida Logarítmica vs Épocas", fontsize=13, fontweight="bold")
    plt.xlabel("Época de Entrenamiento", fontsize=11)
    plt.ylabel("Pérdida (Binary Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()
    print(f"[Grafico] Curva de aprendizaje guardada exitosamente en {ruta_guardado}")


def graficar_frontera_decision(
    X_orig: np.ndarray,
    y: np.ndarray,
    pesos: np.ndarray,
    sesgo: float,
    media: np.ndarray,
    desviacion: np.ndarray,
    ruta_guardado: str = "data/frontera_decision.png"
) -> None:
    """Genera la frontera de decision en las unidades originales de los datos (m3 y personas).
    Ecuacion des-estandarizada:
    x2_orig = std2 * [ -(w1/w2) * ((x1_orig - mean1)/std1) - (b/w2) ] + mean2
    """
    plt.figure(figsize=(9, 6))

    mascara_0 = (y == 0)
    mascara_1 = (y == 1)

    plt.scatter(
        X_orig[mascara_0, 0], X_orig[mascara_0, 1],
        color="#10b981", label="0: Consumo Normal", alpha=0.85, edgecolors="k", s=55
    )
    plt.scatter(
        X_orig[mascara_1, 0], X_orig[mascara_1, 1],
        color="#ef4444", label="1: Consumo Excesivo", alpha=0.85, edgecolors="k", s=55
    )

    w1, w2 = pesos[0], pesos[1]
    b = sesgo
    m1, m2 = media[0], media[1]
    s1, s2 = desviacion[0], desviacion[1]

    x1_vals = np.linspace(X_orig[:, 0].min() - 2, X_orig[:, 0].max() + 2, 200)

    z1_vals = (x1_vals - m1) / s1
    z2_vals = -(w1 * z1_vals + b) / w2
    x2_vals = z2_vals * s2 + m2

    plt.plot(x1_vals, x2_vals, color="#1e293b", linewidth=2.5, linestyle="--", label="Frontera de Decisión (P=0.5)")
    plt.title("Frontera de Decisión del Perceptrón (Unidades Originales)", fontsize=13, fontweight="bold")
    plt.xlabel("Consumo Mensual de Agua (m³/mes)", fontsize=11)
    plt.ylabel("Personas por Hogar", fontsize=11)
    plt.ylim(X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1)
    plt.xlim(X_orig[:, 0].min() - 2, X_orig[:, 0].max() + 2)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()
    print(f"[Grafico] Frontera de decision guardada exitosamente en {ruta_guardado}")