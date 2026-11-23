"""Problema 3: capa entrenable de polinomio cublico y ajuste a cos(2x)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

Path("plots").mkdir(exist_ok=True)
tf.random.set_seed(42)
np.random.seed(42)


class PolinomioCubico(layers.Layer):
    """f(x) = a0 + a1*x + a2*x^2 + a3*x^3, con ai entrenables."""

    def build(self, input_shape):
        self.a0 = self.add_weight(shape=(), initializer="zeros", name="a0")
        self.a1 = self.add_weight(shape=(), initializer="zeros", name="a1")
        self.a2 = self.add_weight(shape=(), initializer="zeros", name="a2")
        self.a3 = self.add_weight(shape=(), initializer="zeros", name="a3")

    def call(self, x):
        return self.a0 + self.a1 * x + self.a2 * x ** 2 + self.a3 * x ** 3


def main():
    x = np.linspace(-1, 1, 400).reshape(-1, 1).astype("float32")
    y = np.cos(2 * x).astype("float32")

    capa = PolinomioCubico()
    modelo = models.Sequential([layers.Input(shape=(1,)), capa])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss="mse")
    modelo.fit(x, y, epochs=2500, batch_size=64, verbose=0)

    a0, a1, a2, a3 = [float(w.numpy()) for w in
                      (capa.a0, capa.a1, capa.a2, capa.a3)]
    y_pred = modelo.predict(x, verbose=0)
    mse = float(np.mean((y - y_pred) ** 2))
    print(f"Coeficientes: a0={a0:.5f}, a1={a1:.5f}, a2={a2:.5f}, a3={a3:.5f}")
    print(f"MSE = {mse:.6e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, label="cos(2x)", linewidth=2)
    ax.plot(x, y_pred, "--", label="Polinomio entrenado", linewidth=2)
    ax.set_title(f"Polinomio cubico: {a0:.3f} {a1:+.3f}x {a2:+.3f}x^2 "
                 f"{a3:+.3f}x^3\nMSE={mse:.2e}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/problema3_polinomio.png", dpi=150)


if __name__ == "__main__":
    main()
