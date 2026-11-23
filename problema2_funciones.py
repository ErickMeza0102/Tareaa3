"""Problema 2: aproximar dos funciones en [-1,1] con una red neuronal."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

Path("plots").mkdir(exist_ok=True)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

x = np.linspace(-1, 1, 1000).reshape(-1, 1).astype("float32")
FUNCIONES = {
    "(a) 3·sin(pi·x)": lambda t: 3 * np.sin(np.pi * t),
    "(b) 1 + 2x + 4x^3": lambda t: 1 + 2 * t + 4 * t ** 3,
}


def construir():
    modelo = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(32, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(1),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                   loss="mse", metrics=["mae"])
    return modelo


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (nombre, funcion) in zip(axes, FUNCIONES.items()):
        y = funcion(x).astype("float32")
        modelo = construir()
        modelo.fit(x, y, epochs=800, batch_size=32, verbose=0)
        y_pred = modelo.predict(x, verbose=0)
        mse = np.mean((y - y_pred) ** 2)

        ax.plot(x, y, label="Funcion real", linewidth=2)
        ax.plot(x, y_pred, "--", label="Red neuronal", linewidth=2)
        ax.set_title(f"{nombre}\nMSE = {mse:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(alpha=0.3)
        ax.legend()
        print(f"{nombre}: MSE={mse:.6e}")
    fig.tight_layout()
    fig.savefig("plots/problema2_funciones.png", dpi=150)


if __name__ == "__main__":
    main()
