"""Problema 4: resolver dos EDOs con una red neuronal y comparar con la solucion analitica.

(a) x*y' + y = x^2*cos(x), y(0)=0  =>  y = x*sin(x) + 2*cos(x) - 2*sin(x)/x
(b) y'' = -y, y(0)=1, y'(0)=-0.5   =>  y = cos(x) - 0.5*sin(x)
"""

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

EPOCHS = 2500
LR = 0.01
x_np = np.linspace(-5, 5, 400).reshape(-1, 1).astype("float32")
x = tf.constant(x_np)
x0 = tf.constant([[0.0]], dtype=tf.float32)
mask = tf.constant(np.abs(x_np).ravel() > 1e-2)


def construir():
    return models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(32, activation="tanh"),
        layers.Dense(32, activation="tanh"),
        layers.Dense(1),
    ])


def analitica_a(t):
    t = np.asarray(t)
    salida = t * np.sin(t) + 2 * np.cos(t) - 2 * np.sin(t) / np.where(
        np.abs(t) < 1e-8, 1e-8, t)
    # Serie de Taylor cerca de 0 para evitar cancelacion numerica.
    cerca = np.abs(t) < 0.1
    salida[cerca] = t[cerca] ** 2 / 3 - t[cerca] ** 4 / 10
    return salida


def analitica_b(t):
    return np.cos(t) - 0.5 * np.sin(t)


def entrenar(orden):
    modelo = construir()
    optim = tf.keras.optimizers.Adam(LR)
    for _ in range(EPOCHS):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                y = modelo(x, training=True)
            dy = tape1.gradient(y, x)
        if orden == 2:
            d2y = tape2.gradient(dy, x)

        with tf.GradientTape() as tape_bc:
            tape_bc.watch(x0)
            y0 = modelo(x0, training=True)
        if orden == 2:
            dy0 = tape_bc.gradient(y0, x0)

        if orden == 1:
            residuo = x * dy + y - x ** 2 * tf.cos(x)
            residuo = tf.boolean_mask(residuo, mask)
            loss = tf.reduce_mean(residuo ** 2) + 10 * tf.reduce_mean(y0 ** 2)
        else:
            residuo = d2y + y
            loss = (tf.reduce_mean(residuo ** 2)
                    + 10 * tf.reduce_mean((y0 - 1.0) ** 2)
                    + 10 * tf.reduce_mean((dy0 + 0.5) ** 2))

        grads = tape2.gradient(loss, modelo.trainable_variables) if orden == 2 \
            else tape1.gradient(loss, modelo.trainable_variables)
        optim.apply_gradients(zip(grads, modelo.trainable_variables))
        del tape1, tape2
    return modelo


def main():
    m_a = entrenar(1)
    m_b = entrenar(2)
    ya = m_a.predict(x_np, verbose=0)
    yb = m_b.predict(x_np, verbose=0)
    ana_a = analitica_a(x_np)
    ana_b = analitica_b(x_np)
    mse_a = float(np.mean((ya - ana_a) ** 2))
    mse_b = float(np.mean((yb - ana_b) ** 2))
    print(f"(a) MSE vs analitica: {mse_a:.6e}")
    print(f"(b) MSE vs analitica: {mse_b:.6e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(x_np, ana_a, label="Analitica", linewidth=2)
    axes[0].plot(x_np, ya, "--", label="Red", linewidth=2)
    axes[0].set_title(f"(a) x y' + y = x^2 cos(x)\nMSE={mse_a:.2e}")
    axes[1].plot(x_np, ana_b, label="Analitica", linewidth=2)
    axes[1].plot(x_np, yb, "--", label="Red", linewidth=2)
    axes[1].set_title(f"(b) y'' = -y\nMSE={mse_b:.2e}")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/problema4_edo.png", dpi=150)


if __name__ == "__main__":
    main()
