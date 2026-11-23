"""Problema 1: capa de Keras que transforma imagenes a color en grises.

No hay nada que entrenar: solo se disena la capa y se prueba que la
conversion funciona usando MNIST (se le agregan 3 canales falsos).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

Path("plots").mkdir(exist_ok=True)


class RgbToGrayscale(layers.Layer):
    """Convierte una imagen RGB (o por lotes) a escala de grises."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # tf.image.rgb_to_grayscale trabaja con ultima dimension = 3.
        return tf.image.rgb_to_grayscale(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


def main():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    originales = x_train[:3].astype("float32") / 255.0

    # Convertimos a RGB repitiendo el canal de gris 3 veces.
    rgb = np.repeat(originales[..., np.newaxis], 3, axis=-1)
    capa = RgbToGrayscale()
    grises = capa(tf.constant(rgb)).numpy()[..., 0]

    error = np.abs(grises - originales).mean()
    print(f"Error medio |gris - original|: {error:.6f}")

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i in range(3):
        axes[i, 0].imshow(rgb[i])
        axes[i, 0].set_title("RGB de entrada")
        axes[i, 1].imshow(grises[i], cmap="gray")
        axes[i, 1].set_title("Salida de la capa")
        axes[i, 2].imshow(originales[i], cmap="gray")
        axes[i, 2].set_title("Original MNIST")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("Capa personalizada RGB -> escala de grises")
    fig.tight_layout()
    fig.savefig("plots/problema1_grayscale.png", dpi=150)


if __name__ == "__main__":
    main()
