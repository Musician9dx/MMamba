import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Conv1D, Layer, Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from dataclasses import dataclass
from einops import rearrange, repeat
from typing import Union

from transformers import AutoTokenizer
from keras_nlp.layers import TokenAndPositionEmbedding
import datasets
import math
import numpy as np

from components.MambaLayer import MambaLayer,ResidualBlock


class Mamba(Model):

    def __init__(self, numMambaLayers=12, vocabSize=20, EmbDimension=20):

        super().__init__()

        self.emb = TokenAndPositionEmbedding(100, 20, 20)
        self.mambaRez = [ResidualBlock() for i in range(numMambaLayers)]
        self.dense = [Dense(20, activation="linear") for i in range(7)]

    def call(self, x):

        x = self.emb(x)
        x = tf.expand_dims(x, axis=0)

        for layer in self.mambaRez:
            x = layer(x)

        for layer in self.dense:
            x = layer(x)

        return x

    def generate(self, idx, new_tokens):
        for _ in range(new_tokens):
            logits = self.call(idx)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.reshape(tf.argmax(probs, axis=-1), (-1, 1))

            #             idx=tf.concat([idx, tf.squeeze(idx_next)], axis=0)
            idx = tf.constant(list(idx.numpy()) + list(idx_next.numpy()[0]))

        return idx

    def fitM(self, xb, yb, steps=10):
        optimizer = tf.keras.optimizers.Adam()

        for step in range(steps):
            with tf.GradientTape() as tape:
                logits = self.call(xb, yb)
                loss = sparse_categorical_crossentropy(xb, yb)
                print(f"Step: {step}", float(loss))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))