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

import datasets
import math
import numpy as np



class MambaLayer(Layer):

    def __init__(self):
        super(MambaLayer, self).__init__()

        self.modelStates = 10
        self.modelInternalDim = 20
        self.delta_t_rank = 5

        self.in_projection = Dense(self.modelInternalDim * 2, activation="linear")
        self.x_projection = Dense(self.delta_t_rank + self.modelStates * 2, activation="linear")
        self.delta_t = Dense(self.modelInternalDim)

        self.out_projection = Dense(self.modelInternalDim, activation="linear")

        self.A = repeat(

            tf.range(1, self.modelStates + 1, dtype=tf.float32
                     ),
            " n -> d n", d=self.modelInternalDim,

        )

        self.A_log = tf.Variable(

            tf.math.log(self.A),
            trainable=True,
            dtype=tf.float32

        )

        self.D = tf.Variable(

            np.ones(self.modelInternalDim),
            dtype='float32'

        )

        self.conv = Conv1D(self.modelInternalDim, kernel_size=3, strides=1, padding="causal")

    def call(self, x):
        x = self.in_projection(x)

        x, res = tf.split(x, [self.modelInternalDim, self.modelInternalDim], axis=-1)

        #         x=rearrange(x,"b l d_in -> b d_in l")
        x = self.conv(x)
        #         x=rearrange(x,"b d_in l -> b l d_in")

        x = tf.nn.swish(x)

        x = self.ssm(x)

        return x

    def ssm(self, x):
        A = -tf.exp(self.A_log)
        D = self.D

        x_2 = self.x_projection(x)
        (delta, B, C) = tf.split(x_2, [self.delta_t_rank, self.modelStates, self.modelStates], axis=-1)

        delta = tf.nn.softmax(self.delta_t(delta))

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        dA = tf.einsum('bld,dn->bldn', delta, A)
        dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

        dA_cumsum = tf.pad(
            dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

        dA_cumsum = tf.reverse(dA_cumsum, axis=[1])

        dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

        dA_cumsum = tf.exp(dA_cumsum)
        dA_cumsum = tf.reverse(dA_cumsum, axis=[1])

        x = dB_u * dA_cumsum
        x = tf.math.cumsum(x, axis=1) / (dA_cumsum + 1e-12)

        x = tf.cast(x, dtype=tf.float32)

        y = tf.einsum('bldn,bln->bld', x, C)

        return y + u * D


class ResidualBlock(Layer):

    def __init__(self):
        super().__init__()

        self.mamba = MambaLayer()

    def call(self, x):
        residue = x

        mamba = self.mamba(x)

        return tf.nn.swish(mamba)