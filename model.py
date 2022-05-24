import tensorflow as tf
from typing import List
from layers import *


class NAFNet(tf.keras.models.Model):
    def __init__(self,
                 width: int = 16,
                 n_middle_blocks: int = 1,
                 n_enc_blocks: List[int] = [1, 1, 1, 28],
                 n_dec_blocks: List[int] = [1, 1, 1, 1],
                 dropout_rate: float = 0.
                 ):
        super(NAFNet, self).__init__()

        self.to_features = tf.keras.layers.Conv2D(width,
                                                  kernel_size=3,
                                                  padding='SAME',
                                                  activation=None,
                                                  strides=1
                                                  )
        self.to_rgb = tf.keras.layers.Conv2D(3,
                                             kernel_size=3,
                                             padding='SAME',
                                             activation=None,
                                             strides=1
                                             )
        n_stages = len(n_dec_blocks)
        self.encoders = []
        self.downs = []
        for i, n in enumerate(n_enc_blocks):
            self.encoders.append(
                tf.keras.Sequential([
                    NAFBlock(width * (2 ** i), dropout_rate) for _ in range(n)
                ])
            )
            self.downs.append(
                tf.keras.layers.Conv2D(width * (2 ** (i + 1)),
                                       kernel_size=2,
                                       padding='valid',
                                       strides=2,
                                       activation=None
                                       )
            )
        self.middles = tf.keras.Sequential([
            NAFBlock(width * (2 ** n_stages), dropout_rate) for _ in range(n_middle_blocks)
        ])
        self.decoders = []
        self.ups = []
        for i, n in enumerate(n_dec_blocks):
            self.ups.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(width * (2 ** (n_stages - i)) * 2,
                                           kernel_size=1,
                                           padding='VALID',
                                           activation=None,
                                           strides=1
                                           ),
                    PixelShuffle(2)
                ])
            )
            self.decoders.append(
                tf.keras.Sequential([
                    NAFBlock(width * (2 ** (n_stages - (i + 1))), dropout_rate) for _ in range(n)
                ])
            )

    def forward(self, x):
        features = self.to_features(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            encs.append(features)
            features = down(features)

        features = self.middles(features)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            features = up(features)
            features = features + enc_skip
            features = decoder(features)

        x_res = self.to_rgb(features)
        x = x + x_res
        return x

    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)
