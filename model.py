import tensorflow as tf
from typing import Sequence, Optional
from layers import *


class NAFNet(tf.keras.models.Model):
    def __init__(
            self, width: int = 16, n_middle_blocks: int = 1, n_enc_blocks: Sequence[int] = (1, 1, 1, 28),
            n_dec_blocks: Sequence[int] = (1, 1, 1, 1), dropout_rate: float = 0.,
            train_size: Sequence[Optional[int]] = (None, 256, 256, 3), tlsc_rate: float = 1.5
    ):
        super(NAFNet, self).__init__()
        self.width = width
        self.n_middle_blocks = n_middle_blocks
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_rate = dropout_rate
        self.train_size = train_size
        self.tlsc_rate = tlsc_rate
        n_stages = len(n_enc_blocks)
        kh, kw = int(train_size[0] * tlsc_rate), int(train_size[1] * tlsc_rate)

        self.to_features = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding='SAME', activation=None,
            strides=1
        )
        self.to_rgb = tf.keras.layers.Conv2D(
            3,  kernel_size=3, padding='SAME', activation=None,
            strides=1
        )
        self.encoders = []
        self.downs = []
        for i, n in enumerate(n_enc_blocks):
            self.encoders.append(
                tf.keras.Sequential([
                    NAFBlock(
                        width * (2 ** i), dropout_rate, kh // (2 ** i), kw // (2 ** i)
                    ) for _ in range(n)
                ])
            )
            self.downs.append(
                tf.keras.layers.Conv2D(
                    width * (2 ** (i + 1)), kernel_size=2, padding='valid', strides=2,
                    activation=None
                )
            )
        self.middles = tf.keras.Sequential([
            NAFBlock(
                width * (2 ** n_stages), dropout_rate, kh // (2 ** n_stages), kw // (2 ** n_stages)
            ) for _ in range(n_middle_blocks)
        ])
        self.decoders = []
        self.ups = []
        for i, n in enumerate(n_dec_blocks):
            self.ups.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        width * (2 ** (n_stages - i)) * 2, kernel_size=1, padding='VALID', activation=None,
                        strides=1
                    ),
                    PixelShuffle(2)
                ])
            )
            self.decoders.append(
                tf.keras.Sequential([
                    NAFBlock(
                        width * (2 ** (n_stages - (i + 1))), dropout_rate,
                        kh // (2 ** (n_stages - (i + 1))), kw // (2 ** (n_stages - (i + 1)))
                    ) for _ in range(n)
                ])
            )

    @tf.function
    def forward(self, x, training=False):
        features = self.to_features(x, training=training)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features, training=training)
            encs.append(features)
            features = down(features, training=training)

        features = self.middles(features, training=training)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            features = up(features, training=training)
            features = features + enc_skip
            features = decoder(features, training=training)

        x_res = self.to_rgb(features, training=training)
        x = x + x_res
        return x

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)
