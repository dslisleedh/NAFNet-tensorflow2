from typing import Sequence, Union
import tensorflow as tf
import tensorflow.keras.backend as K


'''
class PlainBLock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 dw_expansion: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(PlainBLock, self).__init__()
        
        self.n_filters = n_filters
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion
        
        self.spatial = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.dw_filters,
                                   activation=None,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation=None
                                            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   activation=None,
                                   strides=1,
                                   padding='VALID'
                                   )
        ])
        self.channel = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation=None
                                  ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None
                                  )
        ])
        
    def call(self, inputs, *args, **kwargs):
        inputs = self.spatial(inputs) + inputs
        inputs = self.channel(inputs) + inputs
        return inputs
        
        
class CAModule(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 reduction_rate: int = 4
                 ):
        super(CAModule, self).__init__()

        self.n_filters = n_filters
        self.reduction_filters = int(self.n_filters // self.reduction_rate)

        self.pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.reduction_filters,
                                  activation='relu'
                                  ),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None
                                  )
        ])

    def call(self, inputs, *args, **kwargs):
        attention = self.pool(inputs)
        attention = self.forward(attention)
        inputs = inputs * attention


class BaselineBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 dw_expansion: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(BaselineBlock, self).__init__()

        self.n_filters = n_filters
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion
        
        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(self.dw_filters,
                                   kernel_size=1,
                                   strides=1,
                                   activation=None,
                                   padding='VALID'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            activation=None,
                                            padding='SAME'
                                            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   strides=1,
                                   activation=None,
                                   padding='VALID'
                                   )
        ])
        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation=None
                                  ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None
                                  )
        ])
        
    def call(self, inputs, *args, **kwargs):
        inputs = self.spatial(inputs) + inputs
        inputs = self.channel(inputs) + inputs
        return inputs
'''


def edge_padding2d(x, h_pad, w_pad):
    if h_pad[0] != 0:
        x_up = tf.gather(x, indices=[0], axis=1)
        x_up = tf.concat([x_up for _ in range(h_pad[0])], axis=1)
        x = tf.concat([x_up, x], axis=1)
    if h_pad[1] != 0:
        x_down = tf.gather(tf.reverse(x, axis=[1]), indices=[0], axis=1)
        x_down = tf.concat([x_down for _ in range(h_pad[1])], axis=1)
        x = tf.concat([x, x_down], axis=1)
    if w_pad[0] != 0:
        x_left = tf.gather(x, indices=[0], axis=2)
        x_left = tf.concat([x_left for _ in range(w_pad[0])], axis=2)
        x = tf.concat([x_left, x], axis=2)
    if w_pad[1] != 0:
        x_right= tf.gather(tf.reverse(x, axis=[2]), indices=[0], axis=2)
        x_right = tf.concat([x_right for _ in range(w_pad[1])], axis=2)
        x = tf.concat([x, x_right], axis=2)
    return x


class LocalAvgPool2D(tf.keras.layers.Layer):
    def __init__(
            self, local_size: Sequence[int]
    ):
        super(LocalAvgPool2D, self).__init__()
        self.local_size = local_size

    def call(self, inputs, training):
        if training:
            return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

        _, h, w, _ = inputs.get_shape().as_list()
        kh = min(h, self.local_size[0])
        kw = min(w, self.local_size[1])
        inputs = tf.pad(inputs,
                        [[0, 0],
                         [1, 0],
                         [1, 0],
                         [0, 0]]
                        )
        inputs = tf.cumsum(tf.cumsum(inputs, axis=2), axis=1)
        s1 = tf.slice(inputs,
                      [0, 0, 0, 0],
                      [-1, kh, kw, -1]
                      )
        s2 = tf.slice(inputs,
                      [0, 0, (w - kw)+1, 0],
                      [-1, kw, -1, -1]
                      )
        s3 = tf.slice(inputs,
                      [0, (h - kh)+1, 0, 0],
                      [-1, -1, kw, -1]
                      )
        s4 = tf.slice(inputs,
                      [0, (h - kh)+1, (w - kw)+1, 0],
                      [-1, -1, -1, -1]
                      )
        local_ap = (s4 + s1 - s2 - s3) / (kh * kw)

        _, h_, w_, _ = local_ap.get_shape().as_list()
        h_pad, w_pad = [(h - h_) // 2, (h - h_ + 1) // 2], [(w - w_) // 2, (w - w_ + 1) // 2]
        local_ap = edge_padding2d(local_ap, h_pad, w_pad)
        return local_ap


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            inputs, block_size=self.upsample_rate
        )


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(
            inputs, num_or_size_splits=2, axis=-1
        )
        return x1 * x2


class SimpleChannelAttention(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, kh: int, kw: int
    ):
        super(SimpleChannelAttention, self).__init__()
        self.n_filters = n_filters
        self.kh = kh
        self.kw = kw

        self.pool = LocalAvgPool2D((kh, kw))
        self.w = tf.keras.layers.Dense(
            self.n_filters, activation=None
        )

    def call(self, inputs, *args, **kwargs):
        attention = self.pool(inputs)
        attention = self.w(attention)
        return attention * inputs


class NAFBlock(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, dropout_rate: float, kh: int,
            kw: int, dw_expansion: int = 2, ffn_expansion: int = 2
    ):
        super(NAFBlock, self).__init__()
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.kh = kh
        self.kw = kw
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion

        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(
                self.dw_filters, kernel_size=1, strides=1, padding='VALID',
                activation=None
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=1, padding='SAME', activation=None
            ),
            SimpleGate(),
            SimpleChannelAttention(
                self.n_filters, self.kh, self.kw
            ),
            tf.keras.layers.Conv2D(
                self.n_filters, kernel_size=1, strides=1, padding='VALID',
                activation=None
            )
        ])
        self.spatial_drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation=None
                                  ),
            SimpleGate(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None
                                  )
        ])
        self.channel_drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.beta = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32
        )
        self.gamma = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, *args, **kwargs):
        inputs = self.spatial_drop(self.spatial(inputs)) * self.beta + inputs
        inputs = self.channel_drop(self.channel(inputs)) * self.gamma + inputs
        return inputs
