import tensorflow as tf


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
                                   activation='linear',
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation='linear'
                                            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   activation='linear',
                                   strides=1,
                                   padding='VALID'
                                   )
        ])
        self.channel = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation='linear'
                                  ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation='linear'
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

        self.pool = tf.keras.layers.GlobalAvgPool2D()
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.reduction_filters,
                                  activation='relu'
                                  ),
            tf.keras.layers.Dense(self.n_filters,
                                  activation='linear'
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
                                   activation='linear',
                                   padding='VALID'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            activation='linear',
                                            padding='SAME'
                                            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   strides=1,
                                   activation='linear',
                                   padding='VALID'
                                   )
        ])
        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation='linear'
                                  ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.n_filters,
                                  activation='linear'
                                  )
        ])
        
    def call(self, inputs, *args, **kwargs):
        inputs = self.spatial(inputs) + inputs
        inputs = self.channel(inputs) + inputs
        return inputs
'''


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(inputs,
                                    block_size=self.upsample_rate
                                    )


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(inputs,
                          num_or_size_splits=2,
                          axis=-1
                          )
        return x1 * x2


class SimpleChannelAttention(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters
                 ):
        super(SimpleChannelAttention, self).__init__()
        self.n_filters = n_filters

        self.w = tf.keras.layers.Dense(self.n_filters,
                                       activation='linear',
                                       use_bias=False
                                       )

    def call(self, inputs, *args, **kwargs):
        attention = tf.reduce_mean(inputs,
                                   axis=[1, 2],
                                   keepdims=True
                                   )
        attention = self.w(attention)
        return attention * inputs


class NAFBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 dropout_rate: float,
                 dw_expansion: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(NAFBlock, self).__init__()

        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion

        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(self.dw_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=None
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation=None
                                            ),
            SimpleGate(),
            SimpleChannelAttention(self.n_filters),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=1,
                                   strides=1,
                                   padding='VALID',
                                   activation=None
                                   )
        ])
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)

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
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)

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
        inputs = self.drop1(self.spatial(inputs)) * self.beta + inputs
        inputs = self.drop2(self.channel(inputs)) * self.gamma + inputs
        return inputs
