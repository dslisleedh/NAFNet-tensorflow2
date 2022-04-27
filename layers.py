import tensorflow as tf


'''
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


class PlainBLock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 expension_rate: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(PlainBLock, self).__init__()
        
        self.n_filters = n_filters
        self.dw_filters = n_filters * expension_rate
        self.ffn_filters = n_filters * ffn_expansion
        
        self.forward1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.dw_filters,
                                   activation='linear',
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding='SAME',
                                            activation='linear'
                                            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   activation='linear',
                                   strides=1,
                                   padding='SAME'
                                   )
        ])
        self.forward2 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation='linear'
                                  ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation='linear'
                                  )
        ])
        
    def call(self, inputs, *args, **kwargs):
        inputs = self.forward1(inputs) + inputs
        inputs = self.forward2(inputs) + inputs
        return inputs


class BaselineBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 expension_rate: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(BaselineBlock, self).__init__()

        self.n_filters = n_filters
        self.dw_filters = n_filters * expension_rate
        self.ffn_filters = n_filters * ffn_expansion
        
        self.forward1 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(self.dw_filters,
                                   kernel_size=3,
                                   strides=1,
                                   activation='linear',
                                   padding='SAME'
                                   ),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            activation='linear',
                                            padding='SAME'
                                            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   activation='linear',
                                   padding='SAME'
                                   )
        ])
        self.forward2 = tf.keras.Sequential([
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
        inputs = self.forward1(inputs) + inputs
        inputs = self.forward2(inputs) + inputs
        return inputs
'''


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

        self.pool = tf.keras.layers.GlobalAvgPool2D()
        self.w = tf.keras.layers.Dense(self.n_filters,
                                       activation='linear',
                                       use_bias=False
                                       )

    def call(self, inputs, *args, **kwargs):
        attention = self.pool(inputs)
        attention = self.w(attention)
        return attention * inputs


class NAFBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters: int,
                 expension_rate: int = 2,
                 ffn_expansion: int = 2
                 ):
        super(NAFBlock, self).__init__()

        self.n_filters = n_filters
        self.dw_filters = n_filters * expension_rate
        self.ffn_filters = n_filters * ffn_expansion

        self.forward1 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(self.dw_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='linear'
                                   ),
            SimpleGate(),
            SimpleChannelAttention(),
            tf.keras.layers.Conv2D(self.n_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='SAME',
                                   activation='linear'
                                   )
        ])
        self.forward2 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation='linear'
                                  ),
            SimpleGate(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation='linear'
                                  )
        ])

    def call(self, inputs, *args, **kwargs):
        inputs = self.forward1(inputs) + inputs
        inputs = self.forward2(inputs) + inputs
        return inputs
