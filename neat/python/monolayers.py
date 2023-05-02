import tensorflow as tf
from tensorflow import keras

# Source: https://github.com/neural-structured-additive-learning/deeptrafo/blob/main/inst/python/dtlayers/mono_layers.py


class MonoMultiLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim=None,
        dim_bsp=None,
        kernel_regularizer=None,
        trafo=None,
        initializer=keras.initializers.RandomNormal(seed=1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dim_bsp = dim_bsp
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer
        self.trafo = trafo

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer=self.initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

    def call(self, input):
        return tf.matmul(input, self.trafo(self.kernel, self.dim_bsp))

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "output_dim": self.output_dim,
                "dim_bsp": self.dim_bsp,
                "kernel_regularizer": self.kernel_regularizer,
                "initializer": self.initializer,
                "trafo": self.trafo,
            }
        )
        return config


def mono_trafo_multi(w, bsp_dim):
    w_res = tf.reshape(w, shape=[bsp_dim, int(w.shape[0] / bsp_dim)])
    w1 = tf.slice(w_res, [0, 0], [1, w_res.shape[1]])
    wrest = tf.math.softplus(
        tf.slice(w_res, [1, 0], [w_res.shape[0] - 1, w_res.shape[1]])
    )
    w_w_cons = tf.cumsum(tf.concat([w1, wrest], axis=0), axis=1)  # TODO: Check axis
    return tf.reshape(w_w_cons, shape=[w.shape[0], 1])
