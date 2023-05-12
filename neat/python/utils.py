from enum import Enum
from typing import Iterable, Sequence
import numpy as np

import tensorflow as tf
from keras import constraints, initializers
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate, Dense, Input, Dropout
from keras.optimizers import Adam, Optimizer
from keras import regularizers
from tensorflow_probability import distributions as tfd

from monolayers import MonoMultiLayer, mono_trafo_multi
from neat_model_class import NEATModel
from tffuns import RowTensor


class ModelType(Enum):
    TP = "tp"
    LS = "ls"
    INTER = "inter"


def mlp_with_default_layer(
        size: Sequence[int], default_layer: callable, dropout: float, **layer_kwargs
) -> callable:
    def inner(inp):
        x = default_layer(units=size[0], **layer_kwargs)(inp)
        for i in range(1, len(size)):
            if dropout > 0:
                x = Dropout(dropout)(x)
            x = default_layer(units=size[i], **layer_kwargs)(x)
        return x

    return inner


def relu_network(size: Iterable[int], dropout: float) -> callable:
    return mlp_with_default_layer(
        size,
        default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
        dropout=dropout,
    )


def feature_specific_network(
        size: Iterable[int],
        default_layer: callable,
        dropout: float,
) -> callable:
    def inner(x):
        return Concatenate(axis=1)(
            [
                mlp_with_default_layer(size, default_layer, dropout=dropout)(xx)
                for xx in tf.split(x, num_or_size_splits=x.shape[1], axis=1)
            ]
        )

    return inner


def constraint_xavier_init(shape, dtype=None):
    fan_in = shape[0]
    fan_out = shape[1]
    limit = np.sqrt(6. / (fan_in + fan_out))
    return tf.random.uniform(shape, minval=0., maxval=limit, dtype=dtype)

def constraint_xavier_p_init(shape, dtype=None):
    fan_in = shape[0]
    fan_out = shape[1]
    # limit = np.sqrt(6. / (fan_in+fan_out))
    limit = np.sqrt(9. / (np.max([fan_in, fan_out])**2))
    return tf.random.uniform(shape, minval=0, maxval=limit, dtype=dtype)


def layer_nonneg_tanh(units: int, **kwargs) -> callable:
    kernel_initializer = kwargs.pop('kernel_initializer', None)
    return Dense(
        activation="tanh",
        kernel_constraint=constraints.non_neg(),
        kernel_initializer=constraint_xavier_p_init if kernel_initializer is None else kernel_initializer,
        units=units,
        **kwargs,
    )


def layer_nonneg_lin(units: int, **kwargs) -> callable:
    kernel_initializer = kwargs.pop('kernel_initializer', None)
    return Dense(
        activation="linear",
        kernel_constraint=constraints.non_neg(),
        kernel_initializer=constraint_xavier_p_init if kernel_initializer is None else kernel_initializer,
        units=units,
        **kwargs,
    )


def nonneg_tanh_network(size: int, dropout: float, **layer_kwargs) -> callable:
    return mlp_with_default_layer(
        size, default_layer=layer_nonneg_tanh, dropout=dropout, **layer_kwargs
    )


def tensorproduct_network(inpY, inpX, output_dim):
    # x = Concatenate()([inpX, inpY])
    # row_tensor = tf.einsum('ij,ik->jk', inpY, inpX)
    row_tensor = RowTensor()([inpY, inpX])
    return MonoMultiLayer(
        output_dim=output_dim,
        # row_tensor,
        # units=1,
        dim_bsp=inpX.shape[1] * inpY.shape[1],  # TODO: check
        trafo=mono_trafo_multi,
        trainable=True,
    )(row_tensor)


def interconnected_network(
        inpY: Input,
        inpX: Input,
        network_default: callable,
        top_layer: callable,
) -> callable:
    x = Concatenate()([inpX, inpY])
    x = network_default(x)
    x = top_layer(x)
    return x


def layer_inverse_exp(units: int, **kwargs) -> callable:
    def inner(x):
        return tf.math.exp(tf.multiply(Dense(units=units, **kwargs)(x), -0.5))

    return inner


def locscale_network(
        inpY: Input,
        inpX: Input,
        mu_top_layer: callable,
        sd_top_layer: callable,
        top_layer: callable,
) -> callable:
    loc = mu_top_layer(inpX)
    scale_inv = sd_top_layer(inpX)
    outpY = top_layer(inpY)

    return tf.subtract(tf.multiply(scale_inv, outpY), loc)


def get_neat_model(
        dim_features: int,
        net_y_size_trunk: callable,
        net_x_arch_trunk: callable,
        model_type: ModelType,
        base_distribution: tfd.Distribution,
        optimizer: Optimizer,
        **kwds,
):
    # inputs
    inpX = Input(shape=(dim_features,))
    inpY = Input(shape=(1,))

    # (intermediate) outputs
    outpX = net_x_arch_trunk(inpX)  # shape (#n, dim_features*last_default_layer)

    # outputs
    if model_type == ModelType.TP:
        outp = tensorproduct_network(net_y_size_trunk(inpY), outpX, **kwds)
    elif model_type == ModelType.LS:
        outp = locscale_network(net_y_size_trunk(inpY), outpX, **kwds)
    elif model_type == ModelType.INTER:
        outp = interconnected_network(
            inpY, outpX, network_default=net_y_size_trunk, **kwds
        )
    else:
        raise ValueError("model_type must be one of ModelType")

    model = NEATModel([inpX, inpY], outp, base_distribution=base_distribution)

    model.compile(
        loss=lambda y_true, y_pred: -1 * tfd.log_prob(base_distribution, y_pred),
        optimizer=optimizer,
    )

    return model


def fit(epochs, train_data, val_data, **params):
    neat_model = get_neat_model(dim_features=train_data[0].shape[1], **params)

    callback = EarlyStopping(
        patience=100, monitor="val_logLik", restore_best_weights=True
    )
    hist = neat_model.fit(
        x=train_data,
        y=train_data[1],
        validation_data=(val_data, val_data[1]),
        epochs=epochs,
        callbacks=[callback],
    )
    return hist, neat_model


if __name__ == "__main__":
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds:
        model_type=ModelType.LS,
        mu_top_layer=Dense(units=1),
        sd_top_layer=layer_inverse_exp(units=1),
        top_layer=layer_nonneg_lin(units=1),
    )
    neat_model.summary()
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds:
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(units=1),
    )
    neat_model.summary()

    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10], dropout=0),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds
        model_type=ModelType.TP,
        output_dim=1,
    )

    neat_model.summary()
