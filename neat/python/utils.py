from enum import Enum
from typing import Iterable

import tensorflow as tf
from keras import constraints, initializers
from keras.layers import Activation, Concatenate, Dense, Input
from keras.optimizers import Adam, Optimizer
from tensorflow_probability import distributions as tfd

from monolayers import MonoMultiLayer, mono_trafo_multi
from neat_model_class import NEATModel


class ModelType(Enum):
    TP = "tp"
    LS = "ls"
    INTER = "inter"


def mlp_with_default_layer(size: Iterable[int], default_layer: callable) -> callable:
    def inner(inp):
        x = default_layer(units=size[0])(inp)
        for i in range(1, len(size)):
            x = default_layer(units=size[i])(x)
        return x

    return inner


def relu_network(size: Iterable[int]) -> callable:
    return mlp_with_default_layer(
        size, default_layer=lambda **kwargs: Dense(activation="relu", **kwargs)
    )


def feature_specific_network(
    size: Iterable[int],
    default_layer: callable,
) -> callable:
    def inner(x):
        return Concatenate(axis=1)(
            [
                mlp_with_default_layer(size, default_layer)(xx)
                for xx in tf.split(x, num_or_size_splits=x.shape[1], axis=1)
            ]
        )

    return inner


def layer_nonneg_tanh(units: int, **kwargs) -> callable:
    return Dense(
        activation="tanh",
        kernel_constraint=constraints.non_neg(),
        kernel_initializer=initializers.RandomUniform(minval=0, maxval=1),
        units=units,
        **kwargs,
    )


def nonneg_tanh_network(size: int) -> callable:
    return mlp_with_default_layer(size, default_layer=layer_nonneg_tanh)


def tensorproduct_network(inpY, inpX, output_dim):
    x = Concatenate()([inpX, inpY])
    return MonoMultiLayer(
        output_dim=output_dim,
        # row_tensor,
        # units=1,
        dim_bsp=inpX.shape[1] * inpY.shape[1],  # TODO: check
        trafo=mono_trafo_multi,
        trainable=True,
    )(x)


def interconnected_network(
    inpY: Input,
    inpX: Input,
    network_default: callable,
    top_layer: callable,
) -> callable:
    x = Concatenate()([inpX, inpY])
    x = network_default(x)
    x = top_layer(x)
    x = Activation("softplus")(x)
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
    outpY = Activation("softplus")(outpY)

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
    outpX = net_x_arch_trunk(inpX)

    # outputs
    if model_type == ModelType.TP:
        outp = tensorproduct_network(net_y_size_trunk(inpY), outpX, **kwds)
    elif model_type == ModelType.LS:
        outp = locscale_network(net_y_size_trunk(inpY), outpX, **kwds)
    elif model_type == ModelType.INTER:
        outp = interconnected_network(inpY, outpX, **kwds)
    else:
        raise ValueError("model_type must be one of ModelType")

    model = NEATModel([inpX, inpY], outp, base_distribution=base_distribution)

    model.compile(
        loss=lambda y_true, y_pred: -1 * tfd.log_prob(base_distribution, y_pred),
        optimizer=optimizer,
    )

    return model


if __name__ == "__main__":
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10]),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds:
        model_type=ModelType.LS,
        mu_top_layer=Dense(units=1),
        sd_top_layer=layer_inverse_exp(units=1),
        top_layer=layer_nonneg_tanh(units=1),
    )
    neat_model.summary()
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10]),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds:
        model_type=ModelType.INTER,
        network_default=nonneg_tanh_network([50, 50, 10]),
        top_layer=layer_nonneg_tanh(units=1),
    )
    neat_model.summary()
    neat_model = get_neat_model(
        dim_features=3,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
        ),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10]),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds
        model_type=ModelType.TP,
        output_dim=1,
    )

    neat_model.summary()
