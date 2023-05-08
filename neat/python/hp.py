import inspect
import logging
import os
import sys
from itertools import product
import dask.bag as db

import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from fire import Fire
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd

from data_loading import load_data
from utils import (
    ModelType,
    fit,
    layer_inverse_exp,
    nonneg_tanh_network,
    layer_nonneg_lin,
    relu_network,
)


def run(
    data_path: str = None,
    fast: bool = False,
    log_file: str = "train.log",
    log_level: str = "info",
):
    if data_path is None:
        data_sets = get_dataset_names()
        for data_path in data_sets:
            run_single(data_path, fast, log_file, log_level)
    else:
        run_single(data_path, fast, log_file, log_level)


def run_single(
    data_path: str,
    fast: bool,
    log_file: str,
    log_level: str,
) -> None:
    setup_logger(log_file, log_level)

    logging.info(f"TFP Version {tfp.__version__}")
    logging.info(f"TF  Version {tf.__version__}")

    setup_folders(data_path)

    hp_space = get_hp_space()
    hp_space = hp_space[:10] if fast else hp_space
    logging.info(f"Size of search space: {len(hp_space)}")

    mlflow.autolog()
    experiment_id = mlflow.set_experiment(f"{data_path}_runs")


    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_vals = {arg: values[arg] for arg in args}

    fit_args = [
        (params, data_path, experiment_id.experiment_id, arg_vals, fast) for params in hp_space
    ]

    # parallelize using dask instead of starmap
    b = db.from_sequence(fit_args, partition_size=1 if fast else 10)
    b.starmap(fit_func).compute(scheduler='processes', num_workers=os.cpu_count())


def log_fit_params(args, params):
    mlflow.log_params(args)
    mlflow.log_params(
        dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
    )

    mlflow.log_param("x_units", params["net_x_arch_trunk_args"]["x_units"])
    mlflow.log_param("x_layers", params["net_x_arch_trunk_args"]["x_layers"])
    mlflow.log_param("x_dropout", params["net_x_arch_trunk_args"]["dropout"])
    mlflow.log_param("y_top_units", params["net_y_size_trunk_args"]["y_top_units"])
    mlflow.log_param("y_base_units", params["net_y_size_trunk_args"]["y_base_units"])
    mlflow.log_param("y_dropout", params["net_y_size_trunk_args"]["dropout"])
    mlflow.log_param("learning_rate", params["optimizer"].learning_rate.numpy())


def fit_func(params, data_path, experiment_id, args, fast):

    data = load_data(data_path)
    train_data = (data["x_train"], data["y_train"])
    val_data = (data["x_test"], data["y_test"])

    mlflow.start_run(experiment_id=experiment_id)

    log_fit_params(args, params)

    model_type = params["model_type"]
    model_kwargs = get_model_kwargs(model_type)
    params = {**params, **model_kwargs}

    x_args = params.pop("net_x_arch_trunk_args")
    params["net_x_arch_trunk"] = relu_network(
        [x_args["x_units"]] * x_args["x_layers"], dropout=x_args["dropout"]
    )
    y_args = params.pop("net_y_size_trunk_args")
    params["net_y_size_trunk"] = nonneg_tanh_network(
        (y_args["y_base_units"], y_args["y_base_units"], y_args["y_top_units"]),
        dropout=y_args["dropout"],
    )

    seed = params.pop("seed")
    set_seeds(seed)

    hist, neat_model = fit(
        epochs=20 if fast else 10_000,
        train_data=train_data,
        val_data=val_data,
        **params,
    )

    mlflow.log_metric("val_logLik", neat_model.evaluate(x=train_data, y=train_data[1]))
    mlflow.log_metric("train_logLik", neat_model.evaluate(x=val_data, y=val_data[1]))

    mlflow.end_run()


def get_hp_space() -> list[dict]:
    seed = [1, 2, 3]
    dropout = [0, 0.1]
    x_unit = [20, 50, 100]
    x_layer = [1, 2]
    y_base_unit = [5, 10, 20]
    y_top_unit = [5, 10, 20]
    learning_rates = [1e-2, 1e-3, 1e-4]
    model = [ModelType.LS, ModelType.INTER]

    args = []
    for i, (s, d, x_u, x_l, y_b_u, y_t_u, lr, m) in enumerate(product(
        seed, dropout, x_unit, x_layer, y_base_unit, y_top_unit, learning_rates, model
    )):

        args.append({
            "seed": s,
            "net_x_arch_trunk_args": {
                "x_units": x_u,
                "x_layers": x_l,
                "dropout": d,
            },
            "net_y_size_trunk_args": {
                "y_base_units": y_b_u,
                "y_top_units": y_t_u,
                "dropout": d,
            },
            "optimizer": Adam(learning_rate=lr),
            "base_distribution": tfd.Normal(loc=0, scale=1),
            "model_type": m,
        })
    return args



def setup_logger(log_file: str, log_level: str) -> None:
    # Configure logging
    logging.basicConfig(
        level=log_level.upper(),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_model_kwargs(model_type: ModelType):
    model_kwargs = {
        ModelType.LS: dict(
            mu_top_layer=Dense(units=1),
            sd_top_layer=layer_inverse_exp(units=1),
            top_layer=layer_nonneg_lin(units=1),
        ),
        ModelType.INTER: dict(
            top_layer=layer_nonneg_lin(units=1),
        ),
    }
    return model_kwargs[model_type]


def set_seeds(seed: int) -> None:
    # Ensure Reproducibility
    logging.info(f"setting random seed to {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_folders(experiment_name: str) -> None:
    metrics_path = os.path.join("metrics", experiment_name)
    artifacts_path = os.path.join("artifacts", experiment_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)


def get_dataset_names() -> list[str]:
    return [
        "airfoil",
        "boston",
        "concrete",
        "diabetes",
        "energy",
        "fish",
        "forest_fire",
        "ltfsid",
        "real",
        "yacht",
    ]


if __name__ == "__main__":
    Fire(run)
