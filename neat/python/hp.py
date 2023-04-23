import inspect
import logging
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from fire import Fire
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
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
    seed: int = 1,
    log_file: str = "train.log",
    log_level: str = "info",
):
    if data_path is None:
        data_sets = get_dataset_names()
        for data_path in data_sets:
            run_single(data_path, fast, seed, log_file, log_level)
    else:
        run_single(data_path, fast, seed, log_file, log_level)


def run_single(
    data_path: str,
    fast: bool,
    seed: int,
    log_file: str,
    log_level: str,
) -> None:
    setup_logger(log_file, log_level)

    logging.info(f"TFP Version {tfp.__version__}")
    logging.info(f"TF  Version {tf.__version__}")

    set_seeds(seed)
    setup_folders(data_path)

    hp_space = get_hp_space()

    mlflow.autolog()
    mlflow.start_run(run_name=data_path)

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_vals = {arg: values[arg] for arg in args}
    mlflow.log_params(arg_vals)

    fit_func = get_fit_func(seed, data_path, fast, arg_vals)

    trials = Trials()
    best = fmin(
        fit_func,
        hp_space,
        algo=tpe.suggest,
        max_evals=2 if fast else 500,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    mlflow.log_params(best)
    mlflow.log_metric("best_score", min(trials.losses()))
    mlflow.log_dict(trials.trials, "trials.yaml")

    # plot_result(trials)
    mlflow.end_run()


def plot_result(trials):
    tpe_results = np.array(
        [
            [x["result"]["loss"]]
            + [h[0] if len(h) else np.nan for h in x["misc"]["vals"].values()]
            for x in trials.trials
        ]
    )
    hps = [c for c in trials.trials[0]["misc"]["vals"].keys()]
    columns = ["loss"] + hps
    tpe_results_df = pd.DataFrame(tpe_results, columns=columns)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    tpe_results_df[["loss"]].plot(ax=ax[0])
    tpe_results_df[hps].plot(ax=ax[1])
    mlflow.log_figure(fig, "hyperopt.png")


def get_fit_func(seed, data_path, fast, args) -> callable:
    data = load_data(data_path)
    train_data = (data["x_train"], data["y_train"])
    val_data = (data["x_test"], data["y_test"])
    experiment_id = mlflow.set_experiment(f"{data_path}_runs")

    def fit_func(params):
        mlflow.start_run(
            experiment_id=experiment_id._experiment_id,
            nested=mlflow.active_run() is not None,
        )
        mlflow.log_params(args)
        mlflow.log_params(
            dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
        )

        model_type = params["model_type"]
        model_kwargs = get_model_kwargs(model_type)
        params = {**params, **model_kwargs}

        hist, neat_model = fit(
            seed=seed,
            epochs=10 if fast else 10_000,
            train_data=train_data,
            val_data=val_data,
            **params,
        )
        loss = min(hist.history["val_logLik"])
        status = STATUS_OK
        if np.isnan(loss).any():
            status = STATUS_FAIL
        mlflow.end_run("FINISHED" if status == STATUS_OK else "FAILED")
        return {"loss": loss, "status": status}

    return fit_func


def get_hp_space():
    dropout_vals = [0, 0.1]
    x_unit_vals = [20, 50, 100]
    x_layer_vals = [1, 2]
    y_base_unit_vals = [20, 50, 100]
    y_top_unit_vals = [5, 10, 20]
    learning_rates = [1e-2, 1e-3, 1e-4]

    space = dict(
        net_x_arch_trunk=hp.choice(
            "net_x_arch_trunk",
            [
                relu_network(
                    [x_units] * x_layers,
                    dropout=dropout,
                )
                for x_units, x_layers, dropout in product(
                    x_unit_vals, x_layer_vals, dropout_vals
                )
            ],
        ),
        net_y_size_trunk=hp.choice(
            "net_y_size_trunk",
            [
                nonneg_tanh_network(
                    (y_base_units, y_base_units, y_top_units),
                    dropout=dropout,
                )
                for y_base_units, y_top_units, dropout in product(
                    y_base_unit_vals, y_top_unit_vals, dropout_vals
                )
            ],
        ),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=hp.choice("optimizer", [Adam(lr) for lr in learning_rates]),
        model_type=hp.choice("model_type", [ModelType.LS, ModelType.INTER]),
    )
    return space


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
