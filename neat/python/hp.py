#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : hyperopt.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-05-10 17:59:17 (Marcel Arpogaus)
# changed : 2023-04-18 14:43:04 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
import inspect
import logging
import os
import sys

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
    fast: bool = True,
    seed: int = 1,
    log_file: str = "train.log",
    log_level: str = "info",
):
    experiment_name = "neat"

    setup_logger(log_file, log_level)

    logging.info(f"TFP Version {tfp.__version__}")
    logging.info(f"TF  Version {tf.__version__}")

    set_seeds(seed)
    setup_folders(experiment_name)

    common_kwds = get_common_kwds()
    space = get_search_space(common_kwds, fast)

    mlflow.autolog()
    mlflow.start_run(run_name="hypeopt")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arg_vals = {arg: values[arg] for arg in args}

    fit_func = get_fit_func(experiment_name, seed, data_path, fast, arg_vals)

    trials = Trials()
    best = fmin(
        fit_func,
        space,
        algo=tpe.suggest,
        max_evals=2 if fast else 50,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    # mlflow.log_params(best)
    # mlflow.log_metric("best_score", min(trials.losses()))
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


def get_fit_func(experiment_name, seed, data_path, fast, args) -> callable:
    data = load_data(data_path)
    train_data = (data["x_train"], data["y_train"])
    val_data = (data["x_test"], data["y_test"])
    experiment_id = mlflow.set_experiment(experiment_name)

    def fit_func(params):
        mlflow.start_run(
            experiment_id=experiment_id._experiment_id,
            nested=mlflow.active_run() is not None,
        )
        mlflow.log_params(args)
        mlflow.log_params(
            dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
        )

        hist, neat_model = fit(
            seed=seed,
            epochs=10 if fast else 10_000,
            train_data=train_data,
            val_data=val_data,
            **params,
        )
        loss = min(hist.history["val_logLik"])
        mlflow.log_metric("loss", loss)
        status = STATUS_OK
        if np.isnan(loss).any():
            status = STATUS_FAIL
        mlflow.end_run("FINISHED" if status == STATUS_OK else "FAILED")
        return {"loss": loss, "status": status}

    return fit_func


def get_common_kwds():
    common_kwds = dict(
        net_x_arch_trunk=relu_network(
            (100, 100)
        ),  # units(20, 100), layers(1,2), dropout(0, 0.1)
        net_y_size_trunk=hp.choice(
            "net_y_size_trunk",
            [nonneg_tanh_network(s) for s in [(50, 50, 10), (25, 25, 10)]],
        ),
        base_distribution=tfd.Normal(loc=0, scale=1),  # keep fixed
        optimizer=Adam(),  # learning rate
    )
    return common_kwds


# {25, 50, 100} * 2, {5, 10, 20}  # dropout(0, 0.1)?


def get_search_space(common_kwds, fast):
    space = hp.choice(
        "params",
        [
            dict(
                **common_kwds,
                model_type=ModelType.LS,
                mu_top_layer=Dense(units=1),
                sd_top_layer=layer_inverse_exp(units=1),
                top_layer=layer_nonneg_lin(units=1),
            ),
            dict(
                **common_kwds,
                model_type=ModelType.INTER,
                top_layer=layer_nonneg_lin(units=1),
            ),
            # dict(
            #     **common_kwds,
            #     model_type=ModelType.TP,
            #     output_dim=1,
            # ),
        ],
    )
    return space


def setup_logger(log_file: str, log_level: str) -> None:
    # Configure logging
    logging.basicConfig(
        level=log_level.upper(),
        # format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


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


if __name__ == "__main__":
    Fire(run)
