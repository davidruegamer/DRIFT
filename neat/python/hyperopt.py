#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : hyperopt.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-05-10 17:59:17 (Marcel Arpogaus)
# changed : 2023-04-17 14:20:19 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

from .utils import (
    ModelType,
    feature_specific_network,
    flow,
    hist,
    layer_inverse_exp,
    layer_nonneg_tanh,
    nonneg_tanh_network,
)

if __name__ == "__main__":
    experiment_name = "hp_bimodal"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--10sec", help="run faster", action="store_true", dest="_10sec"
    )
    parser.add_argument(
        "--no-mlflow",
        help="disable mlfow tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument("--seed", help="random seed", default=1, type=int)
    parser.add_argument(
        "--log-file",
        default="train.log",
        type=argparse.FileType("a"),
        help="Path to logfile",
    )
    parser.add_argument("--log-level", default="info", help="Provide logging level.")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log_level.upper(),
        # format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(args.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("TFP Version", tfp.__version__)
    logging.info("TF  Version", tf.__version__)

    # Ensure Reproducibility
    logging.info(f"setzing random seed to {args.seed}")
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    metrics_path = os.path.join("metrics", experiment_name)
    artifacts_path = os.path.join("artifacts", experiment_name)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    common_kwds = dict(
        dim_features=3,
        net_x_arch_trunk=hp.choice(
            "net_x_arch_trunk",
            [
                feature_specific_network(
                    size=s,
                    default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
                )
                for s in [(64, 64, 32), (16, 16, 32)]
            ],
        ),
        net_y_size_trunk=hp.choice(
            "net_y_size_trunk",
            [nonneg_tanh_network(s) for s in [(50, 50, 10), (25, 25, 10)]],
        ),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
    )
    space = hp.choice(
        "params",
        [
            dict(
                **common_kwds,
                model_type=ModelType.LS,
                mu_top_layer=Dense(units=1),
                sd_top_layer=layer_inverse_exp(units=1),
                top_layer=layer_nonneg_tanh(units=1),
            ),
            dict(
                **common_kwds,
                model_type=ModelType.INTER,
                network_default=nonneg_tanh_network([50, 50, 10]),
                top_layer=layer_nonneg_tanh(units=1),
            ),
            dict(
                **common_kwds,
                model_type=ModelType.TP,
                output_dim=1,
            ),
        ],
    )

    mlflow.autolog()
    experiment_id = mlflow.set_experiment(experiment_name)
    if os.environ.get("MLFLOW_RUN_ID", False):
        mlflow.start_run()
    else:
        mlflow.start_run(run_name="hypeopt")

    def F(params):
        if args._10sec:
            params["fit_kwds"].update({"epochs": 1})
            params["data_points"] = 25

        mlflow.start_run(
            experiment_id=experiment_id, nested=mlflow.active_run() is not None
        )
        mlflow.log_params(dict(args))
        mlflow.log_params(
            dict(filter(lambda kw: not isinstance(kw[1], dict), params.items()))
        )
        mlflow.log_params(params["fit_kwds"])

        mlflow.log_artifacts(artifacts_path)

        loss = min(hist.history["val_loss"])
        status = STATUS_OK
        if np.isnan(loss).any() or np.isnan(flow.mean().numpy()).any():
            status = STATUS_FAIL
        mlflow.end_run("FINISHED" if status == STATUS_OK else "FAILED")
        return {"loss": loss, "status": status}

    trials = Trials()
    best = fmin(
        F,
        space,
        algo=tpe.suggest,
        max_evals=2 if args._10sec else 50,
        trials=trials,
        rstate=np.random.RandomState(args.seed),
    )
    mlflow.log_params(best)
    mlflow.log_metric("best_score", min(trials.losses()))
    mlflow.log_dict(trials.trials, "trials.yaml")

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
    mlflow.end_run()
