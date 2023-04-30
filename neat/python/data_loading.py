import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

root_path = Path(os.path.realpath(__file__)).parent.parent.parent.absolute()


def load_data(dataset_name: str, seed: int = 1) -> dict:
    data = np.loadtxt(
        os.path.join(root_path, "benchmark", "benchmark_data", dataset_name + ".data")
    )

    x, y = data[:, :-1], data[:, -1]
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.1, random_state=seed)

    scaler_X = StandardScaler().fit(x_t)
    scaler_y = StandardScaler().fit(y_t.reshape(-1, 1))
    X_train = scaler_X.transform(x_t)
    y_train = scaler_y.transform(y_t.reshape(-1, 1))
    X_test = scaler_X.transform(x_v)
    y_test = scaler_y.transform(y_v.reshape(-1, 1))

    params_x = {"mean": scaler_X.mean_, "var": scaler_X.var_}
    params_y = {"mean": scaler_y.mean_, "var": scaler_y.var_}

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "params_x": params_x,
        "params_y": params_y,
        "seed": seed,
    }
