import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd

from utils import get_neat_model, nonneg_tanh_network, ModelType, \
    layer_inverse_exp, layer_nonneg_lin, relu_network


def run_toy_example():
    X, y = get_toy_data()
    run_ls(X, y)


def run_ls(X, y):
    # Model types comparison
    mod = get_neat_model(
        dim_features=X.shape[1],
        net_x_arch_trunk=relu_network((100, 100)),
        net_y_size_trunk=nonneg_tanh_network([50, 50, 10]),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=Adam(),
        # kwds:
        model_type=ModelType.LS,
        mu_top_layer=Dense(units=1),
        sd_top_layer=layer_inverse_exp(units=1),
        top_layer=layer_nonneg_lin(units=1),
    )

    callback = EarlyStopping(patience=5, monitor='val_logLik', restore_best_weights=True)

    mod.fit(x=[X, y], y=y, batch_size=32, epochs=25, validation_split=0.1, callbacks=[callback],
            verbose=1)
    pred = mod.predict([X, y])
    logLik = -mod.evaluate([X, y], y) / X.shape[0]

    P = pred.reshape((11, -1))
    for i in range(P.shape[1]):
        plt.plot(P[:, i], '-')
    plt.show()


def get_toy_data():
    # Data imported from R
    X = np.loadtxt('../tests/toy_data_X.csv', delimiter=',')
    y = np.loadtxt('../tests/toy_data_y.csv', delimiter=',').reshape(-1, 1)
    return X, y


if __name__ == '__main__':
    run_toy_example()