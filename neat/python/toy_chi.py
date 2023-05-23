import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from tensorflow_probability import distributions as tfd
from scipy.stats import probplot
from utils import (
    nonneg_tanh_network,
    get_neat_model,
    ModelType,
    feature_specific_network,
    layer_nonneg_lin,
    layer_inverse_exp,
)

# Data simulation
n = 10000
p = 1
y = np.random.chisquare(df=4, size=n).reshape(-1, 1)
X = np.random.normal(size=(n, p))
model = ModelType.LS

# NEAT
if model == ModelType.LS:

    neat_model = get_neat_model(
        dim_features=p,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([5, 5], dropout=0),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=optimizers.Adam(learning_rate=0.0001),
        # kwds:
        model_type=ModelType.LS,
        mu_top_layer=Dense(units=1),
        sd_top_layer=layer_inverse_exp(units=1),
        top_layer=layer_nonneg_lin(units=1),
    )

elif model == ModelType.INTER:
    neat_model = get_neat_model(
        dim_features=p,
        net_x_arch_trunk=feature_specific_network(
            size=(64, 64, 32),
            default_layer=lambda **kwargs: Dense(activation="relu", **kwargs),
            dropout=0,
        ),
        net_y_size_trunk=nonneg_tanh_network([5, 5], dropout=0),
        base_distribution=tfd.Normal(loc=0, scale=1),
        optimizer=optimizers.Adam(learning_rate=0.0001),
        # kwds:
        model_type=ModelType.INTER,
        top_layer=layer_nonneg_lin(units=1),
    )
else:
    raise NotImplementedError

neat_model.summary()


callback = EarlyStopping(patience=100, monitor="val_logLik", restore_best_weights=True)

neat_model.fit(
    x=(X, y),
    y=y,
    batch_size=400,
    epochs=500,
    validation_split=0.1,
    verbose=True,
    callbacks=[callback],
)

pred_neat = neat_model.predict((X, y))

# Plotting
import matplotlib.pyplot as plt

plt.scatter(pred_neat, y)
plt.xlabel("pred_neat")
plt.ylabel("y")
plt.show()

probplot(pred_neat.flatten(), plot=plt)
plt.show()
