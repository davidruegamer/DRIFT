import tensorflow as tf
from tensorflow_probability import distributions as tfd


class NEATModel(tf.keras.Model):
    def __init__(self, *args, base_distribution=tfd.Normal(loc=0, scale=1), **kwargs):
        super().__init__(*args, **kwargs)
        self.base_distribution = base_distribution

    def loss_fn_unnorm(self, y_true, y_pred):
        return -self.base_distribution.log_prob(y_pred)

    @tf.function
    def train_step(self, data):
        # Compute gradients
        trainable_vars = self.trainable_variables

        # Exact LL part
        with tf.GradientTape(persistent=True) as tape:
            x, y = data

            # Create tensor that you will watch
            x = list(map(lambda xx: tf.convert_to_tensor(xx, dtype=tf.float32), x))

            # Watch x and y
            tape.watch(x)
            # tape.watch(y)

            # Feed forward
            h = self(x, training=True)

            # Gradient and the corresponding loss function
            h_prime = tape.gradient(h, x[1])
            loss_value = self.loss_fn_unnorm(x[1], h)
            logLik = tf.reduce_sum(
                tf.subtract(
                    loss_value,
                    tf.math.log(tf.clip_by_value(h_prime, 1e-8, tf.float32.max)),
                )
            )
            gradients = tape.gradient(logLik, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a named list mapping metric names to current value
        return {"logLik": logLik}

    @tf.function
    def test_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            x, y = data

            # Create tensor that you will watch
            x = list(map(lambda xx: tf.convert_to_tensor(xx, dtype=tf.float32), x))

            tape.watch(x)

            # Feed forward
            h = self(x, training=False)

            # Gradient and the corresponding loss function
            h_prime = tape.gradient(h, x[1])

            loss_value = self.loss_fn_unnorm(x[1], h)
            logLik = tf.reduce_sum(
                tf.subtract(
                    loss_value,
                    tf.math.log(tf.clip_by_value(h_prime, 1e-8, tf.float32.max)),
                )
            )

        return {"logLik": logLik}
