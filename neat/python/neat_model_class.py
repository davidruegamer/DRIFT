from tensorflow_probability import distributions as tfd
import tensorflow as tf


class NEATModel(tf.keras.Model):
    def __init__(self, base_distribution=tfd.Normal(loc=0, scale=1), **kwargs):
        super().__init__(**kwargs)
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
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)

            # Watch x and y
            tape.watch(x)
            # tape.watch(y)

            # Feed forward
            h = self(x, training=True)

            # Gradient and the corresponding loss function
            h_prime = tape.gradient(h, y)
            loss_value = self.loss_fn_unnorm(y, h)
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
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)

            # Watch x and y
            tape.watch(x)
            tape.watch(y)

            # Feed forward
            h = self(x, training=False)

            # Gradient and the corresponding loss function
            h_prime = tape.gradient(h, y)

            loss_value = self.loss_fn_unnorm(y, h)
            logLik = tf.reduce_sum(
                tf.subtract(
                    loss_value,
                    tf.math.log(tf.clip_by_value(h_prime, 1e-8, tf.float32.max)),
                )
            )

        return {"logLik": logLik}
