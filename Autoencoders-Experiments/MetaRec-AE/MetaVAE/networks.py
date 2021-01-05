# Import libraries
import tensorflow as tf

# Import helper functions
import inner as il
from outer import OuterNetwork


class OuterConstantNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops, fixed_lr=fixed_lr)
        self.constant_init = tf.get_variable("inner_init", (self.output_size,), dtype=tf.float32, trainable=True)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.output = tf.tile(tf.expand_dims(self.constant_init, 0), (batch_size, 1))


class OuterLinearNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops)
        self.dense = tf.keras.layers.Dense(self.output_size)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (batch_size, -1))
        self.output = self.dense(inputs)


class InnerVAEEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.down_convs = tf.keras.models.Sequential()
        size = 32
        while size > 2:
            # self.down_convs.add(il.InnerMemorization())
            # self.down_convs.add(il.InnerConv2D(32, 3, (1, 1), padding="SAME", use_bias=False))
            # self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))

            self.down_convs.add(il.InnerConv2D(32, 3, (2, 2), padding="SAME", use_bias=False))
            # self.down_convs.add(il.InnerNormalization())
            self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))
            size //= 2

        assert size == 2

        # self.down_convs.add(il.InnerConv2D(128, 3, (1, 1), padding="SAME", use_bias=True))
        # self.down_convs.add(il.InnerConv2D(64, 2, (1, 1), padding="VALID", use_bias=True))
        # self.down_convs.add(il.InnerNormalization())
        # self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))

        self.down_convs.add(il.InnerReshape((128,)))
        self.down_convs.add(il.InnerDense(256))

        self.layers = [self.down_convs]

    def call(self, inputs):
        output = self.down_convs(inputs)
        half_output = output.shape[-1] // 2
        mean = output[:, :, :half_output]
        logvar = output[:, :, half_output:]
        return mean, logvar


class InnerVAEDecoder(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()

        self.layers = []

        self.up_convs = tf.keras.models.Sequential()

        self.up_convs.add(il.InnerDense(128, use_bias=False))
        self.up_convs.add(il.InnerReshape((2, 2, 32)))
        self.up_convs.add(il.InnerNormalization())
        self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))

        size = 2
        while size < 32:
            # self.up_convs.add(il.InnerMemorization())
            # self.up_convs.add(il.InnerConv2D(32, min(size, 3), (1, 1), padding="SAME", use_bias=False))
            # self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))

            self.up_convs.add(il.InnerConv2D(32, min(size, 3), (1, 1), padding="SAME", use_bias=False))
            self.up_convs.add(il.InnerNormalization())
            self.up_convs.add(il.InnerResize((size * 2, size * 2)))
            self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))
            size *= 2

        # Final refinement
        self.up_convs.add(il.InnerConv2D(output_channels, 3, (1, 1), padding="SAME", use_bias=True))
        # self.up_convs.add(il.InnerMemorization())

        self.layers = [self.up_convs]

    def call(self, latents):
        output = self.up_convs(latents)
        return output


class InnerVAE(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()

        self.encoder = InnerVAEEncoder()
        self.decoder = InnerVAEDecoder(output_channels=output_channels)
        self.layers = [self.encoder, self.decoder]

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, latents):
        return self.decoder(latents)

    def sample_normal(self, mean, logvar):
        return mean + tf.math.exp(0.5 * logvar) * tf.random.normal(logvar.shape)

    def get_loss(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        reconstr = self.decode(latents)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=reconstr), axis=[1, 2, 3, 4])
        # bce = tf.reduce_mean(tf.abs(inputs - tf.nn.sigmoid(reconstr)), axis=list(range(1, len(inputs.shape))))
        kld = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=list(range(1, len(mean.shape))))
        return {"loss": kld + bce,
                "latents": latents,
                "reconstruction": tf.nn.sigmoid(reconstr),
                "bce": bce,
                "kld": kld}

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        return tf.nn.sigmoid(self.decode(latents))
