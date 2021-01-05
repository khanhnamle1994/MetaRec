# Import libraries
import tensorflow as tf

# Import helper functions
import inner as il
from networks import OuterConstantNetwork, OuterLinearNetwork, InnerVAE


class MetaVAE:
    def __init__(self, num_inner_loops=5, first_order=False, adjust_loss=False):
        il.InnerVariable.counter = 0
        self.num_inner_loops = num_inner_loops
        self.first_order = first_order
        self.adjust_loss = adjust_loss

        # Zero variables
        self.input_shape = None
        self.inner_vae = None
        self.outer_network = None
        self.inner_vars = None
        self.trainable_inner_vars = None

    def _build(self, input_shape):
        assert self.input_shape is None or self.input_shape == input_shape

        if self.input_shape == input_shape:
            return

        self.input_shape = input_shape

        self.inner_vae = InnerVAE(int(input_shape[-1]))

        # Warmup the inner network so the layers are built and we can collect the inner variables needed for the
        # outer network.
        il.warmup_inner_layer(self.inner_vae, input_shape)

        self.inner_vars = il.get_inner_variables(self.inner_vae)
        print("Found inner vars:", self.inner_vars)

        # Collect mutable inner variables from inner network.
        # If the outer network outputs the inner variable per-step then we can not mutate it.
        self.trainable_inner_vars = il.get_trainable_inner_variables(self.inner_vae)

        self.outer_network = OuterConstantNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)
        # self.outer_network = OuterLinearNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)
        print("Num inner trainable vars:", self.outer_network.output_size)

    def get_loss(self, train_inputs, test_inputs):
        assert train_inputs.shape[2:] == test_inputs.shape[2:]
        self._build(train_inputs.shape[2:])

        def _image_summary(name, images, index=0):
            tf.summary.image(name, tf.cast(255 * images[:, index], tf.uint8))

        def _avg_scalar_summary(name, values):
            tf.summary.scalar(name, tf.reduce_mean(values))

        def _loss_dict_summary(suffix, loss_dict):
            _avg_scalar_summary("%s_%s" % ("loss", suffix), loss_dict["loss"])
            _avg_scalar_summary("%s_%s" % ("loss_bce", suffix), loss_dict["bce"])
            _avg_scalar_summary("%s_%s" % ("loss_kld", suffix), loss_dict["kld"])
            _image_summary("%s_%s" % ("reconstruction", suffix), loss_dict["reconstruction"])

        # Keep initial and step test losses and average them in the end.
        train_losses = []
        test_losses = []
        adjusted_losses = []

        # Calculate initial weights using outer network
        self.outer_network.calculate_output(train_inputs)

        # Collect initial values for mutable inner variables from outer network.
        mutable_inner_vars = {
            inner_var: self.outer_network.get_inner_variable(inner_var, step=0)
            for inner_var in self.trainable_inner_vars
        }

        # Create the inner variable getter method.
        # Must make sure that the step variable exists (created in loop below).
        def _get_inner_var(inner_var, batch_index, step):
            if inner_var.per_step:
                # print("Getting", inner_var.name, "at step", step)
                return self.outer_network.get_inner_variable(inner_var, step)[batch_index]
            else:
                return mutable_inner_vars[inner_var][batch_index]

        # Assign getters to the function above for the inner variables
        for inner_var in self.inner_vars:
            inner_var.getter = _get_inner_var

        # Test image summary
        _image_summary("test_input", test_inputs)
        _image_summary("train_input", train_inputs)
        tf.summary.image("test_input_mean", tf.cast(255 * tf.reduce_mean(test_inputs, axis=1), tf.uint8))
        tf.summary.image("train_input_mean", tf.cast(255 * tf.reduce_mean(train_inputs, axis=1), tf.uint8))

        # previous_gradients = {}

        for step in range(self.num_inner_loops):
            print("------ Starting step", step)

            # Calculate train loss for this step
            il.set_inner_train_state(self.inner_vae, is_train=True)
            il.set_inner_step(self.inner_vae, step)
            loss_dict = self.inner_vae.get_loss(train_inputs)
            train_losses.append(loss_dict["loss"])
            _loss_dict_summary("train_step_%d" % step, loss_dict)

            # Calculate test loss for this step
            il.set_inner_train_state(self.inner_vae, is_train=False)
            test_loss_dict = self.inner_vae.get_loss(test_inputs)
            test_losses.append(test_loss_dict["loss"])
            _loss_dict_summary("test_step_%d" % step, test_loss_dict)

            if self.adjust_loss:
                adjusted_losses.append(tf.maximum(loss_dict["loss"], test_loss_dict["loss"]))
                _avg_scalar_summary("loss_adjusted_step_%d" % step, adjusted_losses[-1])

            # Mutable inner variable gradient update using train loss for this step
            mutable_inner_vars_keys, mutable_inner_vars_values = list(mutable_inner_vars.keys()), list(
                mutable_inner_vars.values())
            mutable_inner_vars_grads = tf.gradients(loss_dict["loss"], mutable_inner_vars_values)
            for inner_var, weights, grads in zip(mutable_inner_vars_keys, mutable_inner_vars_values,
                                                 mutable_inner_vars_grads):
                lr = self.outer_network.get_learning_rate(inner_var, step)
                assert not inner_var.per_step
                if grads is not None:
                    # print("-- Weights:", weights)
                    # print("Grads:", grads)
                    # print("Lr:", lr)
                    # Prevent second order derivatives for first order training
                    if self.first_order:
                        grads = tf.stop_gradient(grads)

                    # Momentum
                    # if inner_var in previous_gradients:
                    #    grads = 0.9 * previous_gradients[inner_var] + grads

                    # previous_gradients[inner_var] = grads

                    mutable_inner_vars[inner_var] = weights - lr * (
                        tf.stop_gradient(grads) if self.first_order else grads)
                else:
                    raise Exception(
                        "Grads none for %s (tensor: %s) (unused inner variable?)" % (inner_var.name, weights))

        # Calculate final test loss
        print("------- Final evaluation step")
        il.set_inner_train_state(self.inner_vae, is_train=True)
        il.set_inner_step(self.inner_vae, self.num_inner_loops)
        loss_dict = self.inner_vae.get_loss(train_inputs)  # Run on train set to get batch statistics
        train_losses.append(loss_dict["loss"])
        _loss_dict_summary("train_final", loss_dict)
        il.set_inner_train_state(self.inner_vae, is_train=False)
        test_loss_dict = self.inner_vae.get_loss(test_inputs)
        test_losses.append(test_loss_dict["loss"])
        _loss_dict_summary("test_final", test_loss_dict)

        if self.adjust_loss:
            adjusted_losses.append(tf.maximum(loss_dict["loss"], test_loss_dict["loss"]))
            _avg_scalar_summary("loss_adjusted_final", adjusted_losses[-1])

        test_losses = tf.transpose(tf.convert_to_tensor(test_losses))
        train_losses = tf.transpose(tf.convert_to_tensor(train_losses))
        print("Test losses:", test_losses)

        assert train_losses.shape[1] == test_losses.shape[1]

        # Start by uniformly adding losses, later only focus on last loss
        loss_weights = tf.linspace(0.0, 1.0, test_losses.shape[1])
        loss_weights = tf.expand_dims(loss_weights, 0)
        loss_temp = tf.cast(tf.train.get_global_step(), tf.float32) / 1000.
        loss_weights *= loss_temp

        test_loss_total = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(loss_weights) * test_losses, axis=-1))
        _avg_scalar_summary("test_loss_total", test_loss_total)

        train_loss_total = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(loss_weights) * train_losses, axis=-1))
        _avg_scalar_summary("train_loss_total", train_loss_total)

        if self.adjust_loss:
            adjusted_losses = tf.transpose(tf.convert_to_tensor(adjusted_losses))
            adjusted_loss_total = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(loss_weights) * adjusted_losses, axis=-1))
            _avg_scalar_summary("adjusted_loss_total", adjusted_loss_total)

        # Sample summaries
        latents = self.inner_vae.sample_normal(tf.zeros_like(test_loss_dict["latents"]),
                                               tf.ones_like(test_loss_dict["latents"]))
        samples = tf.nn.sigmoid(self.inner_vae.decode(latents))
        for i in range(test_inputs.shape[1]):
            _image_summary("random_samples_%d" % i, samples, index=i)

        print("Latents shape:", latents.shape)

        return adjusted_loss_total if self.adjust_loss else test_loss_total