# Import libraries
import tensorflow as tf
import numpy as np


def get_inner_variables(layer, match_fn=None):
    """
    Get the inner variables
    :param layer: current layer
    :param match_fn: matching function
    :return: inner variables as a list
    """
    inner_variables = []
    if isinstance(layer, InnerLayer):
        for inner_variable in layer.inner_variables.values():
            if match_fn is None or match_fn(inner_variable):
                inner_variables.append(inner_variable)

    if hasattr(layer, "layers"):
        for child_layer in layer.layers:
            inner_variables += get_inner_variables(child_layer, match_fn)

    return list(set(inner_variables))


def get_trainable_inner_variables(layer):
    """
    Get the trainable inner variables
    :param layer: current layer
    :return: trainable inner variables as a list
    """
    return get_inner_variables(layer, match_fn=lambda inner_var: not inner_var.per_step)


def warmup_inner_layer(layer, input_shape):
    """
    Warm up the inner layer
    :param layer: current layer
    :param input_shape: input shape
    :return: An inner layer with dummy input
    """
    outer_batch_size = 1
    inner_batch_size = 1
    dummy_input = tf.placeholder(tf.float32, (outer_batch_size, inner_batch_size, *input_shape))
    layer(dummy_input)


def apply_to_inner_layers(root_layer, fn):
    """
    Apply function to the inner layers
    :param root_layer: the root inner layer
    :param fn: given function
    """
    if isinstance(root_layer, InnerLayer):
        fn(root_layer)
    if hasattr(root_layer, "layers"):
        for child_layer in root_layer.layers:
            apply_to_inner_layers(child_layer, fn)


def set_inner_train_state(root_layer, is_train):
    """
    Set the training state for the inner layer
    :param root_layer: root inner layer
    :param is_train: boolean training mode
    """
    def _set(layer):
        layer.is_train = is_train

    apply_to_inner_layers(root_layer, _set)


def set_inner_step(root_layer, step):
    """
    Set the inner step
    :param root_layer: root inner layer
    :param step: step size
    """
    def _set(layer):
        layer.step = step

    apply_to_inner_layers(root_layer, _set)


class InnerVariable:
    """
    Construct an Inner Variable class
    """
    counter = 0

    def __init__(self, shape, name=None, dtype=tf.float32, per_step=False, initializer=tf.initializers.orthogonal()):
        """
        Initialize the Inner Variable class
        :param shape: variable shape
        :param name: variable name
        :param dtype: variable data type
        :param per_step: boolean to turn on training per step
        :param initializer: variable initializer method
        """
        self.getter = lambda variable, batch_index, step: tf.placeholder(dtype, shape)
        self.initializer = initializer
        self.dtype = dtype
        self.name = name
        self.shape = shape
        self.per_step = per_step

        if self.name is None:
            self.name = "InnerVariable_%d" % InnerVariable.counter
            InnerVariable.counter += 1

    def get(self, batch_index, step):
        """
        Get the Inner Variable class
        :param batch_index: batch index
        :param step: current step
        :return: the inner variable
        """
        variable = self.getter(self, batch_index, step)
        assert variable is not None
        return variable


class InnerLayer(tf.keras.layers.Layer):
    """
    Construct an Inner Layer class
    """
    def __init__(self):
        """
        Initialize the Inner Layer class
        """
        super().__init__()
        self.inner_variables = {}
        self.is_train = False
        self.step = 0

    def create_inner_variable(self, name, shape, dtype=tf.float32,
                              initializer=tf.initializers.orthogonal(), per_step=False):
        """
        Create the Inner Variable class
        :param name: variable name
        :param shape: variable shape
        :param dtype: variable data type
        :param initializer: variable initializer method
        :param per_step: boolean to turn on training per step
        """
        if name in self.inner_variables:
            raise Exception("Tried to create inner variable with existing name")
        self.inner_variables[name] = InnerVariable(shape=shape, dtype=dtype, per_step=per_step, initializer=initializer)
        print("Inner var %s: %s" % (self.inner_variables[name].name, name))
        return self.inner_variables[name]

    def call(self, inputs):
        outer_batch_size = inputs.shape[0]
        results = []
        for batch_index in range(outer_batch_size):
            results.append(self.call_single(inputs[batch_index], batch_index))
        return tf.stack(results)

    def call_single(self, inputs, batch_index):
        pass


class InnerDense(InnerLayer):
    """
    Construct an Inner Dense class
    """
    def __init__(self, dim, use_bias=True):
        """
        Initialize the Inner Dense class
        :param dim: input dimension
        :param use_bias: boolean to use bias
        """
        super().__init__()
        self.dim = dim
        self.use_bias = use_bias

    def build(self, input_shape):
        """
        Build the Inner Dense class
        :param input_shape: input shape
        """
        # Create the inner dense weights
        self.dense_weights = self.create_inner_variable("weights", (input_shape[-1], self.dim))

        # Create the inner bias
        if self.use_bias:
            self.bias = self.create_inner_variable("bias", (self.dim,), initializer=tf.zeros_initializer())

    def call_single(self, inputs, batch_index):
        # Get the inner dense weights
        dense_weights = self.dense_weights.get(batch_index, self.step)
        output = tf.matmul(inputs, dense_weights)

        # Get the inner bias
        if self.use_bias:
            bias = self.bias.get(batch_index, self.step)
            output += bias
        return output

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape
        :param input_shape: input shape
        :return: output shape
        """
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        output_shape = list(input_shape)
        output_shape[-1] = self.dim

        return tuple(output_shape)


class InnerReshape(InnerLayer):
    """
    Construct an Inner Reshape class
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def call(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return tf.reshape(inputs, output_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        batch_sizes = input_shape[:2]
        output_shape = [*batch_sizes, *self.shape]
        return tuple(output_shape)


class InnerFlatten(InnerLayer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return tf.reshape(inputs, output_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        batch_sizes = input_shape[:2]
        output_shape = [*batch_sizes, np.prod(input_shape[2:])]
        return tuple(output_shape)


class InnerResize(InnerLayer):
    def __init__(self, new_size):
        super().__init__()
        self.new_size = new_size

    def call(self, inputs):
        input_shape = inputs.shape
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        output_shape = self.compute_output_shape(inputs.shape)

        # Combine the two batch indices
        inputs = tf.reshape(inputs, (input_shape[0] * input_shape[1], *input_shape[2:]))

        # Resize
        inputs = tf.image.resize_images(inputs, self.new_size)

        # Seperate the two batch indices again (just reshape to target shape)
        return tf.reshape(inputs, output_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        batch_sizes = input_shape[:2]
        output_shape = [*batch_sizes, *self.new_size, input_shape[-1]]
        return tuple(output_shape)


class InnerConv2D(InnerLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, padding="VALID"):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = strides if len(strides) == 4 else (1, *strides, 1)
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        self.kernel = self.create_inner_variable("kernel", (
        self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters))
        if self.use_bias:
            self.bias = self.create_inner_variable("bias", (self.filters,), initializer=tf.zeros_initializer())

    def call_single(self, inputs, batch_index):
        kernel = self.kernel.get(batch_index, self.step)
        output = tf.nn.conv2d(inputs, kernel, strides=self.strides, padding=self.padding)
        if self.use_bias:
            bias = self.bias.get(batch_index, self.step)
            output = tf.nn.bias_add(output, bias)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        if self.padding == "VALID":
            padding = (0, 0)
        elif self.padding == "SAME":
            padding = (self.kernel_size[0] - self.strides[1], self.kernel_size[1] - self.strides[2])
        else:
            raise Exception("Unsupported padding type %s" % self.padding)

        output_shape = list(input_shape)
        output_shape[-3] = (input_shape[-3] - self.kernel_size[0] + 2 * padding[0]) // self.strides[1] + 1
        output_shape[-2] = (input_shape[-2] - self.kernel_size[1] + 2 * padding[1]) // self.strides[2] + 1
        output_shape[-1] = self.filters

        return tuple(output_shape)


class InnerConv2DTranspose(InnerLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, padding="VALID"):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = strides if len(strides) == 4 else (1, *strides, 1)
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        self.kernel = self.create_inner_variable("kernel", (
        self.kernel_size[0], self.kernel_size[1], self.filters, input_shape[-1]))
        if self.use_bias:
            self.bias = self.create_inner_variable("bias", (self.filters,), initializer=tf.zeros_initializer())

    def call_single(self, inputs, batch_index):
        output_shape = self.compute_output_shape(inputs.shape)

        kernel = self.kernel.get(batch_index, self.step)
        output = tf.nn.conv2d_transpose(inputs, kernel, output_shape=output_shape, strides=self.strides,
                                        padding=self.padding)
        if self.use_bias:
            bias = self.bias.get(batch_index, self.step)
            output = tf.nn.bias_add(output, bias)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        if self.padding == "VALID":
            padding = (0, 0)
        elif self.padding == "SAME":
            padding = (self.kernel_size[0] - self.strides[1], self.kernel_size[1] - self.strides[2])
        else:
            raise Exception("Unsupported padding type %s" % self.padding)

        output_shape = list(input_shape)
        output_shape[-3] = (input_shape[-3] - 1) * self.strides[1] + self.kernel_size[0] - 2 * padding[0]
        output_shape[-2] = (input_shape[-2] - 1) * self.strides[2] + self.kernel_size[1] - 2 * padding[1]
        output_shape[-1] = self.filters

        return tuple(output_shape)


class InnerNormalization(InnerLayer):
    def __init__(self, per_step=True):
        super().__init__()
        self.per_step = per_step
        self.stored_means = [0] * 100
        self.stored_vars = [1] * 100

    def build(self, input_shape):
        self.std = self.create_inner_variable("std", (input_shape[-1],), per_step=self.per_step,
                                              initializer=tf.ones_initializer())
        self.mean = self.create_inner_variable("mean", (input_shape[-1],), per_step=self.per_step,
                                               initializer=tf.zeros_initializer())

    def call_single(self, inputs, batch_index):
        std = self.std.get(batch_index, self.step)
        mean = self.mean.get(batch_index, self.step)
        # print("Called normalization with std and mean", self.std.name, self.mean.name, std, mean)
        output = std * inputs + mean
        return output

    def call(self, inputs):
        # Normalize to N(0, 1) over inner-batch axis together.
        # Then do the single-call normalization since every
        # inner batch has its own mean and std
        if self.is_train:
            stored_mean, stored_var = tf.nn.moments(inputs, axes=[1, 2, 3], keep_dims=True)
            self.stored_means[self.step] = stored_mean
            self.stored_vars[self.step] = stored_var
        inputs = (inputs - self.stored_means[self.step]) / tf.sqrt(self.stored_vars[self.step] + 1e-6)
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class InnerMemorization(InnerLayer):
    def __init__(self, per_step=True):
        super().__init__()
        self.per_step = per_step
        self.stored_values = {}

    def _get_stored_value(self, step):
        if step < 0 or step not in self.stored_values:
            return 0
        return self.stored_values[step]

    def build(self, input_shape):
        print("Input shape:", input_shape)
        keep_shape = (input_shape[-1],)
        print("Keep shape:", keep_shape)
        self.keep = self.create_inner_variable("keep", keep_shape, per_step=self.per_step,
                                               initializer=tf.constant_initializer(-1))

    def call(self, inputs):
        print("Call single", inputs, self.step)
        keep = tf.nn.sigmoid(self.keep.get(0, self.step))
        print("Keep:", keep)
        # if batch_index == 0:
        print(self.step, "Inputs pre:", inputs)
        output = (1 - keep) * inputs + keep * self._get_stored_value(self.step - 1)
        # if batch_index == 0:
        print(self.step, "Output:", output)
        if self.is_train:
            self.stored_values[self.step] = output

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
