# Import libraries
import tensorflow as tf
import numpy as np


class OuterNetwork:
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        self.inner_variables = inner_variables
        self.fixed_lr = fixed_lr
        self.output = None
        self.inner_var_index = {}
        self.inner_var_lr_index = {}
        index = 0

        # We need one more variable than steps since the final network will need variables too
        num_vars = num_inner_loops + 1
        for inner_var in inner_variables:
            inner_var_size = np.prod(inner_var.shape)
            self.inner_var_index[inner_var] = []

            for step in range(num_vars):
                self.inner_var_index[inner_var].append(index)
                if inner_var.per_step or step + 1 == num_vars:
                    index += inner_var_size

            if fixed_lr is None and not inner_var.per_step:
                # Learning rate per step
                self.inner_var_lr_index[inner_var] = index
                index += num_inner_loops

        self.output_size = index
        self.num_inner_loops = num_inner_loops
        self.slice_cache = {}

    def _get_cached(self, shape, start_index):
        def _make_sane(x):
            if isinstance(x, tf.Dimension):
                x = x.value
            return int(x)

        sane_start_index = _make_sane(start_index)
        if not sane_start_index in self.slice_cache:
            print("Not in cache")
            end_index = start_index + np.prod(shape)
            self.slice_cache[sane_start_index] = tf.reshape(self.output[:, start_index:end_index],
                                                            (self.output.shape.as_list()[0], *shape))

        return self.slice_cache[sane_start_index]

    def get_inner_variable(self, inner_variable, step):
        """
        Gets the values for the inner variable at the specified step.
        Returns one variable for every outer batch.
        [OuterBatchSize, *VariableShape]
        """
        assert self.output is not None and 0 <= step <= self.num_inner_loops
        start_index = self.inner_var_index[inner_variable][step]
        return self._get_cached(inner_variable.shape, start_index)

    def get_learning_rate(self, inner_variable, step):
        assert self.output is not None and not inner_variable.per_step and 0 <= step < self.num_inner_loops

        if self.fixed_lr is None:
            index = self.inner_var_lr_index[inner_variable] + step
            shape = (1,) * len(inner_variable.shape)
            return self._get_cached(shape, index)
        else:
            return self.fixed_lr

    def calculate_output(self, inputs):
        pass
