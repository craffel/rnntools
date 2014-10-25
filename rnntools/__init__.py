import theano
import theano.tensor as T
import nntools


class RecurrentLayer(nntools.layers.Layer):
    def __init__(self, input_layer, num_units,
                 W_input=nntools.init.Normal(0.01),
                 W_recurrent=nntools.init.Normal(0.01),
                 b=nntools.init.Constant(0.),
                 h=nntools.init.Constant(0.),
                 nonlinearity=nntools.nonlinearities.rectify):
        super(RecurrentLayer, self).__init__(input_layer)
        if nonlinearity is None:
            self.nonlinearity = nntools.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units

        num_inputs = self.input_layer.get_output_shape()[1]

        self.W_input = self.create_param(W_input, (num_inputs, num_units))
        self.W_recurrent = self.create_param(W_recurrent,
                                             (num_units, num_units))
        self.b = self.create_param(b, (num_units,))
        self.h = self.create_param(h, (num_units,))

    def get_params(self):
        return [self.W_input, self.W_recurrent, self.b]

    def get_bias_params(self):
        return [self.b]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            input = input.reshape((input.shape[0], T.prod(input.shape[1:])))

        # Create single recurrent computation step function
        def step(layer_input, previous_output, W_input, W_recurrent, b):
            return self.nonlinearity(T.dot(layer_input, W_input) +
                                     T.dot(previous_output, W_recurrent) +
                                     b)
        return theano.scan(step, sequences=input, outputs_info=[self.h],
                           non_sequences=[self.W_input,
                                          self.W_recurrent,
                                          self.b])[0]
