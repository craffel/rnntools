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


class LSTMLayer(nntools.layers.Layer):
    def __init__(self, input_layer, num_units,
                 W_input_to_input_gate=nntools.init.Normal(0.1),
                 W_hidden_to_input_gate=nntools.init.Normal(0.1),
                 W_cell_to_input_gate=nntools.init.Normal(0.1),
                 b_input_gate=nntools.init.Constant(1.),
                 nonlinearity_input_gate=nntools.nonlinearities.sigmoid,
                 W_input_to_forget_gate=nntools.init.Normal(0.1),
                 W_hidden_to_forget_gate=nntools.init.Normal(0.1),
                 W_cell_to_forget_gate=nntools.init.Normal(0.1),
                 b_forget_gate=nntools.init.Constant(1.),
                 nonlinearity_forget_gate=nntools.nonlinearities.sigmoid,
                 W_input_to_cell=nntools.init.Normal(0.1),
                 W_hidden_to_cell=nntools.init.Normal(0.1),
                 b_cell=nntools.init.Constant(1.),
                 nonlinearity_cell=nntools.nonlinearities.tanh,
                 W_input_to_output_gate=nntools.init.Normal(0.1),
                 W_hidden_to_output_gate=nntools.init.Normal(0.1),
                 W_cell_to_output_gate=nntools.init.Normal(0.1),
                 b_output_gate=nntools.init.Constant(1.),
                 nonlinearity_output_gate=nntools.nonlinearities.sigmoid,
                 nonlinearity_output=nntools.nonlinearities.tanh,
                 c=nntools.init.Constant(0.),
                 h=nntools.init.Constant(0.)):
        super(LSTMLayer, self).__init__(input_layer)

        if nonlinearity_input_gate is None:
            self.nonlinearity_input_gate = nntools.nonlinearities.identity
        else:
            self.nonlinearity_input_gate = nonlinearity_input_gate

        if nonlinearity_forget_gate is None:
            self.nonlinearity_forget_gate = nntools.nonlinearities.identity
        else:
            self.nonlinearity_forget_gate = nonlinearity_forget_gate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nntools.nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_output_gate is None:
            self.nonlinearity_output_gate = nntools.nonlinearities.identity
        else:
            self.nonlinearity_output_gate = nonlinearity_output_gate

        if nonlinearity_output is None:
            self.nonlinearity_output = nntools.nonlinearities.identity
        else:
            self.nonlinearity_output = nonlinearity_output

        self.num_units = num_units

        num_inputs = self.input_layer.get_output_shape()[1]

        self.W_input_to_input_gate = self.create_param(
            W_input_to_input_gate, (num_inputs, num_units))

        self.W_hidden_to_input_gate = self.create_param(
            W_hidden_to_input_gate, (num_units, num_units))

        self.W_cell_to_input_gate = self.create_param(
            W_cell_to_input_gate, (num_units, num_units))

        self.b_input_gate = self.create_param(b_input_gate, (num_units))

        self.W_input_to_forget_gate = self.create_param(
            W_input_to_forget_gate, (num_inputs, num_units))

        self.W_hidden_to_forget_gate = self.create_param(
            W_hidden_to_forget_gate, (num_units, num_units))

        self.W_cell_to_forget_gate = self.create_param(
            W_cell_to_forget_gate, (num_units, num_units))

        self.b_forget_gate = self.create_param(b_forget_gate, (num_units,))

        self.W_input_to_cell = self.create_param(
            W_input_to_cell, (num_inputs, num_units))

        self.W_hidden_to_cell = self.create_param(
            W_hidden_to_cell, (num_units, num_units))

        self.b_cell = self.create_param(b_cell, (num_units,))

        self.W_input_to_output_gate = self.create_param(
            W_input_to_output_gate, (num_inputs, num_units))

        self.W_hidden_to_output_gate = self.create_param(
            W_hidden_to_output_gate, (num_units, num_units))

        self.W_cell_to_output_gate = self.create_param(
            W_cell_to_output_gate, (num_units, num_units))

        self.b_output_gate = self.create_param(b_output_gate, (num_units,))

        self.c = self.create_param(c, (num_units,))
        self.h = self.create_param(c, (num_units,))

    def get_params(self):
        return [self.W_input_to_input_gate,
                self.W_hidden_to_input_gate,
                self.W_cell_to_input_gate,
                self.b_input_gate,
                self.W_input_to_forget_gate,
                self.W_hidden_to_forget_gate,
                self.W_cell_to_forget_gate,
                self.b_forget_gate,
                self.W_input_to_cell,
                self.W_hidden_to_cell,
                self.b_cell,
                self.W_input_to_output_gate,
                self.W_hidden_to_output_gate,
                self.W_cell_to_output_gate,
                self.b_output_gate]

    def get_bias_params(self):
        return [self.b_input_gate, self.b_forget_gate,
                self.b_cell, self.b_output_gate]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 2:
            input = input.reshape((input.shape[0], T.prod(input.shape[1:])))

        # Create single recurrent computation step function
        def step(layer_input, previous_cell, previous_output,
                 W_input_to_input_gate, W_hidden_to_input_gate,
                 W_cell_to_input_gate, b_input_gate, W_input_to_forget_gate,
                 W_hidden_to_forget_gate, W_cell_to_forget_gate, b_forget_gate,
                 W_input_to_cell, W_hidden_to_cell, b_cell,
                 W_input_to_output_gate, W_hidden_to_output_gate,
                 W_cell_to_output_gate, b_output_gate):
            input_gate = self.nonlinearity_input_gate(
                T.dot(layer_input, W_input_to_input_gate) +
                T.dot(previous_output, W_hidden_to_input_gate) +
                T.dot(previous_cell, W_cell_to_input_gate) +
                b_input_gate)
            forget_gate = self.nonlinearity_forget_gate(
                T.dot(layer_input, W_input_to_forget_gate) +
                T.dot(previous_output, W_hidden_to_forget_gate) +
                T.dot(previous_cell, W_cell_to_forget_gate) +
                b_forget_gate)
            cell = (forget_gate*previous_cell +
                    input_gate*self.nonlinearity_cell(
                        T.dot(layer_input, W_input_to_cell) +
                        T.dot(previous_cell, W_hidden_to_cell) +
                        b_cell))
            output_gate = self.nonlinearity_output_gate(
                T.dot(layer_input, W_input_to_output_gate) +
                T.dot(previous_output, W_hidden_to_output_gate) +
                T.dot(cell, W_cell_to_output_gate) +
                b_output_gate)
            output = output_gate*self.nonlinearity_output(cell)
            return [cell, output]

        return theano.scan(step, sequences=input,
                           outputs_info=[self.c, self.h],
                           non_sequences=[self.W_input_to_input_gate,
                                          self.W_hidden_to_input_gate,
                                          self.W_cell_to_input_gate,
                                          self.b_input_gate,
                                          self.W_input_to_forget_gate,
                                          self.W_hidden_to_forget_gate,
                                          self.W_cell_to_forget_gate,
                                          self.b_forget_gate,
                                          self.W_input_to_cell,
                                          self.W_hidden_to_cell,
                                          self.b_cell,
                                          self.W_input_to_output_gate,
                                          self.W_hidden_to_output_gate,
                                          self.W_cell_to_output_gate,
                                          self.b_output_gate])[0][1]
