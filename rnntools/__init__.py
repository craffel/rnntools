'''
Recurrent layer add-ons for nntools
'''

import theano
import theano.tensor as T
import nntools


class RecurrentLayer(nntools.layers.Layer):
    '''
    A "vanilla" recurrent layer.  Has a single hidden recurrent unit.
    '''
    def __init__(self, input_layer, num_units,
                 W_input=nntools.init.Normal(0.01),
                 W_recurrent=nntools.init.Normal(0.01),
                 b=nntools.init.Constant(0.),
                 h=nntools.init.Constant(0.),
                 nonlinearity=nntools.nonlinearities.rectify):
        '''
        Create a new recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_input : function or np.ndarray or theano.shared
                Initial input-to-hidden weight matrix
            - W_recurrent : function or np.ndarray or theano.shared
                Initial hidden (previous time step)-to-hidden weight matrix
            - b : function or np.ndarray or theano.shared
                Initial bias vector
            - h : function or np.ndarray or theano.shared
                Initial hidden state
            - nonlinearity : function
                Nonlinearity to use
        '''
        # Initialize parent layer
        super(RecurrentLayer, self).__init__(input_layer)
        # Use identity (linear) nonlinearity if supplied with None
        if nonlinearity is None:
            self.nonlinearity = nntools.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = self.input_layer.get_output_shape()[1]

        # Initialize parameters using the supplied args
        self.W_input = self.create_param(W_input, (num_inputs, num_units))
        self.W_recurrent = self.create_param(W_recurrent,
                                             (num_units, num_units))
        self.b = self.create_param(b, (num_units,))
        self.h = self.create_param(h, (num_units,))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        return [self.W_input, self.W_recurrent, self.b]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 2:
            input = input.reshape((input.shape[0], T.prod(input.shape[1:])))

        # Create single recurrent computation step function
        def step(layer_input, previous_output, W_input, W_recurrent, b):
            return self.nonlinearity(T.dot(layer_input, W_input) +
                                     T.dot(previous_output, W_recurrent) +
                                     b)
        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
        return theano.scan(step, sequences=input, outputs_info=[self.h],
                           non_sequences=[self.W_input,
                                          self.W_recurrent,
                                          self.b])[0]


class LSTMLayer(nntools.layers.Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition.

    :references:
        .. [#graves2014generator] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks".
    '''
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
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_input_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{xi}`
            - W_hidden_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{hi}`
            - W_cell_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{ci}`
            - b_input_gate : function or np.ndarray or theano.shared
                :math:`b_i`
            - nonlinearity_input_gate : function
                :math:`\sigma`
            - W_input_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{xf}`
            - W_hidden_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{hf}`
            - W_cell_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{cf}`
            - b_forget_gate : function or np.ndarray or theano.shared
                :math:`b_f`
            - nonlinearity_forget_gate : function
                :math:`\sigma`
            - W_input_to_cell : function or np.ndarray or theano.shared
                :math:`W_{ic}`
            - W_hidden_to_cell : function or np.ndarray or theano.shared
                :math:`W_{hc}`
            - b_cell : function or np.ndarray or theano.shared
                :math:`b_c`
            - nonlinearity_cell : function or np.ndarray or theano.shared
                :math:`\tanh`
            - W_input_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{io}`
            - W_hidden_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{ho}`
            - W_cell_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{co}`
            - b_output_gate : function or np.ndarray or theano.shared
                :math:`b_o`
            - nonlinearity_output_gate : function
                :math:`\sigma`
            - nonlinearity_output : function or np.ndarray or theano.shared
                :math:`\tanh`
            - c : function or np.ndarray or theano.shared
                :math:`c_0`
            - h : function or np.ndarray or theano.shared
                :math:`h_0`
        '''
        # Initialize parent layer
        super(LSTMLayer, self).__init__(input_layer)

        # For any of the nonlinearities, if None is supplied, use identity
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

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = self.input_layer.get_output_shape()[1]

        # Initialize parameters using the supplied args
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
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
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
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_input_gate, self.b_forget_gate,
                self.b_cell, self.b_output_gate]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # Treat all layers after the first as flattened feature dimensions
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
            # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
            input_gate = self.nonlinearity_input_gate(
                T.dot(layer_input, W_input_to_input_gate) +
                T.dot(previous_output, W_hidden_to_input_gate) +
                T.dot(previous_cell, W_cell_to_input_gate) +
                b_input_gate)
            # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
            forget_gate = self.nonlinearity_forget_gate(
                T.dot(layer_input, W_input_to_forget_gate) +
                T.dot(previous_output, W_hidden_to_forget_gate) +
                T.dot(previous_cell, W_cell_to_forget_gate) +
                b_forget_gate)
            # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
            cell = (forget_gate*previous_cell +
                    input_gate*self.nonlinearity_cell(
                        T.dot(layer_input, W_input_to_cell) +
                        T.dot(previous_cell, W_hidden_to_cell) +
                        b_cell))
            # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_{t-1} + b_o)
            output_gate = self.nonlinearity_output_gate(
                T.dot(layer_input, W_input_to_output_gate) +
                T.dot(previous_output, W_hidden_to_output_gate) +
                T.dot(cell, W_cell_to_output_gate) +
                b_output_gate)
            # h_t = o_t \tanh(c_t)
            output = output_gate*self.nonlinearity_output(cell)
            return [cell, output]

        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
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
