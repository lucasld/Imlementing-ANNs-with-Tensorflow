import tensorflow as tf
from typing import Tuple

class LSTM_Cell(tf.keras.layers.Layer):
    """LSTM-Cell implementation.
    
    :param units: number of units inside the cell-gates
    :type units: integer
    :param unit_forget_bias: forget gate bias intialized with ones if true,
        else initialized using glorot_uniform, defaults to True
    :type unit_forget_bias: boolean, optional
    """
    def __init__(self, units, unit_forget_bias=True) -> None:
        """Constructor function"""
        super(LSTM_Cell, self).__init__()
        self.units = units
        # forget gate
        self.fg_layer = tf.keras.layers.Dense(
            units,
            activation='sigmoid',
            bias_initializer='ones' if unit_forget_bias else 'glorot_uniform'
        )
        # input gate
        self.ig_layer = self.fg_W = tf.keras.layers.Dense(
            units,
            activation='sigmoid'
        )
        # output gate
        self.og_layer = self.fg_W = tf.keras.layers.Dense(
            units,
            activation='sigmoid'
        )
        # cell
        self.cell_layer = self.fg_W = tf.keras.layers.Dense(
            units,
            activation='tanh'
        )
    

    def call(self, x, states) -> Tuple[tf.Tensor]:
        """Compute forward pass through cell.

        :param x: Cell inputs
        :type x: tf.Tensor
        :param states: previous hidden and cell states
        :type states: tuple of two Tensors
        :return: new cell hidden and cell states
        :rtype: tuple of two tf.Tensor
        """
        prev_hidden_state, prev_cell_state = states
        # gate inputs
        xh = tf.concat([x, prev_hidden_state], axis=1)
        # forget gate output
        ffilter = self.fg_layer(xh)
        # input gate output
        ifilter = self.ig_layer(xh)
        # cell state candidates
        cs_cand = self.cell_layer(xh)
        # update cell state
        cell_state = tf.math.multiply(ffilter, prev_cell_state) +\
                     tf.math.multiply(ifilter, cs_cand)
        # output gate output
        ofilter = self.og_layer(xh)
        # new hidden state
        hidden_state = tf.math.multiply(ofilter, tf.nn.tanh(cell_state))
        return hidden_state, cell_state


class LSTM_Layer(tf.keras.layers.Layer):
    """LSTM-Layer wrapping LSTM-Cell.
    
    :param units: number of units of the LSTM-Cell
    :type units: integer
    """
    def __init__(self, units) -> None:
        """Constructor function"""
        super(LSTM_Layer, self).__init__()
        self.cell = LSTM_Cell(units)
    
    @tf.function
    def call(self, x, states=None) -> tf.Tensor:
        """Compute forward pass through layer.

        :param x: layer inputs
        :type x: tf.Tensor
        :param states: first hidden and cell state, defaults to None and will
            therefore be initialized to all 0's
        :type states: tuple of two Tensors or None, optional
        :return: sequence of cells hidden states
        :rtype: tf.Tensor
        """
        batch_size = tf.shape(x)[0]
        sequence_length = tf.shape(x)[1]
        # initialize states
        if not states:
            states = self.zero_states(batch_size)
        # initialize array for output sequence
        output_sequence = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # iterate through time steps in input sequence
        for seq_idx in tf.range(sequence_length):
            input = x[:, seq_idx]
            states = self.cell(input, states)
            # write the new hidden state to the output_sequence
            output_sequence = output_sequence.write(seq_idx, states[0])
        output_sequence = tf.transpose(output_sequence.stack(), perm=[1,0,2])
        return output_sequence
    
    
    def zero_states(self, batch_size) -> Tuple[tf.Tensor]:
        """Create hidden and cell state tensors filled with 0's for cell's
        initial states.
        
        :param batch_size: batch size of training data
        :type batch_size: integer
        :return: hiddden and cell state tensors of shape
            (batch_size, cell.units)
        :rtype: tuple of two Tensors
        """
        return (tf.zeros((batch_size, self.cell.units)),
                tf.zeros((batch_size, self.cell.units)))


class LSTM_Model(tf.keras.Model):
    """Model containing one or several LSTM-Layer's
    
    :param layer_list: list containing models layers
    :type layer_list: list of layers (tf.layers or LSTM_Layer)
    """
    def __init__(self, layer_list) -> None:
        """Constructor function"""
        super(LSTM_Model, self).__init__()
        self.layer_list = layer_list
    
    def call(self, x):
        """Compute forward pass through model for a sequence of inputs.
        
        :param x: sequence of inputs to the model
        :type x: tf.Tensor
        :return: models output for the last sequence input
        :rtype: tf.Tensor
        """
        input = x
        for layer in self.layer_list:
            input = layer(input)    
        return input[:,-1,:]

        
