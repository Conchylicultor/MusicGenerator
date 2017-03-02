# Copyright 2016 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
"""

import tensorflow as tf

import deepmusic.tfutils as tfutils
import deepmusic.songstruct as music


# TODO: Some class from the encoder and decoder are really similar. Could they be merged ?
class DecoderNetwork:
    """ Predict a keyboard configuration at step t
    This is just an abstract class
    Warning: To encapsulate the weights in the right tf scope, they should be defined
    within the build function
    """
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        self.args = args

    def build(self):
        """ Initialize the weights of the model
        """
        pass

    def init_state(self):
        """ Return the initial cell state
        """
        return None

    def get_cell(self, prev_keyboard, prev_state_enco):
        """ Predict the next keyboard state
        Args:
            prev_keyboard (?): the previous keyboard configuration
            prev_state_enco (?): the encoder output state
        Return:
            Tuple: A tuple containing the predicted keyboard configuration and last decoder state
        """
        raise NotImplementedError('Abstract class')


class Rnn(DecoderNetwork):
    """ Predict a keyboard configuration at step t
    Use a RNN to predict the next configuration
    """
    @staticmethod
    def get_module_id():
        return 'rnn'

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)
        self.rnn_cell = None
        self.project_key = None  # Fct which project the decoder output into a single key space

    def build(self):
        """ Initialize the weights of the model
        """
        self.rnn_cell = tfutils.get_rnn_cell(self.args, "deco_cell")
        self.project_key = tfutils.single_layer_perceptron([self.args.hidden_size, 1],
                                                   'project_key')

    def init_state(self):
        """ Return the initial cell state
        """
        return self.rnn_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

    def get_cell(self, prev_keyboard, prev_state_enco):
        """ a RNN decoder
        See parent class for arguments details
        """

        axis = 1  # The first dimension is the batch, we split the keys
        assert prev_keyboard.get_shape()[axis].value == music.NB_NOTES
        inputs = tf.split(axis, music.NB_NOTES, prev_keyboard)

        outputs, final_state = tf.nn.seq2seq.rnn_decoder(
            decoder_inputs=inputs,
            initial_state=prev_state_enco,
            cell=self.rnn_cell
            # TODO: Which loop function (should use prediction) ? : Should take the previous generated input/ground truth (as the global model loop_fct). Need to add a new bool placeholder
        )

        # Is it better to do the projection before or after the packing ?
        next_keys = []
        for output in outputs:
            next_keys.append(self.project_key(output))

        next_keyboard = tf.concat(axis, next_keys)

        return next_keyboard, final_state


class Perceptron(DecoderNetwork):
    """ Single layer perceptron. Just a proof of concept for the architecture
    """
    @staticmethod
    def get_module_id():
        return 'perceptron'

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)

        self.project_hidden = None  # Fct which decode the previous state
        self.project_keyboard = None  # Fct which project the decoder output into the keyboard space

    def build(self):
        """ Initialize the weights of the model
        """
        # For projecting on the keyboard space
        self.project_hidden = tfutils.single_layer_perceptron([music.NB_NOTES, self.args.hidden_size],
                                                              'project_hidden')

        # For projecting on the keyboard space
        self.project_keyboard = tfutils.single_layer_perceptron([self.args.hidden_size, music.NB_NOTES],
                                                                'project_keyboard')  # Should we do the activation sigmoid here ?

    def get_cell(self, prev_keyboard, prev_state_enco):
        """ Simple 1 hidden layer perceptron
        See parent class for arguments details
        """
        # Don't change the state
        next_state_deco = prev_state_enco  # Return the last state (Useful ?)

        # Compute the next output
        hidden_state = self.project_hidden(prev_keyboard)
        next_keyboard = self.project_keyboard(hidden_state)  # Should we do the activation sigmoid here ? Maybe not because the loss function does it

        return next_keyboard, next_state_deco


class Lstm(DecoderNetwork):
    """ Multi-layer Lstm. Just a wrapper around the official tf
    """
    @staticmethod
    def get_module_id():
        return 'lstm'

    def __init__(self, args, *module_args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)
        self.args = args

        self.rnn_cell = None
        self.project_keyboard = None  # Fct which project the decoder output into the ouput space

    def build(self):
        """ Initialize the weights of the model
        """
        # TODO: Control over the the Cell using module arguments instead of global arguments (hidden_size and num_layer) !!
        # RNN network
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hidden_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hidden_size)
        if not self.args.test:  # TODO: Should use a placeholder instead
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=0.9)  # TODO: Custom values
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.args.num_layers, state_is_tuple=True)

        self.rnn_cell = rnn_cell

        # For projecting on the keyboard space
        self.project_output = tfutils.single_layer_perceptron([self.args.hidden_size, 12 + 1],  # TODO: HACK: Input/output space hardcoded !!!
                                                               'project_output')  # Should we do the activation sigmoid here ?

    def init_state(self):
        """ Return the initial cell state
        """
        return self.rnn_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

    def get_cell(self, prev_input, prev_states):
        """
        """
        next_output, next_state = self.rnn_cell(prev_input, prev_states[1])
        next_output = self.project_output(next_output)
        # No activation function here: SoftMax is computed by the loss function

        return next_output, next_state
