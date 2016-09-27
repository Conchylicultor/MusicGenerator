#!/usr/bin/env python3

# Copyright 2015 Conchylicultor. All Rights Reserved.
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
Main cell which predict the next keyboard configuration

"""

import collections
import tensorflow as tf

import deepmusic.songstruct as music


def single_layer_perceptron(shape, scope_name):
    """ Single layer perceptron
    Project X on the output dimension
    Args:
        X (tf.Tensor): input value of shape TODO
        shape: a tuple (input dim, output dim)
        scope_name (str): encapsulate variables
    Return:
        tf.Ops: The projection operator (see project_fct())
    """
    assert len(shape) == 2

    # Projection on the keyboard
    with tf.variable_scope('weights_' + scope_name):
        W = tf.get_variable(
            'weights',
            shape,
            initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
        )
        b = tf.get_variable(
            'bias',
            shape[1],
            initializer=tf.constant_initializer()
        )

    def project_fct(X):
        """ Project the output of the decoder into the note space
        """
        # TODO: Could we add an activation function as option ?
        with tf.name_scope(scope_name):
            return tf.matmul(X, W) + b

    return project_fct

def get_rnn_cell(args, scope_name):
    """ Return RNN cell, constructed from the parameters
    Args:
        args: the rnn parameters
        scope_name (str): encapsulate variables
    Return:
        tf.RNNCell: a cell
    """
    with tf.variable_scope('weights_' + scope_name):
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hidden_size)
        #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=1.0)  # TODO: Custom values (WARNING: No dropout when testing !!!, possible to use placeholder ?)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * args.num_layers, state_is_tuple=True)
    return rnn_cell


class EncoderNetwork:
    """ From the previous keyboard configuration, prepare the state for the next one
    Encode the keyboard configuration at a state t
    This abstract class has no effect be is here to be subclasses
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

    def init_state(self):
        """ Return the initial cell state
        """
        return None

    def get_cell(self, prev_keyboard, prev_state):
        """ Predict the next keyboard state
        Args:
            prev_keyboard (tf.Tensor): the previous keyboard configuration
            prev_state (Tuple): the previous decoder state
        Return:
            tf.Tensor: the final encoder state
        """
        prev_state_enco, prev_state_deco = prev_state

        # This simple class just pass the previous state
        next_state_enco = prev_state_enco

        return next_state_enco


class EncoderNetworkRnn(EncoderNetwork):
    """ Read each keyboard configuration note by note and encode it's configuration
    """
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)
        self.rnn_cell = None

    def build(self):
        """ Initialize the weights of the model
        """
        self.rnn_cell = get_rnn_cell(self.args, "deco_cell")

    def init_state(self):
        """ Return the initial cell state
        """
        return self.rnn_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

    def get_cell(self, prev_keyboard, prev_state):
        """ a RNN encoder
        See parent class for arguments details
        """
        prev_state_enco, prev_state_deco = prev_state

        axis = 1  # The first dimension is the batch, we split the keys
        assert prev_keyboard.get_shape()[axis].value == music.NB_NOTES
        inputs = tf.split(axis, music.NB_NOTES, prev_keyboard)

        _, final_state = tf.nn.rnn(
            self.rnn_cell,
            inputs,
            initial_state=prev_state_deco
        )

        return final_state


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


class DecoderNetworkRNN(DecoderNetwork):
    """ Predict a keyboard configuration at step t
    Use a RNN to predict the next configuration
    """
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
        self.rnn_cell = get_rnn_cell(self.args, "deco_cell")
        self.project_key = single_layer_perceptron([self.args.hidden_size, 1],
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


class DecoderNetworkPerceptron(DecoderNetwork):
    """ Single layer perceptron. Just a proof of concept for the architecture
    """
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
        self.project_hidden = single_layer_perceptron([music.NB_NOTES, self.args.hidden_size],
                                                      'project_hidden')

        # For projecting on the keyboard space
        self.project_keyboard = single_layer_perceptron([self.args.hidden_size, music.NB_NOTES],
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


class KeyboardCell(tf.nn.rnn_cell.RNNCell):
    """ Cell which wrap the encoder/decoder network
    """
    CHOICE_ENCO = collections.OrderedDict([  # Need ordered because the fist element will be the default choice
        ('rnn', EncoderNetworkRnn),
        ('none', EncoderNetwork)
    ])
    CHOICE_DECO = collections.OrderedDict([
        ('rnn', DecoderNetworkRNN),
        ('mlp', DecoderNetworkPerceptron)
    ])

    @staticmethod
    def get_enco_choices():
        """ Return the list of the different modes
        Useful when parsing the command lines arguments
        """
        return list(KeyboardCell.CHOICE_ENCO.keys())

    @staticmethod
    def get_deco_choices():
        """ Return the list of the different modes
        Useful when parsing the command lines arguments
        """
        return list(KeyboardCell.CHOICE_DECO.keys())

    def __init__(self, args):
        self.args = args
        self.is_init = False

        # Get the chosen enco/deco
        self.encoder = KeyboardCell.CHOICE_ENCO[self.args.enco](self.args)
        self.decoder = KeyboardCell.CHOICE_DECO[self.args.deco](self.args)

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    def __call__(self, prev_keyboard, prev_state, scope=None):
        """ Run the cell at step t
        Args:
            prev_keyboard: keyboard configuration for the step t-1 (Ground truth or previous step)
            prev_state: a tuple (prev_state_enco, prev_state_deco)
            scope: TensorFlow scope
        Return:
            Tuple: the keyboard configuration and the enco and deco states
        """

        # First time only (we do the initialisation here to be on the global rnn loop scope)
        if not self.is_init:
            with tf.variable_scope('weights_keyboard_cell'):
                # TODO: With self.args, see which network we have chosen (create map 'network name':class)
                self.encoder.build()
                self.decoder.build()

                prev_state = self.encoder.init_state(), self.decoder.init_state()
                self.is_init = True

        # TODO: If encoder act as VAE, we should sample here, from the previous state

        # Encoder/decoder network
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Encoder"):
                next_state_enco = self.encoder.get_cell(prev_keyboard, prev_state)
            with tf.variable_scope("Decoder"):  # Reset gate and update gate.
                next_keyboard, next_state_deco = self.decoder.get_cell(prev_keyboard, next_state_enco)
        return next_keyboard, (next_state_enco, next_state_deco)
