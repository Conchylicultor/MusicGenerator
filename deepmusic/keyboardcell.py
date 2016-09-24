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

import tensorflow as tf

import deepmusic.songstruct as music


class EncoderNetwork:
    """ From the previous keyboard configuration, prepare the state for the next one
    """
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        self.args = args

    def get_cell(self, prev_keyboard, prev_state):
        prev_state_enco, prev_state_deco = prev_state

        # TODO
        next_state_enco = prev_state_enco

        return next_state_enco


class DecoderNetwork:
    """ Predict a keyboard configuration at step t
    """
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        self.args = args

        # Projection on the keyboard
        with tf.variable_scope('weights_note_projection'):
            self.W = tf.get_variable(
                'weights',
                [self.args.hidden_size, music.NB_NOTES],
                initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of hidden size)
            )
            self.b = tf.get_variable(
                'bias',
                [music.NB_NOTES],
                initializer=tf.constant_initializer()
            )

        # TODO: DELETE
        with tf.variable_scope('weights_to_delete'):
            self.W_to_delete = tf.get_variable(
                'weights',
                [music.NB_NOTES, self.args.hidden_size],
                initializer=tf.truncated_normal_initializer()
            )

    def _project_note(self, X):
        """ Project the output of the decoder into the note space
        """
        with tf.name_scope('note_projection'):
            return tf.matmul(X, self.W) + self.b  # [batch_size, NB_NOTE]  # Should we do the activation sigmoid here ? probably

    def get_cell(self, prev_keyboard, prev_state_enco):

        # TODO
        hidden_state = tf.matmul(prev_keyboard, self.W_to_delete)
        next_keyboard = self._project_note(hidden_state)
        next_state_deco = prev_state_enco  # Return the last state (Useful ?)

        return next_keyboard, next_state_deco


class KeyboardCell(tf.nn.rnn_cell.RNNCell):
    """ Cell which wrap the encoder/decoder network
    """

    def __init__(self, args):
        self.args = args
        self.init = False
        self.decoder = None

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
        if not self.init:
            with tf.variable_scope('weights_keyboard_cell'):
                # TODO: With self.args, see which network we have chosen (create map 'network name':class)
                self.encoder = EncoderNetwork(self.args)
                self.decoder = DecoderNetwork(self.args)
                self.init = True

        # TODO: If encoder act as VAE, we should sample here, from the previous state

        # Encoder/decoder network
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Encoder"):
                next_state_enco = self.encoder.get_cell(prev_keyboard, prev_state)
            with tf.variable_scope("Decoder"):  # Reset gate and update gate.
                next_keyboard, next_state_deco = self.decoder.get_cell(prev_keyboard, next_state_enco)
        return next_keyboard, (next_state_enco, next_state_deco)

    @staticmethod
    def init_state():
        init_state_enco = None
        init_state_deco = None  # Initial states (placeholder ? variables ? zeros ?), (What about the keyboard ?)

        # TODO: Call self.encoder.init_state()
        return init_state_enco, init_state_deco
