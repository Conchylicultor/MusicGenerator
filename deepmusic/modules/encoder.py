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
        raise NotImplementedError('Abstract Class')


class Identity(EncoderNetwork):
    """ Implement lookup for note embedding
    """

    @staticmethod
    def get_module_id():
        return 'identity'

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)

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


class Rnn(EncoderNetwork):
    """ Read each keyboard configuration note by note and encode it's configuration
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

    def build(self):
        """ Initialize the weights of the model
        """
        self.rnn_cell = tfutils.get_rnn_cell(self.args, "deco_cell")

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


class Embedding(EncoderNetwork):
    """ Implement lookup for note embedding
    """
    @staticmethod
    def get_module_id():
        return 'embedding'

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        super().__init__(args)

    def build(self):
        """ Initialize the weights of the model
        """

    def init_state(self):
        """ Return the initial cell state
        """

    def get_cell(self, prev_keyboard, prev_state):
        """ a RNN encoder
        See parent class for arguments details
        """
        # TODO:
        return

