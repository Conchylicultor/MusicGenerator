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
Main cell which predict the next keyboard configuration

"""

import collections
import tensorflow as tf

from deepmusic.moduleloader import ModuleLoader
import deepmusic.songstruct as music


class KeyboardCell(tf.nn.rnn_cell.RNNCell):
    """ Cell which wrap the encoder/decoder network
    """

    def __init__(self, args):
        self.args = args
        self.is_init = False

        # Get the chosen enco/deco
        self.encoder = ModuleLoader.enco_cells.build_module(self.args)
        self.decoder = ModuleLoader.deco_cells.build_module(self.args)

    @property
    def state_size(self):
        raise NotImplementedError('Abstract method')

    @property
    def output_size(self):
        raise NotImplementedError('Abstract method')

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
            with tf.variable_scope('Encoder'):
                # TODO: Should be enco_output, enco_state
                next_state_enco = self.encoder.get_cell(prev_keyboard, prev_state)
            with tf.variable_scope('Decoder'):  # Reset gate and update gate.
                next_keyboard, next_state_deco = self.decoder.get_cell(prev_keyboard, (next_state_enco, prev_state[1]))
        return next_keyboard, (next_state_enco, next_state_deco)
