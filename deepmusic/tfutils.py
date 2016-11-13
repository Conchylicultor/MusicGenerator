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
Some functions to help define neural networks
"""

import tensorflow as tf


def single_layer_perceptron(shape, scope_name):
    """ Single layer perceptron
    Project X on the output dimension
    Args:
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
        Args:
            X (tf.Tensor): input value
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
