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


class LoopProcessing:
    """ Apply some processing to the rnn loop which connect the output to
    the next input.
    Is called on the loop_function attribute of the rnn decoder
    """
    def __init__(self, args):
        pass

    def __call__(self, prev_output):
        """ Function which apply the preprocessing
        Args:
            prev_output (tf.Tensor): the ouput on which applying the transformation
        Return:
            tf.Ops: the processing operator
        """
        raise NotImplementedError('Abstract Class')

    def get_op(self):
        """ Return the chosen labels from the softmax distribution
        Allows to reconstruct the song
        """
        return ()  # Empty tuple


class SampleSoftmax(LoopProcessing):
    """ Sample from the softmax distribution
    """
    @staticmethod
    def get_module_id():
        return 'sample_softmax'

    def __init__(self, args, *args_module):

        self.temperature = args.temperature  # Control the sampling (more or less concervative predictions) (TODO: Could be a argument of modeule, but in this case will automatically be restored when --test, should also be in the save name)
        self.chosen_labels = []  # Keep track of the chosen labels (to reconstruct the chosen song)

    def __call__(self, prev_output):
        """ Use TODO formula
        Args:
            prev_output (tf.Tensor): the ouput on which applying the transformation
        Return:
            tf.Ops: the processing operator
        """
        # prev_output size: [batch_size, nb_labels]
        nb_labels = prev_output.get_shape().as_list()[-1]

        if False:  # TODO: Add option to control argmax
            #label_draws = tf.argmax(prev_output, 1)
            label_draws = tf.multinomial(tf.log(prev_output), 1)  # Draw 1 sample from the distribution
            label_draws = tf.squeeze(label_draws, [1])
            self.chosen_labels.append(label_draws)
            next_input = tf.one_hot(label_draws, nb_labels)
            return next_input
        # Could use the Gumbel-Max trick to sample from a softmax distribution ?

        soft_values = tf.exp(tf.div(prev_output, self.temperature))  # Pi = exp(pi/t)
        # soft_values size: [batch_size, nb_labels]

        normalisation_coeff = tf.expand_dims(tf.reduce_sum(soft_values, 1), -1)
        # normalisation_coeff size: [batch_size, 1]
        probs = tf.div(soft_values, normalisation_coeff + 1e-8)  # = Pi / sum(Pk)
        # probs size: [batch_size, nb_labels]
        label_draws = tf.multinomial(tf.log(probs), 1)  # Draw 1 sample from the log-probability distribution
        # probs label_draws: [batch_size, 1]
        label_draws = tf.squeeze(label_draws, [1])
        # label_draws size: [batch_size,]
        self.chosen_labels.append(label_draws)
        next_input = tf.one_hot(label_draws, nb_labels)  # Reencode the next input vector
        # next_input size: [batch_size, nb_labels]
        return next_input

    def get_op(self):
        """ Return the chosen labels from the softmax distribution
        Allows to reconstruct the song
        """
        return self.chosen_labels


class ActivateScale(LoopProcessing):
    """ Activate using sigmoid and scale the prediction on [-1, 1]
    """
    @staticmethod
    def get_module_id():
        return 'activate_and_scale'

    def __init__(self, args):
        pass

    def __call__(X):
        """ Predict the output from prev and scale the result on [-1, 1]
        Use sigmoid activation
        Args:
            X (tf.Tensor): the input
        Return:
            tf.Ops: the activate_and_scale operator
        """
        # TODO: Use tanh instead ? tanh=2*sigm(2*x)-1
        with tf.name_scope('activate_and_scale'):
            return tf.sub(tf.mul(2.0, tf.nn.sigmoid(X)), 1.0)  # x_{i} = 2*sigmoid(y_{i-1}) - 1
