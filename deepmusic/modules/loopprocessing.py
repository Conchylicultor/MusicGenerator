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


class SampleSoftmax(LoopProcessing):
    """ Sample from the softmax distribution
    """
    @staticmethod
    def get_module_id():
        return 'sample_softmax'

    def __init__(self, args, *args_module):
        self.temperature = 1.0  # Control the sampling (more or less concervative predictions)

    def __call__(self, prev_output):
        """ Use TODO formula
        Args:
            prev_output (tf.Tensor): the ouput on which applying the transformation
        Return:
            tf.Ops: the processing operator
        """
        # TODO
        return prev_output


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
