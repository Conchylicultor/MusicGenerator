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
Model to generate new songs

"""

import tensorflow as tf

from deepmusic.musicdata import Batch


class Model:
    """
    Base class which manage the different models and experimentations.
    """
    
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        print("Model creation...")

        self.args = args  # Keep track of the parameters of the model

        # Placeholders
        self.inputs = None

        # Main operators
        self.opt_op = None  # Optimizer
        self.outputs = None  # Outputs of the network,

        # Construct the graphs
        self._build_network()

    def _build_network(self):
        """ Create the computational graph
        """

        # TODO

        # For testing only
        if self.args.test:
            self.outputs = None  # TODO
            # Attach a summary to visualize the output image ?

        # For training only
        else:
            # Finally, we define the loss function
            loss_fct = None  # TODO
            tf.scalar_summary('training_loss', loss_fct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.opt_op = opt.minimize(loss_fct)
    
    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training,) operator or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feed_dict = {}
        ops = None

        if not self.args.test:  # Training
            # TODO: Feed placeholder

            ops = (self.opt_op,)
        else:  # Testing (batch_size == 1)
            # TODO: Feed placeholder

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feed_dict
