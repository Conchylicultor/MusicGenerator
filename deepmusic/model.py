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
import deepmusic.songstruct as music


class Model:
    """
    Base class which manage the different models and experimentation.
    """
    
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        print('Model creation...')

        self.args = args  # Keep track of the parameters of the model

        # Placeholders
        self.inputs = None
        self.targets = None
        self.use_prev = None  # Boolean tensor which say at Graph evaluation time if we use the input placeholder or the previous output.

        # Main operators
        self.opt_op = None  # Optimizer
        self.outputs = None  # Outputs of the network
        self.final_state = None  # When testing, we feed this value as initial state ?

        # Construct the graphs
        self._build_network()

    def _build_network(self):
        """ Create the computational graph
        """

        # Placeholders (Use tf.SparseTensor with training=False instead)
        with tf.name_scope('placeholder_inputs'):
            self.inputs = [
                tf.placeholder(
                    tf.float32,  # -1.0/1.0 ? Probably better for the sigmoid
                    [self.args.batch_size, music.NB_NOTES],
                    name='input')
                for _ in range(self.args.sample_length)
                ]
        with tf.name_scope('placeholder_targets'):
            self.targets = [
                tf.placeholder(
                    tf.float32,  # 0/1
                    [self.args.batch_size, music.NB_NOTES],
                    name='target')
                for _ in range(self.args.sample_length)
                ]
        with tf.name_scope('placeholder_use_prev'):
            self.use_prev = [
                tf.placeholder(
                    tf.bool,
                    [],
                    name='use_prev')
                for _ in range(self.args.sample_length)  # The first value will never be used (always takes self.input for the first step)
                ]

        # Projection on the keyboard
        with tf.name_scope('note_projection_weights'):
            W = tf.Variable(
                tf.truncated_normal([self.args.hidden_size, music.NB_NOTES]),
                name='weights'
            )
            b = tf.Variable(
                tf.truncated_normal([music.NB_NOTES]),  # Tune the initializer ?
                name='bias',
            )

        def project_note(X):
            with tf.name_scope('note_projection'):
                return tf.matmul(X, W) + b  # [batch_size, NB_NOTE]

        # RNN network
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hidden_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hidden_size)
        #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1.0, output_keep_prob=1.0)  # TODO: Custom values (WARNING: No dropout when testing !!!, possible to use placeholder ?)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * self.args.num_layers, state_is_tuple=True)

        initial_state = rnn_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32)

        def loop_rnn(prev, i):
            """ Loop function used to connect one output of the rnn to the next input.
            Will re-adapt the output shape to the input one.
            This is useful to use the same network for both training and testing. Warning: Because of the fixed
            batch size, we have to predict batch_size sequences when testing.
            """
            # Predict the output from prev and scale the result on [-1, 1]
            next_input = project_note(prev)
            next_input = tf.sub(tf.mul(2.0, tf.nn.sigmoid(next_input)), 1.0)  # x_{i} = 2*sigmoid(y_{i-1}) - 1

            # On training, we force the correct input, on testing, we use the previous output as next input
            return tf.cond(self.use_prev[i], lambda: next_input, lambda: self.inputs[i])

        (outputs, self.final_state) = tf.nn.seq2seq.rnn_decoder(
            decoder_inputs=self.inputs,
            initial_state=initial_state,
            cell=rnn_cell,
            loop_function=loop_rnn  # TODO: WARNING!!! Check the graph, looks strange ??
        )

        # Final projection
        with tf.name_scope('final_output'):
            self.outputs = []
            for output in outputs:
                proj = project_note(output)
                self.outputs.append(proj)

        # For training only
        if not self.args.test:
            # Finally, we define the loss function

            # The network will predict a mix a wrong and right notes. For the loss function, we would like to
            # penalize note which are wrong. Eventually, the penalty should be less if the network predict the same
            # note but not in the right pitch (ex: C4 instead of C5), with a decay the further the prediction
            # is (D5 and D1 more penalized than D4 and D3 if the target is D2)

            # For now, by using sigmoid_cross_entropy_with_logits, the task is formulated as a NB_NOTES binary
            # classification problems

            loss_fct = tf.nn.seq2seq.sequence_loss(  # Or sequence_loss_by_example ??
                self.outputs,
                self.targets,
                [tf.ones(self.targets[0].get_shape()) for _ in range(len(self.targets))],  # Weights
                softmax_loss_function=tf.nn.sigmoid_cross_entropy_with_logits
            )
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
            # Feed placeholder
            for i in range(self.args.sample_length):
                feed_dict[self.inputs[i]] = batch.inputs[i]
                feed_dict[self.targets[i]] = batch.targets[i]
                # TODO: S. Bengio trick
                feed_dict[self.use_prev[i]] = False

            ops = (self.opt_op,)
        else:  # Testing (batch_size == 1)
            # Feed placeholder
            # TODO: What to put for initialisation state (empty ? random ?) ?
            # TODO: Modify use_prev
            for i in range(self.args.sample_length):
                if i < len(batch.inputs):
                    feed_dict[self.inputs[i]] = batch.inputs[i]
                    feed_dict[self.use_prev[i]] = False
                else:  # Even not used, we still need to feed a placeholder
                    feed_dict[self.inputs[i]] = batch.inputs[0]  # Could be anything but we need it to be of the right shape
                    feed_dict[self.use_prev[i]] = True  # When we don't have an input, we use the previous output instead

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feed_dict
