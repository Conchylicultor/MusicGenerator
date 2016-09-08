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

import numpy as np  # To generate random numbers
import tensorflow as tf

from deepmusic.musicdata import Batch
import deepmusic.songstruct as music


class Model:
    """
    Base class which manage the different models and experimentation.
    """

    class TargetWeightsPolicy:
        """ Structure to represent the different policy for choosing the target weights
        This is used to scale the contribution of each timestep to the global loss
        """
        NONE = 'none'  # All weights equals (=1.0) (default behavior)
        LINEAR = 'linear'  # The first outputs are less penalized than the last ones
        STEP = 'step'  # We start penalizing only after self.args.target_weight_step

        def __init__(self, args):
            """
            Args:
                args: parameters of the model
            """
            self.args = args

        def get_weight(self, i):
            """ Return the target weight for the given step i using the chosen policy
            """
            if not self.args.target_weights or self.args.target_weights == Model.TargetWeightsPolicy.NONE:
                return 1.0
            elif self.args.target_weights == Model.TargetWeightsPolicy.LINEAR:
                return i / (self.args.sample_length - 1)  # Gradually increment the loss weight
            elif self.args.target_weights == Model.TargetWeightsPolicy.STEP:
                raise NotImplementedError('Step target weight policy not implemented yet, please consider another policy')
            else:
                raise ValueError('Unknown chosen target weight policy: {}'.format(self.args.target_weights))

        @staticmethod
        def get_policies():
            """ Return the list of the different modes
            Useful when parsing the command lines arguments
            """
            return [
                Model.TargetWeightsPolicy.NONE,
                Model.TargetWeightsPolicy.LINEAR,
                Model.TargetWeightsPolicy.STEP
            ]

    class ScheduledSamplingPolicy:
        """ Container for the schedule sampling policy
        See http://arxiv.org/abs/1506.03099 for more details
        """
        NONE = 'none'  # No scheduled sampling (always take the given input)
        ALWAYS = 'always'  # Always samples from the predicted output
        LINEAR = 'linear'  # Gradually increase the sampling rate
        
        def __init__(self, args):
            self.sampling_policy_fct = None

            assert args.scheduled_sampling
            assert len(args.scheduled_sampling) > 0

            policy = args.scheduled_sampling[0]
            if policy == Model.ScheduledSamplingPolicy.NONE:
                self.sampling_policy_fct = lambda step: 1.0
            elif policy == Model.ScheduledSamplingPolicy.ALWAYS:
                self.sampling_policy_fct = lambda step: 0.0
            elif policy == Model.ScheduledSamplingPolicy.LINEAR:
                if len(args.scheduled_sampling) != 5:
                    raise ValueError('Not the right arguments for the sampling linear policy ({} instead of 4)'.format(len(args.scheduled_sampling)-1))

                start_step = int(args.scheduled_sampling[1])
                end_step = int(args.scheduled_sampling[2])
                start_value = float(args.scheduled_sampling[3])
                end_value = float(args.scheduled_sampling[4])

                if (start_step >= end_step or
                   not (0.0 <= start_value <= 1.0) or
                   not (0.0 <= end_value <= 1.0)):
                    raise ValueError('Some schedule sampling parameters incorrect.')

                # TODO: Check arguments validity, add default values (as optional arguments)

                def linear_policy(step):
                    if step < start_step:
                        threshold = start_value
                    elif start_step <= step < end_step:
                        slope = (start_value-end_value)/(start_step-end_step)  # < 0 (because end_step>start_step and start_value>end_value)
                        threshold = slope*(step-start_step) + start_value
                    elif end_step <= step:
                        threshold = end_value
                    else:
                        raise RuntimeError('Invalid value for the sampling policy')  # Parameters have not been correctly defined!
                    assert 0.0 <= threshold <= 1.0
                    return threshold

                self.sampling_policy_fct = linear_policy
            else:
                raise ValueError('Unknown chosen schedule sampling policy: {}'.format(policy))

        def get_prev_threshold(self, glob_step, i=0):
            """ Return the previous sampling probability for the current step.
            If above, the RNN should use the previous step instead of the given input.
            Args:
                glob_step (int): the global iteration step for the training
                i (int): the timestep of the RNN (TODO: implement incrementive slope (progression like -\|), remove the '=0')
            """
            return self.sampling_policy_fct(glob_step)

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

        # Other options
        self.target_weights_policy = None
        self.schedule_policy = None

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
            loop_function=loop_rnn
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

            self.schedule_policy = Model.ScheduledSamplingPolicy(self.args)
            self.target_weights_policy = Model.TargetWeightsPolicy(self.args)  # Load the chosen policy

            # TODO: If train on different length, check that the loss is proportional to the length or average ???
            loss_fct = tf.nn.seq2seq.sequence_loss(  # Or sequence_loss_by_example ??
                self.outputs,
                self.targets,
                [tf.constant(self.target_weights_policy.get_weight(i), shape=self.targets[0].get_shape()) for i in range(len(self.targets))],  # Weights
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
    
    def step(self, batch, train_set=True, glob_step=-1):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
            train_set (Bool): indicate if the batch come from the test/train set
            glob_step (int): indicate the global step for the schedule sampling
        Return:
            (ops), dict: A tuple of the (training_step,) operator or (outputs,) in testing mode with the associated feed dictionary
        """
        # TODO: Could optimize feeding between train/test/generating (compress code)

        # Feed the dictionary
        feed_dict = {}
        ops = None

        if not self.args.test:  # Training
            # Feed placeholder
            for i in range(self.args.sample_length):
                feed_dict[self.inputs[i]] = batch.inputs[i]
                feed_dict[self.targets[i]] = batch.targets[i]
                #if not train_set or np.random.rand() > self.schedule_policy.get_prev_threshold(glob_step)*self.target_weights_policy.get_weight(i):  # Regular Schedule sample (TODO: Try sampling with the weigths or a mix of weights/sampling)
                if not train_set or np.random.rand() > self.schedule_policy.get_prev_threshold(glob_step):  # Weight the threshold by the target weights (don't schedule sample if weight=0)
                    feed_dict[self.use_prev[i]] = True
                else:
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
