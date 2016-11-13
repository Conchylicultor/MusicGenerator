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
The learning rate policy control the evolution of the learning rate during
the training
"""


class LearningRatePolicy:
    """ Contains the different policies for the learning rate decay
    """
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """

    def get_learning_rate(self, glob_step):
        """ Return the learning rate associated at the current training step
        Args:
            glob_step (int): Number of iterations since the beginning of training
        Return:
            float: the learning rate at the given step
        """
        raise NotImplementedError('Abstract class')


class LearningRatePolicyOld:
    """ Contains the different policies for the learning rate decay
    """
    CST = 'cst'  # Fixed learning rate over all steps (default behavior)
    STEP = 'step'  # We divide the learning rate every x iterations
    EXPONENTIAL = 'exponential'  #

    @staticmethod
    def get_policies():
        """ Return the list of the different modes
        Useful when parsing the command lines arguments
        """
        return [
            LearningRatePolicy.CST,
            LearningRatePolicy.STEP,
            LearningRatePolicy.EXPONENTIAL
        ]

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        self.learning_rate_fct = None

        assert args.learning_rate
        assert len(args.learning_rate) > 0

        policy = args.learning_rate[0]

        if policy == LearningRatePolicy.CST:
            if not len(args.learning_rate) == 2:
                raise ValueError(
                    'Learning rate cst policy should be on the form: {} lr_value'.format(Model.LearningRatePolicy.CST))
            self.learning_rate_init = float(args.learning_rate[1])
            self.learning_rate_fct = self._lr_cst

        elif policy == LearningRatePolicy.STEP:
            if not len(args.learning_rate) == 3:
                raise ValueError('Learning rate step policy should be on the form: {} lr_init decay_period'.format(
                    LearningRatePolicy.STEP))
            self.learning_rate_init = float(args.learning_rate[1])
            self.decay_period = int(args.learning_rate[2])
            self.learning_rate_fct = self._lr_step

        else:
            raise ValueError('Unknown chosen learning rate policy: {}'.format(policy))

    def _lr_cst(self, glob_step):
        """ Just a constant learning rate
        """
        return self.learning_rate_init

    def _lr_step(self, glob_step):
        """ Every decay period, the learning rate is divided by 2
        """
        return self.learning_rate_init / 2 ** (glob_step // self.decay_period)

    def get_learning_rate(self, glob_step):
        """ Return the learning rate associated at the current training step
        Args:
            glob_step (int): Number of iterations since the beginning of training
        Return:
            float: the learning rate at the given step
        """
        return self.learning_rate_fct(glob_step)


class Cst(LearningRatePolicy):
    """ Fixed learning rate over all steps (default behavior)
    """
    @staticmethod
    def get_module_id():
        return 'cst'

    def __init__(self, args, lr=0.0001):
        """
        Args:
            args: parameters of the model
        """
        self.lr = lr

    def get_learning_rate(self, glob_step):
        """ Return the learning rate associated at the current training step
        Args:
            glob_step (int): Number of iterations since the beginning of training
        Return:
            float: the learning rate at the given step
        """
        return self.lr


class StepsWithDecay(LearningRatePolicy):
    """
    """

    @staticmethod
    def get_module_id():
        return 'step'


class Adaptive(LearningRatePolicy):
    """ Decrease the learning rate when training error
    reach a plateau
    """

    @staticmethod
    def get_module_id():
        return 'adaptive'
