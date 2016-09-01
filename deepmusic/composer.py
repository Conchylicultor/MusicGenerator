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
Music composer. Act as the coordinator. Orchestrate and call the different models, see the readme for more details.

Use python 3
"""

import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
from typing import Dict, Tuple, List

from tqdm import tqdm  # Progress bar
import tensorflow as tf

from deepmusic.musicdata import MusicData
from deepmusic.model import Model


class Composer:
    """
    Main class which launch the training or testing mode
    """

    class TestMode:
        """ Simple structure representing the different testing modes
        """
        ALL = 'all'  # The network try to generate a new original composition with all models present (with the tag)
        DAEMON = 'daemon'  # Runs on background and can regularly be called to predict something (Not implemented)

        @staticmethod
        def get_test_modes() -> List[str]:
            """ Return the list of the different testing modes
            Useful on when parsing the command lines arguments
            """
            return [Composer.TestMode.ALL, Composer.TestMode.DAEMON]

    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific object
        self.music_data = None  # Dataset
        self.model = None  # Sequence to sequence model

        # TensorFlow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.model_dir = ''  # Where the model is saved
        self.glob_step = 0  # Represent the number of iteration for the current model

        # TensorFlow main session (we keep track for the daemon)
        self.sess = None

        # Filename and directories constants
        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.1'  # Ensure to raise a warning if there is a change in the format

    @staticmethod
    def _parse_args(args):
        """
        Parse the arguments from the given command line
        Args:
            args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
        """

        parser = argparse.ArgumentParser()

        # Global options
        global_args = parser.add_argument_group('Global options')
        global_args.add_argument('--test', nargs='?', choices=Composer.TestMode.get_test_modes(), const=Composer.TestMode.ALL, default=None,
                                 help='if present, launch the program try to answer all sentences from data/test/ with'
                                      ' the defined model(s), in interactive mode, the user can wrote his own sentences,'
                                      ' use daemon mode to integrate the chatbot in another program')
        global_args.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        global_args.add_argument('--keep_all', action='store_true', help='If this option is set, all saved model will be keep (Warning: make sure you have enough free disk space or increase save_every)')  # TODO: Add an option to delimit the max size
        global_args.add_argument('--model_tag', type=str, default=None, help='tag to differentiate which model to store/load')
        global_args.add_argument('--root_dir', type=str, default=None, help='folder where to look for the models and data')
        global_args.add_argument('--device', type=str, default=None, help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')

        # Dataset options
        dataset_args = parser.add_argument_group('Dataset options')
        dataset_args.add_argument('--dataset_tag', type=str, default='joplin', help='tag to differentiate which data use (if the data are not present, the program will try to generate from the midi folder)')
        dataset_args.add_argument('--create_dataset', action='store_true', help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        dataset_args.add_argument('--play_dataset', type=int, nargs='?', const=10, default=None,  help='if set, the program  will randomly play some samples(can be use conjointly with create_dataset if this is the only action you want to perform)')  # TODO: Play midi ? / Or show sample images ? Both ?
        dataset_args.add_argument('--ratio_dataset', type=float, default=1.0, help='ratio of dataset to separate training/testing')  # Not implemented, here maybe useless because we would like to overfit

        # Network options (Warning: if modifying something here, also make the change on save/restore_params() )
        nn_args = parser.add_argument_group('Network options', 'architecture related option')

        # Training options (Warning: if modifying something here, also make the change on save/restore_params() )
        training_args = parser.add_argument_group('Training options')
        training_args.add_argument('--num_epochs', type=int, default=30, help='maximum number of epochs to run')
        training_args.add_argument('--save_every', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        training_args.add_argument('--batch_size', type=int, default=10, help='mini-batch size')
        training_args.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

        return parser.parse_args(args)

    def main(self, args=None):
        """
        Launch the training and/or the interactive mode
        """
        print('Welcome to DeepMusic v0.1 !')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # General initialisations

        tf.logging.set_verbosity(tf.logging.INFO)  # DEBUG, INFO, WARN (default), ERROR, or FATAL

        self.args = self._parse_args(args)
        if not self.args.root_dir:
            self.args.root_dir = os.getcwd()  # Use the current working directory

        self._restore_params()  # Update the self.model_dir and self.glob_step, for now, not used when loading Model

        self.music_data = MusicData(self.args)
        if self.args.create_dataset:
            print('Dataset created! You can now try to train some models.')
            return  # No need to go further

        with tf.device(self._get_device()):
            self.model = Model(self.args)

        # Saver/summaries
        self.writer = tf.train.SummaryWriter(self.model_dir)
        self.saver = tf.train.Saver(max_to_keep=200)  # Arbitrary limit ?

        # TODO: Fixed seed (WARNING: If dataset shuffling, make sure to do that after saving the
        # dataset, otherwise, all what comes after the shuffling won't be replicable when
        # reloading the dataset). How to restore the seed after loading ??
        # Also fix seed for random.shuffle (does it works globally for all files ?)

        # Running session

        self.sess = tf.Session()

        print('Initialize variables...')
        self.sess.run(tf.initialize_all_variables())

        # Reload the model eventually (if it exist), on testing mode, the models are not loaded here (but in main_test())
        self._restore_previous_model(self.sess)

        if self.args.test:
            if self.args.test == Composer.TestMode.ALL:
                self._main_test()
            elif self.args.test == Composer.TestMode.DAEMON:
                print('Daemon mode, running in background...')
                raise NotImplementedError('No daemon mode')  # Come back later
            else:
                raise RuntimeError('Unknown test mode: {}'.format(self.args.test))  # Should never happen
        else:
            self._main_train()

        if self.args.test != Composer.TestMode.DAEMON:
            self.sess.close()
            print("The End! Thanks for using this program")

    def _main_train(self):
        """ Training loop
        """
        assert self.sess

        # Specific training dependent loading (Warning: When restoring a model, we don't restore the progression
        # bar, nor the current batch.)

        merged_summaries = tf.merge_all_summaries()
        if self.glob_step == 0:  # Not restoring from previous run
            self.writer.add_graph(self.sess.graph)  # First time only

        print('Start training (press Ctrl+C to save and exit)...')

        try:  # If the user exit while training, we still try to save the model
            for e in range(self.args.num_epochs):

                print()
                print("------- Epoch {}/{} (lr={}) -------".format(e+1, self.args.num_epochs, self.args.learning_rate))

                batches = self.music_data.get_batches()

                # Also update learning parameters eventually ??

                tic = datetime.datetime.now()
                for next_batch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feed_dict = self.model.step(next_batch)
                    assert len(ops) == 1  # training
                    summary, _ = self.sess.run((merged_summaries,) + ops, feed_dict)
                    self.writer.add_summary(summary, self.glob_step)
                    self.glob_step += 1

                    # Checkpoint
                    if self.glob_step % self.args.save_every == 0:
                        self._save_session(self.sess)

                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(toc-tic))  # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        self._save_session(self.sess)  # Ultimate saving before complete exit

    def _main_test(self):
        """ Generate some songs
        """
        assert self.sess

        print('Start predicting...')
        pass

    def _restore_previous_model(self, sess):
        """ Restore or reset the model, depending of the parameters
        If testing mode is set, the function has no effect
        If the destination directory already contains some file, it will handle the conflict as following:
         * If --reset is set, all present files will be removed (warning: no confirmation is asked) and the training
         restart from scratch (globStep & cie reinitialized)
         * Otherwise, it will depend of the directory content. If the directory contains:
           * No model files (only summary logs): works as a reset (restart from scratch)
           * Other model files, but model_name not found (surely keep_all option changed): raise error, the user should
           decide by himself what to do
           * The right model file (eventually some other): no problem, simply resume the training
        In any case, the directory will exist as it has been created by the summary writer
        Args:
            sess: The current running session
        """

        if self.args.test == Composer.TestMode.ALL:  # On testing, the models are not restored here
            return

        print('WARNING: ', end='')

        model_name = self._get_model_name()

        if os.listdir(self.model_dir):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.model_dir))
            # Analysing directory content
            elif os.path.exists(model_name):  # Restore the model
                print('Restoring previous model from {}'.format(model_name))
                self.saver.restore(sess, model_name)  # Will crash when --reset is not activated and the model has not been saved yet
                print('Model restored.')
            elif self._get_model_list():
                print('Conflict with previous models.')
                raise RuntimeError('Some models are already present in \'{}\'. You should check them first'.format(self.model_dir))
            else:  # No other model to conflict with (probably summary files)
                print('No previous model found, but some files found at {}. Cleaning...'.format(self.model_dir))  # Warning: No confirmation asked
                self.args.reset = True

            if self.args.reset:
                file_list = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
                for f in file_list:
                    print('Removing {}'.format(f))
                    os.remove(f)

        else:
            print('No previous model found, starting from clean directory: {}'.format(self.model_dir))

    def _save_session(self, sess):
        """ Save the model parameters and the variables
        Args:
            sess: the current session
        """
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        self._save_params()
        self.saver.save(sess, self._get_model_name())  # Put a limit size (ex: 3GB for the model_dir) ?
        tqdm.write('Model saved.')

    def _restore_params(self):
        """ Load the some values associated with the current model, like the current glob_step value.
        Needs to be called before any other function because it initialize some variables used on the rest of the
        program

        Warning: if you modify this function, make sure the changes mirror _save_params, also check if the parameters
        should be reset in manage_previous_model
        """
        # Compute the current model path
        self.model_dir = os.path.join(self.args.root_dir, self.MODEL_DIR_BASE)
        if self.args.model_tag:
            self.model_dir += '-' + self.args.model_tag

        # If there is a previous model, restore some parameters
        config_name = os.path.join(self.model_dir, self.CONFIG_FILENAME)
        if not self.args.reset and not self.args.create_dataset and os.path.exists(config_name):
            # Loading
            config = configparser.ConfigParser()
            config.read(config_name)

            # Check the version
            current_version = config['General'].get('version')
            if current_version != self.CONFIG_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(current_version, self.CONFIG_VERSION, config_name))

            # Restoring the the parameters
            self.glob_step = config['General'].getint('glob_step')

            # No restoring for training params, batch size or other non model dependent parameters

            # Show the restored params
            print()
            print('Warning: Restoring parameters:')
            print('glob_step: {}'.format(self.glob_step))
            print()

    def _save_params(self):
        """ Save the params of the model, like the current glob_step value
        Warning: if you modify this function, make sure the changes mirror load_params
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version'] = self.CONFIG_VERSION
        config['General']['glob_step'] = str(self.glob_step)
        
        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learning_rate'] = str(self.args.learning_rate)
        config['Training (won\'t be restored)']['batch_size'] = str(self.args.batch_size)

        with open(os.path.join(self.model_dir, self.CONFIG_FILENAME), 'w') as config_file:
            config.write(config_file)

    def _get_model_name(self):
        """ Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first time the model is load. If keep_all option is set, the
        glob_step value will be included in the name.
        Return:
            str: The path and name were the model need to be saved
        """
        model_name = os.path.join(self.model_dir, self.MODEL_NAME_BASE)
        if self.args.keep_all:  # We do not erase the previously saved model by including the current step on the name
            model_name += '-' + str(self.glob_step)
        return model_name + self.MODEL_EXT

    def _get_model_list(self):
        """ Return the list of the model files inside the model directory
        """
        return [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]

    def _get_device(self):
        """ Parse the argument to decide on which device run the model
        Return:
            str: The name of the device on which run the program
        """
        if self.args.device == 'cpu':
            return '"/cpu:0'
        elif self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
