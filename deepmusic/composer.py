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
Music composer. Act as the coordinator. Orchestrate and call the different models, see the readme for more details.

Use python 3
"""

import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
from tqdm import tqdm  # Progress bar
import tensorflow as tf
import gc  # Manual garbage collect before each epoch

from deepmusic.moduleloader import ModuleLoader
from deepmusic.musicdata import MusicData
from deepmusic.midiconnector import MidiConnector
from deepmusic.imgconnector import ImgConnector
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
        INTERACTIVE = 'interactive'  # The user start a melodie and the neural network complete (Not implemented)

        @staticmethod
        def get_test_modes():
            """ Return the list of the different testing modes
            Useful on when parsing the command lines arguments
            """
            return [Composer.TestMode.ALL, Composer.TestMode.DAEMON, Composer.TestMode.INTERACTIVE]

    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = None

        # Task specific objects
        self.music_data = None  # Dataset
        self.model = None  # Base model class

        # TensorFlow utilities for convenience saving/logging
        self.writer = None
        self.writer_test = None
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
        self.CONFIG_VERSION = '0.5'  # Ensure to raise a warning if there is a change in the format

        self.TRAINING_VISUALIZATION_STEP = 1000  # Plot a training sample every x iterations (Warning: There is a really low probability that on a epoch, it's always the same testing bach which is visualized)
        self.TRAINING_VISUALIZATION_DIR = 'progression'
        self.TESTING_VISUALIZATION_DIR = 'midi'  # Would 'generated', 'output' or 'testing' be a best folder name ?

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
        global_args.add_argument('--keep_all', action='store_true', help='if this option is set, all saved model will be keep (Warning: make sure you have enough free disk space or increase save_every)')  # TODO: Add an option to delimit the max size
        global_args.add_argument('--model_tag', type=str, default=None, help='tag to differentiate which model to store/load')
        global_args.add_argument('--sample_length', type=int, default=40, help='number of time units (steps) of a training sentence, length of the sequence to generate')  # Warning: the unit is defined by the MusicData.MAXIMUM_SONG_RESOLUTION parameter
        global_args.add_argument('--root_dir', type=str, default=None, help='folder where to look for the models and data')
        global_args.add_argument('--device', type=str, default=None, help='\'gpu\' or \'cpu\' (Warning: make sure you have enough free RAM), allow to choose on which hardware run the model')
        global_args.add_argument('--temperature', type=float, default=1.0, help='Used when testing, control the ouput sampling')

        # Dataset options
        dataset_args = parser.add_argument_group('Dataset options')
        dataset_args.add_argument('--dataset_tag', type=str, default='ragtimemusic', help='tag to differentiate which data use (if the data are not present, the program will try to generate from the midi folder)')
        dataset_args.add_argument('--create_dataset', action='store_true', help='if present, the program will only generate the dataset from the corpus (no training/testing)')
        dataset_args.add_argument('--play_dataset', type=int, nargs='?', const=10, default=None,  help='if set, the program  will randomly play some samples(can be use conjointly with create_dataset if this is the only action you want to perform)')  # TODO: Play midi ? / Or show sample images ? Both ?
        dataset_args.add_argument('--ratio_dataset', type=float, default=0.9, help='ratio of songs between training/testing. The ratio is fixed at the beginning and cannot be changed')
        ModuleLoader.batch_builders.add_argparse(dataset_args, 'Control the song representation for the inputs of the neural network.')

        # Network options (Warning: if modifying something here, also make the change on save/restore_params() )
        nn_args = parser.add_argument_group('Network options', 'architecture related option')
        ModuleLoader.enco_cells.add_argparse(nn_args, 'Encoder cell used.')
        ModuleLoader.deco_cells.add_argparse(nn_args, 'Decoder cell used.')
        nn_args.add_argument('--hidden_size', type=int, default=512, help='Size of one neural network layer')
        nn_args.add_argument('--num_layers', type=int, default=2, help='Nb of layers of the RNN')
        nn_args.add_argument('--scheduled_sampling', type=str, nargs='+', default=[Model.ScheduledSamplingPolicy.NONE], help='Define the schedule sampling policy. If set, have to indicates the parameters of the chosen policy')
        nn_args.add_argument('--target_weights', nargs='?', choices=Model.TargetWeightsPolicy.get_policies(), default=Model.TargetWeightsPolicy.LINEAR,
                             help='policy to choose the loss contribution of each step')
        ModuleLoader.loop_processings.add_argparse(nn_args, 'Transformation to apply on each ouput.')

        # Training options (Warning: if modifying something here, also make the change on save/restore_params() )
        training_args = parser.add_argument_group('Training options')
        training_args.add_argument('--num_epochs', type=int, default=0, help='maximum number of epochs to run (0 for infinity)')
        training_args.add_argument('--save_every', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        training_args.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
        ModuleLoader.learning_rate_policies.add_argparse(training_args, 'Learning rate initial value and decay policy.')
        training_args.add_argument('--testing_curve', type=int, default=10, help='Also record the testing curve each every x iteration (given by the parameter)')

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

        ModuleLoader.register_all()  # Load available modules
        self.args = self._parse_args(args)
        if not self.args.root_dir:
            self.args.root_dir = os.getcwd()  # Use the current working directory

        self._restore_params()  # Update the self.model_dir and self.glob_step, for now, not used when loading Model
        self._print_params()

        self.music_data = MusicData(self.args)
        if self.args.create_dataset:
            print('Dataset created! You can start training some models.')
            return  # No need to go further

        with tf.device(self._get_device()):
            self.model = Model(self.args)

        # Saver/summaries
        self.writer = tf.train.SummaryWriter(os.path.join(self.model_dir, 'train'))
        self.writer_test = tf.train.SummaryWriter(os.path.join(self.model_dir, 'test'))
        self.saver = tf.train.Saver(max_to_keep=200)  # Set the arbitrary limit ?

        # TODO: Fixed seed (WARNING: If dataset shuffling, make sure to do that after saving the
        # dataset, otherwise, all what comes after the shuffling won't be replicable when
        # reloading the dataset). How to restore the seed after loading ?? (with get_state/set_state)
        # Also fix seed for np.random (does it works globally for all files ?)

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
            print('The End! Thanks for using this program')

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
            e = 0
            while self.args.num_epochs == 0 or e < self.args.num_epochs:  # Main training loop (infinite if num_epoch==0)
                e += 1

                print()
                print('------- Epoch {} (lr={}) -------'.format(
                    '{}/{}'.format(e, self.args.num_epochs) if self.args.num_epochs else '{}'.format(e),
                    self.model.learning_rate_policy.get_learning_rate(self.glob_step))
                )

                # Explicit garbage collector call (clear the previous batches)
                gc.collect()  # TODO: Better memory management (use generators,...)

                batches_train, batches_test = self.music_data.get_batches()

                # Also update learning parameters eventually ?? (Some is done in the model class with the policy classes)

                tic = datetime.datetime.now()
                for next_batch in tqdm(batches_train, desc='Training'):  # Iterate over the batches
                    # TODO: Could compute the perfs (time forward pass vs time batch pre-processing)
                    # Indicate if the output should be computed or not
                    is_output_visualized = self.glob_step % self.TRAINING_VISUALIZATION_STEP == 0

                    # Training pass
                    ops, feed_dict = self.model.step(
                        next_batch,
                        train_set=True,
                        glob_step=self.glob_step,
                        ret_output=is_output_visualized
                    )
                    outputs_train = self.sess.run((merged_summaries,) + ops, feed_dict)
                    self.writer.add_summary(outputs_train[0], self.glob_step)

                    # Testing pass (record the testing curve and visualize some testing predictions)
                    # TODO: It makes no sense to completely disable the ground truth feeding (it's impossible to the
                    # network to do a good prediction with only the first step)
                    if is_output_visualized or (self.args.testing_curve and self.glob_step % self.args.testing_curve == 0):
                        next_batch_test = batches_test[self.glob_step % len(batches_test)]  # Generate test batches in a cycling way (test set smaller than train set)
                        ops, feed_dict = self.model.step(
                            next_batch_test,
                            train_set=False,
                            ret_output=is_output_visualized
                        )
                        outputs_test = self.sess.run((merged_summaries,) + ops, feed_dict)
                        self.writer_test.add_summary(outputs_test[0], self.glob_step)

                    # Some visualisation (we compute some training/testing samples and compare them to the ground truth)
                    if is_output_visualized:
                        visualization_base_name = os.path.join(self.model_dir, self.TRAINING_VISUALIZATION_DIR, str(self.glob_step))
                        tqdm.write('Visualizing: ' + visualization_base_name)
                        self._visualize_output(
                            visualization_base_name,
                            outputs_train[-1],
                            outputs_test[-1]  # The network output will always be the last operator returned by model.step()
                        )

                    # Checkpoint
                    self.glob_step += 1  # Iterate here to avoid saving at the first iteration
                    if self.glob_step % self.args.save_every == 0:
                        self._save_session(self.sess)

                toc = datetime.datetime.now()

                print('Epoch finished in {}'.format(toc-tic))  # Warning: Will overflow if an epoch takes more than 24 hours, and the output isn't really nicer
        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        self._save_session(self.sess)  # Ultimate saving before complete exit

    def _main_test(self):
        """ Generate some songs
        The midi files will be saved on the same model_dir
        """
        assert self.sess
        assert self.args.batch_size == 1

        print('Start predicting...')

        model_list = self._get_model_list()
        if not model_list:
            print('Warning: No model found in \'{}\'. Please train a model before trying to predict'.format(self.model_dir))
            return

        batches, names = self.music_data.get_batches_test_old()
        samples = list(zip(batches, names))

        # Predicting for each model present in modelDir
        for model_name in tqdm(sorted(model_list), desc='Model', unit='model'):  # TODO: Natural sorting / TODO: tqdm ?
            self.saver.restore(self.sess, model_name)

            for next_sample in tqdm(samples, desc='Generating ({})'.format(os.path.basename(model_name)), unit='songs'):
                batch = next_sample[0]
                name = next_sample[1]  # Unzip

                ops, feed_dict = self.model.step(batch)
                assert len(ops) == 2  # sampling, output
                chosen_labels, outputs = self.sess.run(ops, feed_dict)

                model_dir, model_filename = os.path.split(model_name)
                model_dir = os.path.join(model_dir, self.TESTING_VISUALIZATION_DIR)
                model_filename = model_filename[:-len(self.MODEL_EXT)] + '-' + name

                # Save piano roll as image (color map red/blue to see the prediction confidence)
                # Save the midi file
                self.music_data.visit_recorder(
                    outputs,
                    model_dir,
                    model_filename,
                    [ImgConnector, MidiConnector],
                    chosen_labels=chosen_labels
                )
                # TODO: Print song statistics (nb of generated notes, closest songs in dataset ?, try to compute a
                # score to indicate potentially interesting songs (low score if too repetitive) ?,...). Create new
                # visited recorder class ?
                # TODO: Include infos on potentially interesting songs (include metric in the name ?), we should try to detect
                # the loops, simple metric: nb of generated notes, nb of unique notes (Metric: 2d
                # tensor [NB_NOTES, nb_of_time_the_note_is played], could plot histogram normalized by nb of
                # notes). Is piano roll enough ?

        print('Prediction finished, {} songs generated'.format(self.args.batch_size * len(model_list) * len(batches)))

    def _visualize_output(self, visualization_base_name, outputs_train, outputs_test):
        """ Record some result/generated songs during training.
        This allow to see the training progression and get an idea of what the network really learned
        Args:
            visualization_base_name (str):
            outputs_train: Output of the forward pass(training set)
            outputs_test: Output of the forward pass (testing set)
        """
        # Record:
        # * Training/testing:
        #   * Prediction/ground truth:
        #     * piano roll
        #     * midi file
        # Format name: <glob_step>-<train/test>-<pred/truth>-<mini_batch_id>.<png/mid>
        # TODO: Also records the ground truth

        model_dir, model_filename = os.path.split(visualization_base_name)
        for output, set_name in [(outputs_train, 'train'), (outputs_test, 'test')]:
            self.music_data.visit_recorder(
                output,
                model_dir,
                model_filename + '-' + set_name,
                [ImgConnector, MidiConnector]
            )

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
                print('No previous model found, but some files/folders found at {}. Cleaning...'.format(self.model_dir))  # Warning: No confirmation asked
                self.args.reset = True

            if self.args.reset:
                # WARNING: No confirmation is asked. All subfolders will be deleted
                for root, dirs, files in os.walk(self.model_dir, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        print('Removing {}'.format(file_path))
                        os.remove(file_path)
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
            self.args.keep_all = config['General'].getboolean('keep_all')
            self.args.dataset_tag = config['General'].get('dataset_tag')
            if not self.args.test:  # When testing, we don't use the training length
                self.args.sample_length = config['General'].getint('sample_length')

            self.args.hidden_size = config['Network'].getint('hidden_size')
            self.args.num_layers = config['Network'].getint('num_layers')
            self.args.target_weights = config['Network'].get('target_weights')
            self.args.scheduled_sampling = config['Network'].get('scheduled_sampling').split(' ')

            self.args.batch_size = config['Training'].getint('batch_size')
            self.args.save_every = config['Training'].getint('save_every')
            self.args.ratio_dataset = config['Training'].getfloat('ratio_dataset')
            self.args.testing_curve = config['Training'].getint('testing_curve')

            ModuleLoader.load_all(self.args, config)

            # Show the restored params
            print('Warning: Restoring parameters from previous configuration (you should manually edit the file if you want to change one of those)')

        # When testing, only predict one song at the time
        if self.args.test:
            self.args.batch_size = 1
            self.args.scheduled_sampling = [Model.ScheduledSamplingPolicy.NONE]

    def _save_params(self):
        """ Save the params of the model, like the current glob_step value
        Warning: if you modify this function, make sure the changes mirror load_params
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version'] = self.CONFIG_VERSION
        config['General']['glob_step'] = str(self.glob_step)
        config['General']['keep_all'] = str(self.args.keep_all)
        config['General']['dataset_tag'] = self.args.dataset_tag
        config['General']['sample_length'] = str(self.args.sample_length)

        config['Network'] = {}
        config['Network']['hidden_size'] = str(self.args.hidden_size)
        config['Network']['num_layers'] = str(self.args.num_layers)
        config['Network']['target_weights'] = self.args.target_weights  # Could be modified manually
        config['Network']['scheduled_sampling'] = ' '.join(self.args.scheduled_sampling)

        # Keep track of the learning params (are not model dependent so can be manually edited)
        config['Training'] = {}
        config['Training']['batch_size'] = str(self.args.batch_size)
        config['Training']['save_every'] = str(self.args.save_every)
        config['Training']['ratio_dataset'] = str(self.args.ratio_dataset)
        config['Training']['testing_curve'] = str(self.args.testing_curve)

        # Save the chosen modules and their configuration
        ModuleLoader.save_all(config)

        with open(os.path.join(self.model_dir, self.CONFIG_FILENAME), 'w') as config_file:
            config.write(config_file)

    def _print_params(self):
        """ Print the current params
        """
        print()
        print('Current parameters:')
        print('glob_step: {}'.format(self.glob_step))
        print('keep_all: {}'.format(self.args.keep_all))
        print('dataset_tag: {}'.format(self.args.dataset_tag))
        print('sample_length: {}'.format(self.args.sample_length))

        print('hidden_size: {}'.format(self.args.hidden_size))
        print('num_layers: {}'.format(self.args.num_layers))
        print('target_weights: {}'.format(self.args.target_weights))
        print('scheduled_sampling: {}'.format(' '.join(self.args.scheduled_sampling)))

        print('batch_size: {}'.format(self.args.batch_size))
        print('save_every: {}'.format(self.args.save_every))
        print('ratio_dataset: {}'.format(self.args.ratio_dataset))
        print('testing_curve: {}'.format(self.args.testing_curve))

        ModuleLoader.print_all(self.args)
        print()

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
        elif self.args.device == 'gpu':  # Won't work in case of multiple GPUs
            return '/gpu:0'
        elif self.args.device is None:  # No specified device (default)
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
