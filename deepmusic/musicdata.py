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
Loads the midi song, build the dataset
"""

from tqdm import tqdm  # Progress bar when creating dataset
import pickle  # Saving the data
import os  # Checking file existence
import random  # When shuffling

from deepmusic.midireader import MidiReader


class Batch:
    """Structure containing batches info
    """
    def __init__(self):
        pass


class MusicData:
    """Dataset class
    """
    
    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Filename and directories constants
        self.DATA_VERSION = '0.1'  # Assert compatibility between versions
        self.DATA_DIR_MIDI = 'data/midi'  # Originals midi files
        self.DATA_DIR_SAMPLES = 'data/samples'  # Training/testing samples after pre-processing
        self.DATA_SAMPLES_EXT = '.pkl'
        self.FILE_EXT = '.mid'  # Could eventually add support for other format later ?

        # Model parameters
        self.args = args

        # Dataset
        self.songs = []
        
        self._restore_dataset()

        # Plot some stats:
        print('Loaded: {} songs'.format(len(self.songs)))  # TODO: Print average, max, min duration

        if self.args.play_dataset:
            # TODO: Display some images corresponding to the loaded songs
            raise NotImplementedError('Can\'t play a song for now')

    def _restore_dataset(self):
        """Load/create the conversations data
        """

        # Construct the dataset name
        samples_path = os.path.join(
            self.args.root_dir,
            self.DATA_DIR_SAMPLES,
            self.args.dataset_tag + self.DATA_SAMPLES_EXT
        )

        # Restoring precomputed model
        if os.path.exists(samples_path):
            print('Restoring dataset from {}...'.format(samples_path))
            self._restore_samples(samples_path)

        # First time we load the database: creating all files
        else:
            print('Training samples not found. Creating dataset...')
            self._create_samples()

            print('Saving dataset...')
            #self._save_samples(samples_path)

    def _restore_samples(self, samples_path):
        """ Load samples from file
        Args:
            samples_path (str): The path where to load the model (all dirs should exist)
        Return:
            List[Song]: The training data
        """
        with open(samples_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset

            # Check the version
            current_version = data('version')
            if current_version != self.DATA_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}.'.format(current_version, self.DATA_VERSION))

            # Restore parameters
            self.songs = data['songs']

    def _save_samples(self, samples_path):
        """ Save samples to file
        Args:
            samples_path (str): The path where to save the model (all dirs should exist)
        """

        with open(samples_path, 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'version': self.DATA_VERSION,
                'songs': self.songs
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def _create_samples(self):
        """ Create the database from the midi files
        """
        midi_dir = os.path.join(self.args.root_dir, self.DATA_DIR_MIDI, self.args.dataset_tag)
        midi_files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(self.FILE_EXT)]

        for file in tqdm(midi_files):

            tqdm.write('')
            tqdm.write('Parsing {}'.format(file))

            midi_song = MidiReader(file)
            self.songs.append(midi_song)  # TODO: Only add if valid


        # TODO: Assert midi dir exist/ contains song
        pass

    def get_batches(self):
        """Prepare the batches for the current epoch
        Return:
            List[Batch]: Get a list of the batches for the next epoch
        """
        print("Shuffling the dataset...")
        random.shuffle(self.songs)  # TODO: Segment the songs ?
        
        batches = []
        
        # Use tf.train.batch() ??

        return batches
