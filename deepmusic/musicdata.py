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
import numpy as np  # Batch data
# TODO: import cv2  # Plot the piano roll

from deepmusic.midireader import MidiReader
from deepmusic.midireader import MidiInvalidException
import deepmusic.songstruct as music


class Batch:
    """Structure containing batches info
    """
    def __init__(self):
        self.inputs = []
        self.targets = []


class MusicData:
    """Dataset class
    """
    
    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Filename and directories constants
        self.DATA_VERSION = '0.2'  # Assert compatibility between versions
        self.DATA_DIR_MIDI = 'data/midi'  # Originals midi files
        self.DATA_DIR_SAMPLES = 'data/samples'  # Training/testing samples after pre-processing
        self.DATA_SAMPLES_EXT = '.pkl'
        self.FILE_EXT = '.mid'  # Could eventually add support for other format later ?

        # Define the time unit
        # Invert of time note which define the maximum resolution for a song. Ex: 2 for 1/2 note, 4 for 1/4 of note
        # TODO: Where to define ? Should be in self.args.
        self.MAXIMUM_SONG_RESOLUTION = 4
        self.NOTES_PER_BAR = 4

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
            self._save_samples(samples_path)

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
            current_version = data['version']
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

        for filename in tqdm(midi_files):

            tqdm.write('')
            tqdm.write('Parsing {}'.format(filename))

            try:
                new_song = MidiReader.load_file(filename)
            except MidiInvalidException as e:
                tqdm.write('File ignored: {}'.format(e))
            else:
                self.songs.append(self._convert_song2array(new_song))

        if not self.songs:
            raise ValueError('Empty dataset. Check that the folder exist and contains supported midi files.')

        pass

    def _convert_song2array(self, song):
        """ Convert a given song to a numpy multi-dimensional array (piano roll)
        The song is temporally normalized, meaning that all ticks and duration will be converted to a specific
        ticks_per_beat independent unit.
        For now, the changes of tempo are ignored. Only 4/4 is supported.
        Warning: The duration is ignored: All note have the same duration (1 unit)
        Args:
            song (Song): The song to convert
        Return:
            Array: the numpy array: a binary matrix of shape [NB_NOTES, song_length]
        """

        # Convert the absolute ticks in standardized unit
        song_length = len(song)
        scale = self._get_scale(song)

        # TODO: Not sure why this plot a decimal value (x.66). Investigate...
        # print(song_length/scale)

        # Use sparse array instead ?
        piano_roll = np.zeros([music.NB_NOTES, int(np.ceil(song_length/scale))], dtype=int)

        # Adding all notes
        for track in song.tracks:
            for note in track.notes:
                piano_roll[note.get_relative_note()][note.tick//scale] = 1

        return piano_roll

    def _convert_array2song(self, array):
        """ Create a new song from a numpy array
        A note will be created for each non empty case of the array. The song will contain a single track, and use the
        default beats_per_tick as midi resolution
        For now, the changes of tempo are ignored. Only 4/4 is supported.
        Warning: All note have the same duration, the default value defined in music.Note
        Args:
            np.array: the numpy array (Warning: could be a array of int or float containing the prediction before the sigmoid)
        Return:
            song (Song): The song to convert
        """

        new_song = music.Song()
        main_track = music.Track()

        scale = self._get_scale(new_song)

        for index, x in np.ndenumerate(array):  # Add some notes
            if x > 1e-12:  # Note added (TODO: What should be the condition, =1 ? sigmoid>0.5 ?)
                new_note = music.Note()

                new_note.set_relative_note(index[0])
                new_note.tick = index[1] * scale  # Absolute time in tick from the beginning

                main_track.notes.append(new_note)

        new_song.tracks.append(main_track)

        return new_song

    def _get_scale(self, song):
        """ Compute the unit scale factor for the given song
        The scale factor allow to have a tempo independent time unit, to represent the song as an array
        of dimension [key, time_unit]. Once computed, one has just to divide (//) the ticks or multiply
        the time units to go from one representation to the other.

        Args:
            song (Song): a song object from which will be extracted the tempo information
        Return:
            int: the scale factor for the current song
        """

        # TODO: Assert that the scale factor is not a float (the % =0)
        return 4 * song.ticks_per_beat // (self.MAXIMUM_SONG_RESOLUTION*self.NOTES_PER_BAR)

    def get_batches(self):
        """Prepare the batches for the current epoch
        Return:
            List[Batch]: Get a list of the batches for the next epoch
        """
        batches = []

        # TODO: Create batches (randomly cut each song in some small parts (need to know the total length for that)
        # then create the big matrix (NB_NOTE*sample_length) and turn that into batch). If process too long,
        # could save the created batches in a new folder, data/samples or save/model.

        # First part: Randomly extract subsamples of the songs
        print("Subsampling songs...")

        sample_subsampling_length = self.args.sample_length+1  # We add 1 because each input has to predict the next output

        sub_songs = []
        for song in self.songs:
            len_song = song.shape[-1]  # The last dimension correspond to the song duration
            max_start = len_song - sample_subsampling_length
            assert max_start >= 0  # TODO: Error handling (and if =0, compatible with randint ?)
            nb_sample_song = 2*len_song // self.args.sample_length  # The number of subsample is proportional to the song length
            for _ in range(nb_sample_song):
                start = np.random.randint(max_start)
                sub_song = song[:, start:start+sample_subsampling_length]
                sub_songs.append(sub_song)

        # Second part: Shuffle the song extracts
        print("Shuffling the dataset...")
        np.random.shuffle(sub_songs)

        # Third part: Group the samples together to create the batches
        print("Generating batches...")

        def gen_next_samples():
            """ Generator over the mini-batch training samples
            Warning: the last samples will be ignored if the number of batch does not match the number of samples
            """
            nb_samples = len(sub_songs)
            for i in range(nb_samples//self.args.batch_size):
                yield sub_songs[i*self.args.batch_size:(i+1)*self.args.batch_size]

        for samples in gen_next_samples():
            batch = Batch()

            # samples has shape [batch_size, NB_NOTES, sample_subsampling_length]
            assert len(samples) == self.args.batch_size
            assert samples[0].shape == (music.NB_NOTES, sample_subsampling_length)

            # Define targets and inputs
            for i in range(self.args.sample_length):
                input = -np.ones([len(samples), music.NB_NOTES])
                target = np.zeros([len(samples), music.NB_NOTES])
                for j, sample in enumerate(samples):  # len(samples) == self.args.batch_size
                    # TODO: Could reuse boolean idx computed (from target to next input)
                    input[j, sample[:, i] == 1] = 1.0
                    target[j, sample[:, i+1] == 1] = 1.0

                batch.inputs.append(input)
                batch.targets.append(target)

            batches.append(batch)
        
        # Use tf.train.batch() ??

        # TODO: Save some batches as midi to see if correct

        return batches

    def get_batch_test(self):
        """ Return an initial batch as initiator for the RNN
        Only the initial position is defined
        """
        batch = Batch()

        input = None
        size_input = [music.NB_NOTES,]
        for i in range(self.args.batch_size):
            if i == 0:
                input = [-np.ones(size_input)]  # No key pressed
            elif i == 1:
                input = np.append(input, [np.ones(size_input)], axis=0)  # All key pressed
            elif i < 4:
                input = np.append(input, [np.random.uniform(-1.0, 1.0, size=size_input)], axis=0)
            elif i < 8:
                input = np.append(input, [np.random.normal(size=size_input)], axis=0)
            else:
                input = np.append(input, [np.random.normal(-0.5, size=size_input)], axis=0)

        batch.inputs.append(input)

        return batch

    def convert_to_songs(self, outputs):
        """ Create songs from the decoder outputs.
        Args:
            outputs (List[np.array]): The list of the predictions of the decoder
        Return:
            List[Song]: the list of the songs (one song by batch)
        """

        # Step 1: Extract the batches and recreate the array for each batch
        piano_rolls = []
        for i in range(outputs[0].shape[0]):  # Iterate over the batches
            piano_roll = None
            for j in range(len(outputs)):  # Iterate over the sample length
                # outputs[j][i, :] has shape [NB_NOTES, 1]
                if piano_roll is None:
                    piano_roll = [outputs[j][i, :]]
                else:
                    piano_roll = np.append(piano_roll, [outputs[j][i, :]], axis=0)
            piano_rolls.append(piano_roll.T)

        # Step 2: Create the song from the array
        songs = []
        for array in piano_rolls:
            songs.append(self._convert_array2song(array))

        return songs
