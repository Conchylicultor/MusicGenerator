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
Loads the midi song, build the dataset
"""

from tqdm import tqdm  # Progress bar when creating dataset
import pickle  # Saving the data
import os  # Checking file existence
import numpy as np  # Batch data
import json  # Load initiators (inputs for generating new songs)

from deepmusic.moduleloader import ModuleLoader
from deepmusic.midiconnector import MidiConnector
from deepmusic.midiconnector import MidiInvalidException
import deepmusic.songstruct as music


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
        self.DATA_DIR_PLAY = 'data/play'  # Target folder to show the reconstructed files
        self.DATA_DIR_SAMPLES = 'data/samples'  # Training/testing samples after pre-processing
        self.DATA_SAMPLES_RAW = 'raw'  # Unpreprocessed songs container tag
        self.DATA_SAMPLES_EXT = '.pkl'
        self.TEST_INIT_FILE = 'data/test/initiator.json'  # Initial input for the generated songs
        self.FILE_EXT = '.mid'  # Could eventually add support for other format later ?

        # Model parameters
        self.args = args

        # Dataset
        self.songs = []
        self.songs_train = None
        self.songs_test = None

        # TODO: Dynamic loading of the the associated dataset flag (ex: data/samples/pianoroll/...)
        self.batch_builder = ModuleLoader.batch_builders.build_module(args)

        if not self.args.test:  # No need to load the dataset when testing
            self._restore_dataset()

            if self.args.play_dataset:
                print('Play some songs from the formatted data')
                # Generate songs
                for i in range(min(10, len(self.songs))):
                    raw_song = self.batch_builder.reconstruct_song(self.songs[i])
                    MidiConnector.write_song(raw_song, os.path.join(self.DATA_DIR_PLAY, str(i)))
                # TODO: Display some images corresponding to the loaded songs
                raise NotImplementedError('Can\'t play a song for now')

            self._split_dataset()  # Warning: the list order will determine the train/test sets (so important that it don't change from run to run)

            # Plot some stats:
            print('Loaded: {} songs ({} train/{} test)'.format(
                len(self.songs_train) + len(self.songs_test),
                len(self.songs_train),
                len(self.songs_test))
            )  # TODO: Print average, max, min duration

    def _restore_dataset(self):
        """Load/create the conversations data
        Done in two steps:
         * Extract the midi files as a raw song format
         * Transform this raw song as neural networks compatible input
        """

        # Construct the dataset names
        samples_path_generic = os.path.join(
            self.args.root_dir,
            self.DATA_DIR_SAMPLES,
            self.args.dataset_tag + '-{}' + self.DATA_SAMPLES_EXT
        )
        samples_path_raw = samples_path_generic.format(self.DATA_SAMPLES_RAW)
        samples_path_preprocessed = samples_path_generic.format(ModuleLoader.batch_builders.get_chosen_name())

        # TODO: the _restore_samples from the raw songs and precomputed database should have different versions number

        # Restoring precomputed database
        if os.path.exists(samples_path_preprocessed):
            print('Restoring dataset from {}...'.format(samples_path_preprocessed))
            self._restore_samples(samples_path_preprocessed)

        # First time we load the database: creating all files
        else:
            print('Training samples not found. Creating dataset from the songs...')
            # Restoring raw songs
            if os.path.exists(samples_path_raw):
                print('Restoring songs from {}...'.format(samples_path_raw))
                self._restore_samples(samples_path_raw)

            # First time we load the database: creating all files
            else:
                print('Raw songs not found. Extracting from midi files...')
                self._create_raw_songs()
                print('Saving raw songs...')
                self._save_samples(samples_path_raw)

            # At this point, self.songs contain the list of the raw songs. Each
            # song is then preprocessed by the batch builder

            # Generating the data from the raw songs
            print('Pre-processing songs...')
            for i, song in tqdm(enumerate(self.songs), total=len(self.songs)):
                self.songs[i] = self.batch_builder.process_song(song)

            print('Saving dataset...')
            np.random.shuffle(self.songs)  # Important to do that before saving so the train/test set will be fixed each time we reload the dataset
            self._save_samples(samples_path_preprocessed)

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

    def _create_raw_songs(self):
        """ Create the database from the midi files
        """
        midi_dir = os.path.join(self.args.root_dir, self.DATA_DIR_MIDI, self.args.dataset_tag)
        midi_files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith(self.FILE_EXT)]

        for filename in tqdm(midi_files):

            try:
                new_song = MidiConnector.load_file(filename)
            except MidiInvalidException as e:
                tqdm.write('File ignored ({}): {}'.format(filename, e))
            else:
                self.songs.append(new_song)
                tqdm.write('Song loaded {}: {} tracks, {} notes, {} ticks/beat'.format(
                    filename,
                    len(new_song.tracks),
                    sum([len(t.notes) for t in new_song.tracks]),
                    new_song.ticks_per_beat
                ))

        if not self.songs:
            raise ValueError('Empty dataset. Check that the folder exist and contains supported midi files.')

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

    def _split_dataset(self):
        """ Create the test/train set from the loaded songs
        The dataset has been shuffled when calling this function (Warning: the shuffling
        is done and fixed before saving the dataset the first time so it is important to
        NOT call shuffle a second time)
        """
        split_nb = int(self.args.ratio_dataset * len(self.songs))
        self.songs_train = self.songs[:split_nb]
        self.songs_test = self.songs[split_nb:]
        self.songs = None  # Not needed anymore (free some memory)

    def get_batches(self):
        """ Prepare the batches for the current epoch
        WARNING: The songs are not shuffled in this functions. We leave the choice
        to the batch_builder to manage the shuffling
        Return:
            list[Batch], list[Batch]: The batches for the training and testing set (can be generators)
        """
        return (
            self.batch_builder.get_list(self.songs_train, name='train'),
            self.batch_builder.get_list(self.songs_test, name='test'),
        )

    # def get_batches_test(self, ):  # TODO: Should only return a single batch (loading done in main class)
    #     """ Return the batch which initiate the RNN when generating
    #     The initial batches are loaded from a json file containing the first notes of the song. The note values
    #     are the standard midi ones. Here is an examples of an initiator file:
    #     Args:
    #         TODO
    #     Return:
    #         Batch: The generated batch
    #     """
    #     assert self.args.batch_size == 1
    #     batch = None  # TODO
    #     return batch

    def get_batches_test_old(self):  # TODO: This is the old version. Ideally should use the version above
        """ Return the batches which initiate the RNN when generating
        The initial batches are loaded from a json file containing the first notes of the song. The note values
        are the standard midi ones. Here is an examples of an initiator file:
        ```
        {"initiator":[
            {"name":"Simple_C4",
             "seq":[
                {"notes":[60]}
            ]},
            {"name":"some_chords",
             "seq":[
                {"notes":[60,64]}
                {"notes":[66,68,71]}
                {"notes":[60,64]}
            ]}
        ]}
        ```
        Return:
            List[Batch], List[str]: The generated batches with the associated names
        """
        assert self.args.batch_size == 1

        batches = []
        names = []

        with open(self.TEST_INIT_FILE) as init_file:
            initiators = json.load(init_file)

        for initiator in initiators['initiator']:
            raw_song = music.Song()
            main_track = music.Track()

            current_tick = 0
            for seq in initiator['seq']:  # We add a few notes
                for note_pitch in seq['notes']:
                    new_note = music.Note()
                    new_note.note = note_pitch
                    new_note.tick = current_tick
                    main_track.notes.append(new_note)
                current_tick += 1

            raw_song.tracks.append(main_track)
            raw_song.normalize(inverse=True)

            batch = self.batch_builder.process_batch(raw_song)

            names.append(initiator['name'])
            batches.append(batch)

        return batches, names

    @staticmethod
    def _convert_to_piano_rolls(outputs):
        """ Create songs from the decoder outputs.
        Reshape the list of outputs to list of piano rolls
        Args:
            outputs (List[np.array]): The list of the predictions of the decoder
        Return:
            List[np.array]: the list of the songs (one song by batch) as piano roll
        """

        # Extract the batches and recreate the array for each batch
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

        return piano_rolls

    def visit_recorder(self, outputs, base_dir, base_name, recorders, chosen_labels=None):
        """ Save the predicted output songs using the given recorder
        Args:
            outputs (List[np.array]): The list of the predictions of the decoder
            base_dir (str): Path were to save the outputs
            base_name (str): filename of the output (without the extension)
            recorders (List[Obj]): Interfaces called to convert the song into a file (ex: midi or png). The recorders
                need to implement the method write_song (the method has to add the file extension) and the
                method get_input_type.
            chosen_labels (list[np.Array[batch_size, int]]): the chosen class at each timestep (useful to reconstruct the generated song)
        """

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for batch_id in range(outputs[0].shape[0]):  # Loop over batch_size
            song = self.batch_builder.reconstruct_batch(outputs, batch_id, chosen_labels)
            for recorder in recorders:
                if recorder.get_input_type() == 'song':
                    input = song
                elif recorder.get_input_type() == 'array':
                    #input = self._convert_song2array(song)
                    continue  # TODO: For now, pianoroll desactivated
                else:
                    raise ValueError('Unknown recorder input type.'.format(recorder.get_input_type()))
                base_path = os.path.join(base_dir, base_name + '-' + str(batch_id))
                recorder.write_song(input, base_path)
