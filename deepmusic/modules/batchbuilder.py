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
The batch builder convert the songs into data readable by the neural networks.
Used for training, testing and generating
"""

import random
import json

import deepmusic.songstruct as music


class BatchBuilder:
    """ Class which create and manage batches
    Batches are created from the songs
    Define the song representation input (and output) format so the network must
    support the format
    The class has the choice to either entirely create the
    batches when get list is called or to create the batches
    as the training progress (more memory efficient)
    """
    # TODO: Should add a option to pre-compute a lot of batches and
    # cache them in the hard drive
    # TODO: For generating mode, add another function
    # TODO: Add a function to get the length too (for tqdm when generators) ?
    def __init__(self, args):
        """
        """
        self.args = args

    @staticmethod
    def get_module_id():
        """ Return the unique id associated with the builder
        Ultimately, the id will be used for saving/loading the dataset, and
        as parameter argument.
        Returns:
            str: The name of the builder
        """
        raise NotImplementedError('Abstract class')

    def get_list(self, dataset):
        """ Compute the batches for the current epoch
        Is called twice (for training and testing)
        Args:
            dataset (list[Objects]): the training/testing set ()
        Return:
            list[Batch]: the batches to process
        """
        raise NotImplementedError('Abstract class')

    def build_next(self, batch):
        """ In case of a generator (batches non precomputed), compute the batch given
        the batch id passed
        Args:
            batch: the current testing or training batch or id of batch to generate
        Return:
            Batch: the computed batch
        """
        return batch

    def prepare_data(self, song):
        """ Apply some pre-processing to the songs so the song
        already get the right input representation.
        Do it once globally for all songs
        Args:
            song (Song): the training/testing set
        Return:
            Object: the song after formatting
        """
        return song  # By default no pre-processing


class Relative(BatchBuilder):
    """ Prepare the batches for the current epoch.
    Generate batches of the form:
        12 values for relative position with previous notes (modulo 12)
        14 values for the relative pitch (+/-7)
        12 values for the relative positions with the previous note
    """

    class RelativeSong:
        """ Struct which define a song in a relative way (intern class format)
        Can only play 7 octave (so the upper and lower notes of the
        piano are never reached (not that important in practice))
        """
        def __init__(self):
            """ All attribute are defined with respect with the previous one
            """
            self.first_note = None  # Define the reference note
            # TODO: Try with time relative to prev vs next
            # TODO: Try to randomly permute chords vs low to high pitch
            # TODO: Try pitch %7 vs fixed +/-7
            # TODO: Try to add a channel number for each note (2 class softmax) <= Would require a clean database where the melodie/bass are clearly separated
            self.pitch_class = []  # A, B, C,... +/- %12
            self.pitch = []  # Octave +/- % 7
            self.prev_tick = []  # Distance from previous note (from -0 up to -MAXIMUM_SONG_RESOLUTION*NOTES_PER_BAR (=1 bar))

    # TODO: How to optimize !! (precompute all values, use sparse arrays ?)
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def get_module_id():
        return 'relative'

    def prepare_data(self, dataset):
        """ Pre-process the data once globally
        Do it once globally.
        Args:
            dataset (list[Song]):
        Returns:
            list[Obj]: the new dataset
        """
        # TODO: If we transform songs by song instead of all in once, we could greatly
        # optimize memory (instead of having 2x the dataset loaded in memory)
        # Otherwise, could clear the songs here each time we process it (not very
        # smart because we would have to do it for every builder class)

        new_dataset = []
        for old_song in dataset:
            new_song = Relative.RelativeSong()

            old_song.normalize()

            # Gather all notes and sort them
            all_notes = []
            for track in old_song.tracks:
                for note in track.notes:
                    all_notes.append(note)
            all_notes.sort(key=lambda n: n.tick)

            # Compute the relative position for each note
            prev_note = all_notes[0]
            new_song.first_note = prev_note  # TODO: What if the song start by a chord ?
            for note in all_notes[1:]:
                # TODO: Replace numbers by consts
                new_song.pitch_class.append((note.note - prev_note.note) % 12)
                new_song.pitch.append((note.note//12 - prev_note.note//12) % 7)  # TODO: add offset for the notes ? (where does the game begins ?)
                new_song.prev_tick.append(note.tick - prev_note.tick)

                prev_note = note

            new_dataset.append(new_song)

        songs_set = dataset
        for song in songs_set:
            len_song = song.shape[-1]  # The last dimension correspond to the song duration
            max_start = len_song - sample_subsampling_length
            assert max_start >= 0  # TODO: Error handling (and if =0, compatible with randint ?)
            nb_sample_song = 2*len_song // self.args.sample_length  # The number of subsample is proportional to the song length
            for _ in range(nb_sample_song):
                start = np.random.randint(max_start)  # TODO: Add mode to only start at the begining of a bar
                sub_song = song[:, start:start+sample_subsampling_length]
                sub_songs.append(sub_song)

        return 'relative'

    def get_list(self,  dataset):
        """ See parent class for more details
        Args:
            dataset (list[Song]): the training/testing set
        Return:
            list[Batch]: the batches to process
        """
        # Shuffle the song extracts
        print("Shuffling the dataset...")
        random.shuffle(dataset)

        # Group the samples together to create the batches
        print("Generating batches...")

        pass

    def build_next(self, batch):
        pass


class PianoRoll(BatchBuilder):
    """ Old piano roll format (legacy code). Won't work as it is
    """
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def get_module_id():
        return 'pianoroll'

    def get_list(self, dataset):

        # On the original version, the songs were directly converted to piano roll
        # self._convert_song2array()

        batches = []

        # TODO: Create batches (randomly cut each song in some small parts (need to know the total length for that)
        # then create the big matrix (NB_NOTE*sample_length) and turn that into batch). If process too long,
        # could save the created batches in a new folder, data/samples or save/model.

        # TODO: Create batches from multiples length (buckets). How to change the loss functions weights (longer
        # sequences more penalized ?)

        # TODO: Optimize memory management

        # First part: Randomly extract subsamples of the songs
        print('Subsampling songs ({})...'.format('train' if train_set else 'test'))

        sample_subsampling_length = self.args.sample_length+1  # We add 1 because each input has to predict the next output

        sub_songs = []
        songs_set = dataset
        for song in songs_set:
            len_song = song.shape[-1]  # The last dimension correspond to the song duration
            max_start = len_song - sample_subsampling_length
            assert max_start >= 0  # TODO: Error handling (and if =0, compatible with randint ?)
            nb_sample_song = 2*len_song // self.args.sample_length  # The number of subsample is proportional to the song length
            for _ in range(nb_sample_song):
                start = np.random.randint(max_start)  # TODO: Add mode to only start at the begining of a bar
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

        for samples in gen_next_samples():  # TODO: tqdm with persist = False / will this work with generators ?
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

    def get_batches_test(self):  # TODO: Move that to BatchBuilder
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
            batch = Batch()

            for seq in initiator['seq']:  # We add a few notes
                new_input = -np.ones([self.args.batch_size, music.NB_NOTES])  # No notes played by default
                for note in seq['notes']:
                    new_input[0, note] = 1.0
                batch.inputs.append(new_input)

            names.append(initiator['name'])
            batches.append(batch)

        return batches, names
