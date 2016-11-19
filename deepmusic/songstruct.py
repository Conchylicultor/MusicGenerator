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
Hierarchical data structures of a song
"""

import operator  # To rescale the song


MIDI_NOTES_RANGE = [21, 108]  # Min and max (included) midi note on a piano
# TODO: Warn/throw when we try to add a note outside this range
# TODO: Easy conversion from this range to tensor vector id (midi_note2tf_id)

NB_NOTES = MIDI_NOTES_RANGE[1] - MIDI_NOTES_RANGE[0] + 1  # Vertical dimension of a song (=88 of keys for a piano)

BAR_DIVISION = 16  # Nb of tics in a bar (What about waltz ? is 12 better ?)


class Note:
    """ Structure which encapsulate the song data
    """
    def __init__(self):
        self.tick = 0
        self.note = 0
        self.duration = 32  # TODO: Define the default duration / TODO: Use standard musical units (quarter note/eighth note) ?, don't convert here

    def get_relative_note(self):
        """ Convert the absolute midi position into the range given by MIDI_NOTES_RANGE
        Return
            int: The new position relative to the range (position on keyboard)
        """
        return self.note - MIDI_NOTES_RANGE[0]

    def set_relative_note(self, rel):
        """ Convert given note into a absolute midi position
        Args:
            rel (int): The new position relative to the range (position on keyboard)
        """
        # TODO: assert range (rel < NB_NOTES)?
        self.note = rel + MIDI_NOTES_RANGE[0]


class Track:
    """ Structure which encapsulate a track of the song
    Ideally, each track should correspond to a single instrument and one channel. Multiple tracks could correspond
    to the same channel if different instruments use the same channel.
    """
    def __init__(self):
        #self.tempo_map = None  # Use a global tempo map
        self.instrument = None
        self.notes = []  # List[Note]
        #self.color = (0, 0, 0)  # Color of the track for visual plotting
        self.is_drum = False

    def set_instrument(self, msg):
        """ Initialize from a mido message
        Args:
            msg (mido.MidiMessage): a valid control_change message
        """
        if self.instrument is not None:  # Already an instrument set
            return False

        assert msg.type == 'program_change'

        self.instrument = msg.program
        if msg.channel == 9 or msg.program > 112:  # Warning: Mido shift the channels (start at 0)
            self.is_drum = True

        return True


class Song:
    """ Structure which encapsulate the song data
    """

    # Define the time unit
    # TODO: musicdata should have possibility to modify those parameters (through self.args)
    # Invert of time note which define the maximum resolution for a song. Ex: 2 for 1/2 note, 4 for 1/4 of note
    MAXIMUM_SONG_RESOLUTION = 4
    NOTES_PER_BAR = 4  # Waltz not supported

    def __init__(self):
        self.ticks_per_beat = 96
        self.tempo_map = []
        self.tracks = []  # List[Track]

    def __len__(self):
        """ Return the absolute tick when the last note end
        Note that the length is recomputed each time the function is called
        """
        return max([max([n.tick + n.duration for n in t.notes]) for t in self.tracks])

    def _get_scale(self):
        """ Compute the unit scale factor for the song
        The scale factor allow to have a tempo independent time unit, to represent the song as an array
        of dimension [key, time_unit]. Once computed, one has just to divide (//) the ticks or multiply
        the time units to go from one representation to the other.

        Return:
            int: the scale factor for the current song
        """

        # TODO: Assert that the scale factor is not a float (the % =0)
        return 4 * self.ticks_per_beat // (Song.MAXIMUM_SONG_RESOLUTION*Song.NOTES_PER_BAR)

    def normalize(self, inverse=False):
        """ Transform the song into a tempo independent song
        Warning: If the resolution of the song is is more fine that the given
        scale, some information will be definitively lost
        Args:
            inverse (bool): if true, we reverse the normalization
        """
        scale = self._get_scale()
        op = operator.floordiv if not inverse else operator.mul

        # TODO: Not sure why this plot a decimal value (x.66). Investigate...
        # print(song_length/scale)

        # Shifting all notes
        for track in self.tracks:
            for note in track.notes:
                note.tick = op(note.tick, scale)  # //= or *=
