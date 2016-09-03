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
Hierarchical data structures of a song
"""


class Note:
    """ Structure which encapsulate the song data
    """
    def __init__(self):
        self.tick = 0
        self.note = 0
        self.duration = 0


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
        if msg.channel == 10 or msg.program > 112:
            self.is_drum = True

        return True


class Song:
    """ Structure which encapsulate the song data
    """
    def __init__(self):
        self.ticks_per_beat = 96
        self.tempo_map = []
        self.tracks = []  # List[Track]
