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
Mid-level interface for the python files
"""

import mido  # Midi lib
from tqdm import tqdm  # Plotting messages


class MidiInvalidException(Exception):
    pass


class MidiReader:
    """ Class which manage the midi files at the message level
    """

    def __init__(self, filename):
        """
        TODO: For now read only. Make it write too (from an empty file)
        """
        self.NB_KEYS = 88  # Vertical dimension of a song
        self.BAR_DIVISION = 16  # Nb of tics in a bar (What about waltz ? is 12 better ?)
        # Define a max song length ?

        self.resolution = 0  # bpm
        self.initial_tempo = 0.0

        self.data = None  #  Sparse tensor of size [NB_KEYS,nb_bars*BAR_DIVISION]

        self.load_file(filename)

    def load_file(self, filename):
        """ Extract data from midi file
        Args:
            filename: a valid midi file
        Return:
            TODO
        """
        # Load in the MIDI data using the midi module
        midi_data = mido.MidiFile(filename)

        # Get header values

        # 3 midi types:
        # * type 0 (single track): all messages are saved in one multi-channel track
        # * type 1 (synchronous): all tracks start at the same time
        # * type 2 (asynchronous): each track is independent of the others

        # Nb tracks
        # The first track is usually (always a header track)

        # Division (ticks per beat notes or SMTPE timecode)
        # If negative (first byte=1), the mode is SMTPE timecode (unsupported)
        # 1 MIDI clock = 1 beat = 1 quarter note

        print('{}: type {}, {} tracks, {} tics'.format(
            filename,
            midi_data.type,
            len(midi_data.tracks),
            midi_data.ticks_per_beat
        ))

        # Assert
        if midi_data.type != 1:
            raise MidiInvalidException('Only type 1 supported ({} given)'.format(midi_data.type))
        if not 0 < midi_data.ticks_per_beat < 128:
            raise MidiInvalidException('SMTPE timecode not supported ({} given)'.format(midi_data.ticks_per_beat))

        # Get tracks messages

        # The tracks are a mix of meta messages, which determine the tempo and signature, and note messages, which
        # correspond to the melodie.
        # Generally, the meta event are set at the beginning of each tracks. In format 1, these meta-events should be
        # contained in the first track (known as 'Tempo Map').

        # If not set, default parameters are:
        #  * time signature: 4/4
        #  * tempo: 120 beats per minute

        # Each event contain begins by a delta time value, which correspond to the number of ticks from the previous
        # event (0 for simultaneous event)

        # Merge tracks ?
        #midi_data.tracks = [mido.merge_tracks(midi_data.tracks)] ??

        tempo_map = midi_data.tracks[0]  # Will contains the tick scales

        # Assert
        for message in tempo_map:
            if not isinstance(message, mido.MetaMessage):
                raise MidiInvalidException('Tempo map should not contains notes')

        for i, track in enumerate(midi_data.tracks[1:]):  # We ignore the tempo map
            tqdm.write('Track {}: {}'.format(i, track.name))
            for message in track:
                if isinstance(message, mido.MetaMessage):  # Lyrics, track name and other useless info
                    if message.type == 'instrument_name':
                        tqdm.write('{}'.format(message))
                    elif message._spec.type_byte > 0x50:
                        # Following the midi specification, those signals contains more than
                        # simple indication and should modify the way the piece is played (tempo, key signature,...)
                        raise MidiInvalidException('Track {} should not contain {}'.format(i, message.type))
                    # What about 'sequence_number', cue_marker ???
                else:  # Note event
                    tqdm.write('{}'.format(message.time))

        return

        #midi_data.

        # Load in the MIDI data using the midi module
        midi_data = mido.MidiFile(midi_file)

        # Convert tick values in midi_data to absolute, a useful thing.
        midi_data.make_ticks_abs()

        # Store the resolution for later use
        self.resolution = midi_data.resolution

        # Populate the list of tempo changes (tick scales)
        self._load_tempo_changes(midi_data)

        # Update the array which maps ticks to time
        max_tick = max([max([e.tick for e in t]) for t in midi_data]) + 1
        # If max_tick is huge, the MIDI file is probably corrupt
        # and creating the __tick_to_time array will thrash memory
        if max_tick > MAX_TICK:
            raise ValueError(('MIDI file has a largest tick of {},'
                              ' it is likely corrupt'.format(max_tick)))

        # Create list that maps ticks to time in seconds
        self._update_tick_to_time(max_tick)

        # Populate the list of key and time signature changes
        self._load_metadata(midi_data)

        # Check that there are tempo, key and time change events
        # only on track 0
        if sum([sum([isinstance(event, (midi.events.SetTempoEvent,
                                        midi.events.KeySignatureEvent,
                                        midi.events.TimeSignatureEvent))
                    for event in track]) for track in midi_data[1:]]):
            warnings.warn(("Tempo, Key or Time signature change events"
                           " found on non-zero tracks."
                           "  This is not a valid type 0 or type 1 MIDI"
                           " file. Tempo, Key or Time Signature"
                           " may be wrong."),
                          RuntimeWarning)

        # Populate the list of instruments
        self._load_instruments(midi_data)