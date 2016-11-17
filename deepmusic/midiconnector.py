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
Mid-level interface for the python files
"""

import mido  # Midi lib

import deepmusic.songstruct as music


class MidiInvalidException(Exception):
    pass


class MidiConnector:
    """ Class which manage the midi files at the message level
    """
    META_INFO_TYPES = [  # Can safely be ignored
        'midi_port',
        'track_name',
        'lyrics',
        'end_of_track',
        'copyright',
        'marker',
        'text'
    ]
    META_TEMPO_TYPES = [  # Have an impact on how the song is played
        'key_signature',
        'set_tempo',
        'time_signature'
    ]

    MINIMUM_TRACK_LENGTH = 4  # Bellow this value, the track will be ignored

    MIDI_CHANNEL_DRUMS = 10  # The channel reserved for the drums (according to the specs)

    # Define a max song length ?

    #self.resolution = 0  # bpm
    #self.initial_tempo = 0.0

    #self.data = None  # Sparse tensor of size [NB_KEYS,nb_bars*BAR_DIVISION] or simply a list of note ?

    @staticmethod
    def load_file(filename):
        """ Extract data from midi file
        Args:
            filename (str): a valid midi file
        Return:
            Song: a song object containing the tracks and melody
        """
        # Load in the MIDI data using the midi module
        midi_data = mido.MidiFile(filename)

        # Get header values

        # 3 midi types:
        # * type 0 (single track): all messages are saved in one multi-channel track
        # * type 1 (synchronous): all tracks start at the same time
        # * type 2 (asynchronous): each track is independent of the others

        # Division (ticks per beat notes or SMTPE timecode)
        # If negative (first byte=1), the mode is SMTPE timecode (unsupported)
        # 1 MIDI clock = 1 beat = 1 quarter note

        # Assert
        if midi_data.type != 1:
            raise MidiInvalidException('Only type 1 supported ({} given)'.format(midi_data.type))
        if not 0 < midi_data.ticks_per_beat < 128:
            raise MidiInvalidException('SMTPE timecode not supported ({} given)'.format(midi_data.ticks_per_beat))

        # TODO: Support at least for type 0

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

        tempo_map = midi_data.tracks[0]  # Will contains the tick scales
        # TODO: smpte_offset

        # Warning: The drums are filtered

        # Merge tracks ? < Not when creating the dataset
        #midi_data.tracks = [mido.merge_tracks(midi_data.tracks)] ??

        new_song = music.Song()

        new_song.ticks_per_beat = midi_data.ticks_per_beat

        # TODO: Normalize the ticks per beats (same for all songs)

        for message in tempo_map:
            # TODO: Check we are only 4/4 (and there is no tempo changes ?)
            if not isinstance(message, mido.MetaMessage):
                raise MidiInvalidException('Tempo map should not contains notes')
            if message.type in MidiConnector.META_INFO_TYPES:
                pass
            elif message.type == 'set_tempo':
                new_song.tempo_map.append(message)
            elif message.type in MidiConnector.META_TEMPO_TYPES:  # We ignore the key signature and time_signature ?
                pass
            elif message.type == 'smpte_offset':
                pass  # TODO
            else:
                err_msg = 'Header track contains unsupported meta-message type ({})'.format(message.type)
                raise MidiInvalidException(err_msg)

        for i, track in enumerate(midi_data.tracks[1:]):  # We ignore the tempo map
            i += 1  # Warning: We have skipped the track 0 so shift the track id
            #tqdm.write('Track {}: {}'.format(i, track.name))

            new_track = music.Track()

            buffer_notes = []  # Store the current notes (pressed but not released)
            abs_tick = 0  # Absolute nb of ticks from the beginning of the track
            for message in track:
                abs_tick += message.time
                if isinstance(message, mido.MetaMessage):  # Lyrics, track name and other meta info
                    if message.type in MidiConnector.META_INFO_TYPES:
                        pass
                    elif message.type in MidiConnector.META_TEMPO_TYPES:
                        # TODO: Could be just a warning
                        raise MidiInvalidException('Track {} should not contain {}'.format(i, message.type))
                    else:
                        err_msg = 'Track {} contains unsupported meta-message type ({})'.format(i, message.type)
                        raise MidiInvalidException(err_msg)
                    # What about 'sequence_number', cue_marker ???
                else:  # Note event
                    if message.type == 'note_on' and message.velocity != 0:  # Note added
                        new_note = music.Note()
                        new_note.tick = abs_tick
                        new_note.note = message.note
                        if message.channel+1 != i and message.channel+1 != MidiConnector.MIDI_CHANNEL_DRUMS:  # Warning: Mido shift the channels (start at 0) # TODO: Channel management for type 0
                            raise MidiInvalidException('Notes belong to the wrong tracks ({} instead of {})'.format(i, message.channel))  # Warning: May not be an error (drums ?) but probably
                        buffer_notes.append(new_note)
                    elif message.type == 'note_off' or message.type == 'note_on':  # Note released
                        for note in buffer_notes:
                            if note.note == message.note:
                                note.duration = abs_tick - note.tick
                                buffer_notes.remove(note)
                                new_track.notes.append(note)
                    elif message.type == 'program_change':  # Instrument change
                        if not new_track.set_instrument(message):
                            # TODO: We should create another track with the new instrument
                            raise MidiInvalidException('Track {} as already a program defined'.format(i))
                        pass
                    elif message.type == 'control_change':  # Damper pedal, mono/poly, channel volume,...
                        # Ignored
                        pass
                    elif message.type == 'aftertouch':  # Signal send after a key has been press. What real effect ?
                        # Ignored ?
                        pass
                    elif message.type == 'pitchwheel':  # Modulate the song
                        # Ignored
                        pass
                    else:
                        err_msg = 'Track {} contains unsupported message type ({})'.format(i, message)
                        raise MidiInvalidException(err_msg)
                # Message read
            # Track read

            # Assert
            if buffer_notes:  # All notes should have ended
                raise MidiInvalidException('Some notes ({}) did not ended'.format(len(buffer_notes)))
            if len(new_track.notes) < MidiConnector.MINIMUM_TRACK_LENGTH:
                #tqdm.write('Track {} ignored (too short): {} notes'.format(i, len(new_track.notes)))
                continue
            if new_track.is_drum:
                #tqdm.write('Track {} ignored (is drum)'.format(i))
                continue

            new_song.tracks.append(new_track)
        # All track read

        if not new_song.tracks:
            raise MidiInvalidException('Empty song. No track added')

        return new_song

    @staticmethod
    def write_song(song, filename):
        """ Save the song on disk
        Args:
            song (Song): a song object containing the tracks and melody
            filename (str): the path were to save the song (don't add the file extension)
        """

        midi_data = mido.MidiFile(ticks_per_beat=song.ticks_per_beat)

        # Define track 0
        new_track = mido.MidiTrack()
        midi_data.tracks.append(new_track)
        new_track.extend(song.tempo_map)

        for i, track in enumerate(song.tracks):
            # Define the track
            new_track = mido.MidiTrack()
            midi_data.tracks.append(new_track)
            new_track.append(mido.Message('program_change', program=0, time=0))  # Played with standard piano

            messages = []
            for note in track.notes:
                # Add all messages in absolute time
                messages.append(mido.Message(
                    'note_on',
                    note=note.note,  # WARNING: The note should be int (NOT np.int64)
                    velocity=64,
                    channel=i,
                    time=note.tick))
                messages.append(mido.Message(
                    'note_off',
                    note=note.note,
                    velocity=64,
                    channel=i,
                    time=note.tick+note.duration)
                )

            # Reorder the messages chronologically
            messages.sort(key=lambda x: x.time)

            # Convert absolute tick in relative tick
            last_time = 0
            for message in messages:
                message.time -= last_time
                last_time += message.time

                new_track.append(message)

        midi_data.save(filename + '.mid')

    @staticmethod
    def get_input_type():
        return 'song'
