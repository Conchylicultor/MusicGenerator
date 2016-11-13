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
Image connector interface

"""

import cv2 as cv
import numpy as np

import deepmusic.songstruct as music  # Should we use that to tuncate the top and bottom image ?


class ImgConnector:
    """ Class to read and write songs (piano roll arrays) as images
    """

    @staticmethod
    def load_file(filename):
        """ Extract data from midi file
        Args:
            filename (str): a valid img file
        Return:
            np.array: the piano roll associated with the
        """
        # TODO ? Could be useful to load initiators created with Gimp (more intuitive than the current version)

    @staticmethod
    def write_song(piano_roll, filename):
        """ Save the song on disk
        Args:
            piano_roll (np.array): a song object containing the tracks and melody
            filename (str): the path were to save the song (don't add the file extension)
        """
        note_played = piano_roll > 0.5
        piano_roll_int = np.uint8(piano_roll*255)

        b = piano_roll_int * (~note_played).astype(np.uint8)  # Note silenced
        g = np.zeros(piano_roll_int.shape, dtype=np.uint8)    # Empty channel
        r = piano_roll_int * note_played.astype(np.uint8)     # Notes played

        img = cv.merge((b, g, r))

        # TODO: We could insert a first column indicating the piano keys (black/white key)

        cv.imwrite(filename + '.png', img)

    @staticmethod
    def get_input_type():
        return 'array'
