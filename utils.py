#!/usr/bin/env python3

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
Some utilities functions, to easily manipulate large volume of downloaded files.
Independent of the main program but useful to extract/create the dataset
"""

import os
import subprocess
import glob


def extract_files():
    """ Recursively extract all files from a given directory
    """
    input_dir = '../www.chopinmusic.net/'
    output_dir = 'chopin_clean/'

    os.makedirs(output_dir, exist_ok=True)

    print('Extracting:')
    i = 0
    for filename in glob.iglob(os.path.join(input_dir, '**/*.mid'), recursive=True):
        print(filename)
        os.rename(filename, os.path.join(output_dir, os.path.basename(filename)))
        i += 1
    print('{} files extracted.'.format(i))


def rename_files():
    """ Rename all files of the given directory following some rules
    """
    input_dir = 'chopin/'
    output_dir = 'chopin_clean/'

    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    list_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]

    print('Renaming {} files:'.format(len(list_files)))
    for prev_name in list_files:
        new_name = prev_name.replace('midi.asp?file=', '')
        new_name = new_name.replace('%2F', '_')
        print('{} -> {}'.format(prev_name, new_name))
        os.rename(os.path.join(input_dir, prev_name), os.path.join(output_dir, new_name))


def convert_midi2mp3():
    """ Convert all midi files of the given directory to mp3
    """
    input_dir = 'docs/midi/'
    output_dir = 'docs/mp3/'

    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('Converting:')
    i = 0
    for filename in glob.iglob(os.path.join(input_dir, '**/*.mid'), recursive=True):
        print(filename)
        in_name = filename
        out_name = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.mp3')
        command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(in_name, out_name)  # TODO: Redirect stdout to avoid polluting the screen (have cleaner printing)
        subprocess.call(command, shell=True)
        i += 1
    print('{} files converted.'.format(i))


if __name__ == '__main__':
    convert_midi2mp3()
