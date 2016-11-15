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

""" Register all available modules
All new module should be added here
"""

from deepmusic.modulemanager import ModuleManager

from deepmusic.modules import batchbuilder
from deepmusic.modules import learningratepolicy
from deepmusic.modules import encoder
from deepmusic.modules import decoder
from deepmusic.modules import loopprocessing


class ModuleLoader:
    """ Global module manager, synchronize the loading, printing, parsing of
    all modules.
    The modules are then instantiated and use in their respective class
    """
    enco_cells = None
    deco_cells = None
    batch_builders = None
    learning_rate_policies = None
    loop_processings = None

    @staticmethod
    def register_all():
        """ List all available modules for the current session
        This function should be called only once at the beginning of the
        program, before parsing the command lines arguments
        It doesn't instantiate anything here (just notify the program).
        The module manager name will define the command line flag
        which will be used
        """
        ModuleLoader.batch_builders = ModuleManager('batch_builder')
        ModuleLoader.batch_builders.register(batchbuilder.Relative)
        ModuleLoader.batch_builders.register(batchbuilder.PianoRoll)

        ModuleLoader.learning_rate_policies = ModuleManager('learning_rate')
        ModuleLoader.learning_rate_policies.register(learningratepolicy.Cst)
        ModuleLoader.learning_rate_policies.register(learningratepolicy.StepsWithDecay)
        ModuleLoader.learning_rate_policies.register(learningratepolicy.Adaptive)

        ModuleLoader.enco_cells = ModuleManager('enco_cell')
        ModuleLoader.enco_cells.register(encoder.Identity)
        ModuleLoader.enco_cells.register(encoder.Rnn)
        ModuleLoader.enco_cells.register(encoder.Embedding)

        ModuleLoader.deco_cells = ModuleManager('deco_cell')
        ModuleLoader.deco_cells.register(decoder.Lstm)
        ModuleLoader.deco_cells.register(decoder.Perceptron)
        ModuleLoader.deco_cells.register(decoder.Rnn)

        ModuleLoader.loop_processings = ModuleManager('loop_processing')
        ModuleLoader.loop_processings.register(loopprocessing.SampleSoftmax)
        ModuleLoader.loop_processings.register(loopprocessing.ActivateScale)

    @staticmethod
    def save_all(config):
        """ Save the modules configurations
        """
        config['Modules'] = {}
        ModuleLoader.batch_builders.save(config['Modules'])
        ModuleLoader.learning_rate_policies.save(config['Modules'])
        ModuleLoader.enco_cells.save(config['Modules'])
        ModuleLoader.deco_cells.save(config['Modules'])
        ModuleLoader.loop_processings.save(config['Modules'])

    @staticmethod
    def load_all(args, config):
        """ Restore the module configuration
        """
        ModuleLoader.batch_builders.load(args, config['Modules'])
        ModuleLoader.learning_rate_policies.load(args, config['Modules'])
        ModuleLoader.enco_cells.load(args, config['Modules'])
        ModuleLoader.deco_cells.load(args, config['Modules'])
        ModuleLoader.loop_processings.load(args, config['Modules'])

    @staticmethod
    def print_all(args):
        """ Print modules current configuration
        """
        ModuleLoader.batch_builders.print(args)
        ModuleLoader.learning_rate_policies.print(args)
        ModuleLoader.enco_cells.print(args)
        ModuleLoader.deco_cells.print(args)
        ModuleLoader.loop_processings.print(args)
