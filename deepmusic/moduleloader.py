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

""" Register all available modules
All new module should be added here
"""

from deepmusic.modulemanager import ModuleManager

from deepmusic.musicdata import BatchBuilderPianoRoll
from deepmusic.musicdata import BatchBuilderRelative

from deepmusic.modules import learning_rate


class ModuleLoader:
    """ Global module manager, synchronize the loading, printing, parsing of
    all modules.
    The modules are then instantiated and use in their respective class
    """
    enco_cells = None
    deco_cells = None
    batch_builders = None
    learning_rate_policies = None

    @staticmethod
    def register_all():
        """ List all available modules for the current session
        This function should be called only once at the beginning of the
        program, before parsing the command lines arguments
        Don't instantiate anything here (just notify the program).
        The module manager name will define the command line flag
        which will be used
        """
        ModuleLoader.batch_builders = ModuleManager('batch_builder')
        ModuleLoader.batch_builders.register(BatchBuilderRelative)
        ModuleLoader.batch_builders.register(BatchBuilderPianoRoll)

        ModuleLoader.learning_rate_policies = ModuleManager('learning_rate')
        ModuleLoader.learning_rate_policies.register(learning_rate.LearningRatePolicyCst)
        ModuleLoader.learning_rate_policies.register(learning_rate.LearningRatePolicyStepsWithDecay)
        ModuleLoader.learning_rate_policies.register(learning_rate.LearningRatePolicyAdaptive)

    @staticmethod
    def save_all(config):
        """ Save the modules configurations
        """
        config['Modules'] = {}
        ModuleLoader.batch_builders.save(config['Modules'])
        ModuleLoader.learning_rate_policies.save(config['Modules'])

    @staticmethod
    def load_all(args, config):
        """ Restore the module configuration
        """
        ModuleLoader.batch_builders.load(args, config['Modules'])
        ModuleLoader.learning_rate_policies.load(args, config['Modules'])

    @staticmethod
    def print_all(args):
        """ Print modules current configuration
        """
        ModuleLoader.batch_builders.print(args)
        ModuleLoader.learning_rate_policies.print(args)
