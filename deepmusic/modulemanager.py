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
"""

from collections import OrderedDict


class ModuleManager:
    """ Class which manage modules
    A module can be any class, as long as it implement the static method
    get_module_id and has a compatible constructor. The role of the module
    manager is to ensure that only one of the registered classes are used
    on the program.
    The first added module will be the one used by default.
    For now the modules can't save their state.
    """
    def __init__(self):
        """
        """
        self.modules = OrderedDict()  # The order on which the modules are added is conserved
        self.module_instance = None  # Reference to the chosen module
        self.module_parameters = None  # Argument passed (for saving/loading)

    def register(self, module):
        """ Register a new module
        The only restriction is that the given class has to implement the static
        method get_module_id
        Args:
            module (Class): the module class to register
        """
        assert not module.get_module_id() in self.modules  # Overwriting not allowed
        self.modules[module.get_module_id()] = module

    def get_modules_ids(self):
        """ Return the list of added modules
        Useful for instance for the command line parser
        Returns:
            list[str]: the list of modules
        """
        return self.modules.keys()

    def build_module(self, module_args, args):
        """ Instantiate the chosen module
        This function can be called only once when initializing the module
        Args:
            module_args (list[str]): name of the module and its eventual additional parameters
            args (Obj): the global program parameters
        Returns:
            Obj: the created module
        """
        assert self.module_instance is not None

        module_name = module_args[0]
        additional_args = module_args[1:]
        self.module_instance = self.modules[module_name](args, *additional_args)
        self.module_parameters = module_args
        return self.module_instance

    def add_argparse(self, group_args, name, comment):
        """ Add the module to the command line parser
        All modules have to be registered before that call
        Args:
            group_args (ArgumentParser):
            name (str): name of the
            comment (str): help to add
        """
        keys = list(self.modules.keys())
        group_args.add_argument(
            '--{}'.format(name),
            type=str,
            nargs='+',
            default=[keys[0]],  # No defaults optional argument (implemented in the module class)
            help=comment + ' Choices available: {}'.format(', '.join(keys))
        )

    def save(self):
        """
        """

    def load(self):
        """
        """
