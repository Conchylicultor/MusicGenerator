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

""" Module manager class definition
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
    def __init__(self, name):
        """
        Args:
            name (str): the name of the module manager (useful for saving/printing)
        """
        self.name = name
        self.modules = OrderedDict()  # The order on which the modules are added is conserved
        self.module_instance = None  # Reference to the chosen module
        self.module_name = ''  # Type of module chosen (useful when saving/loading)
        self.module_parameters = []  # Arguments passed (for saving/loading)

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

    def get_chosen_name(self):
        """ Return the name of the chosen module
        Name is defined by get_module_id
        Returns:
            str: the name of the chosen module
        """
        return self.module_name

    def get_module(self):
        """ Return the chosen module
        Returns:
            Obj: the reference on the module instance
        """
        assert self.module_instance is not None
        return self.module_instance

    def build_module(self, args):
        """ Instantiate the chosen module
        This function can be called only once when initializing the module
        Args:
            args (Obj): the global program parameters
        Returns:
            Obj: the created module
        """
        assert self.module_instance is None

        module_args = getattr(args, self.name)  # Get the name of the module and its eventual additional parameters (Exception will be raised if the user try incorrect module)

        self.module_name = module_args[0]
        self.module_parameters = module_args[1:]
        self.module_instance = self.modules[self.module_name](args, *self.module_parameters)
        return self.module_instance

    def add_argparse(self, group_args, comment):
        """ Add the module to the command line parser
        All modules have to be registered before that call
        Args:
            group_args (ArgumentParser):
            comment (str): help to add
        """
        assert len(self.modules.keys())   # Should contain at least 1 module

        keys = list(self.modules.keys())
        group_args.add_argument(
            '--{}'.format(self.name),
            type=str,
            nargs='+',
            default=[keys[0]],  # No defaults optional argument (implemented in the module class)
            help=comment + ' Choices available: {}'.format(', '.join(keys))
        )

    def save(self, config_group):
        """ Save the current module parameters
        Args:
            config_group (dict): dictionary where to write the configuration
        """
        config_group[self.name] = ' '.join([self.module_name] + self.module_parameters)
        # TODO: The module state should be saved here

    def load(self, args, config_group):
        """ Restore the parameters from the configuration group
        Args:
            args (parse_args() returned Obj): the parameters of the models (will be modified)
            config_group (dict): the module group parameters to extract
        Warning: Only restore the arguments. The instantiation is not done here
        """
        setattr(args, self.name, config_group.get(self.name).split(' '))

    def print(self, args):
        """ Just print the current module configuration
        We use the args parameters because the function is called
        before build_module
        Args:
            args (parse_args() returned Obj): the parameters of the models
        """
        print('{}: {}'.format(
            self.name,
            ' '.join(getattr(args, self.name))
        ))
