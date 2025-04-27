# -*- coding: utf-8 -*-
# ========================================================================
#  2021/04/26 下午 09:57
#                 _____   _   _   _____   __   _   _____
#                /  ___| | | | | | ____| |  \ | | /  ___|
#                | |     | |_| | | |__   |   \| | | |
#                | |     |  _  | |  __|  | |\   | | |  _
#                | |___  | | | | | |___  | | \  | | |_| |
#                \_____| |_| |_| |_____| |_|  \_| \_____/
# ------------------------------------------------------------------------
#
# Read input parameters from gnoa.in
#
# ========================================================================


import os
import ast
import configparser

from utils.print_utils import print_run_info
from utils.compound_utils import compound_split


class ReadInput:
    @print_run_info("Read input file")
    def __init__(self, input_file_path='gnoa.in', input_config=None):
        if input_file_path:
            if not os.path.isfile(input_file_path):
                raise IOError("Could not find `gnoa.in` file!")

            config = configparser.RawConfigParser()
            config.read(input_file_path, encoding='utf-8')
        elif input_file_path is None and input_config:
            config = input_config
        else:
            raise RuntimeError('Please input some thing!')

        self._compound = config.get('BASE', 'compound').replace(' ', '')
        self.elements, self.elements_count = compound_split(self._compound)
        # self._total_atom_count = config.getint('BASE', 'atom_count')
        self._total_atom_count = sum(self.elements_count)
        self._gn_model_path = config.get('BASE', 'gn_model_path')
        self._output_path = config.get('BASE', 'output_path')
        self._is_use_gpu = config.getboolean('BASE', 'use_gpu')

        self._space_group = ast.literal_eval(config.get('LATTICE', 'space_group'))
        self._lattice_a = ast.literal_eval(config.get('LATTICE', 'lattice_a'))
        self._lattice_b = ast.literal_eval(config.get('LATTICE', 'lattice_b'))
        self._lattice_c = ast.literal_eval(config.get('LATTICE', 'lattice_c'))
        self._lattice_alpha = ast.literal_eval(config.get('LATTICE', 'lattice_alpha'))
        self._lattice_beta = ast.literal_eval(config.get('LATTICE', 'lattice_beta'))
        self._lattice_gamma = ast.literal_eval(config.get('LATTICE', 'lattice_gamma'))

        self._algorithm = config.get('PROGRAM', 'algorithm')
        self._n_init = config.getint('PROGRAM', 'n_init')
        self._max_step = config.getint('PROGRAM', 'max_step')
        self._rand_seed = config.getint('PROGRAM', 'rand_seed')

        print("  Compound: %s    Total atoms count: %d" % (self._compound, self._total_atom_count))
        print('  a:', self._lattice_a, '  b:', self._lattice_b, '  c:', self._lattice_c)
        print('  alpha:', self._lattice_alpha, '  beta:', self._lattice_beta, '  gamma:', self._lattice_gamma)
        print('  algorithm: %s    Max step: %d' % (self._algorithm, self._max_step))

    @property
    def gn_model_path(self):
        return self._gn_model_path

    @gn_model_path.setter
    def gn_model_path(self, gn_model_path):
        self._gn_model_path = gn_model_path

    @property
    def is_use_gpu(self):
        return self._is_use_gpu

    @is_use_gpu.setter
    def is_use_gpu(self, is_use_gpu):
        self._is_use_gpu = is_use_gpu

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        self._output_path = output_path

    @property
    def compound(self):
        return self._compound

    @compound.setter
    def compound(self, compound):
        self._compound = compound

    @property
    def total_atom_count(self):
        return self._total_atom_count

    @total_atom_count.setter
    def total_atom_count(self, total_atom_count):
        self._total_atom_count = total_atom_count

    @property
    def space_group(self):
        return self._space_group

    @space_group.setter
    def space_group(self, space_group):
        self._space_group = space_group

    @property
    def lattice_a(self):
        return self._lattice_a

    @lattice_a.setter
    def lattice_a(self, lattice_a):
        self._lattice_a = lattice_a

    @property
    def lattice_b(self):
        return self._lattice_b

    @lattice_b.setter
    def lattice_b(self, lattice_b):
        self._lattice_b = lattice_b

    @property
    def lattice_c(self):
        return self._lattice_c

    @lattice_c.setter
    def lattice_c(self, lattice_c):
        self._lattice_c = lattice_c

    @property
    def lattice_alpha(self):
        return self._lattice_alpha

    @lattice_alpha.setter
    def lattice_alpha(self, lattice_alpha):
        self._lattice_alpha = lattice_alpha

    @property
    def lattice_beta(self):
        return self._lattice_beta

    @lattice_beta.setter
    def lattice_beta(self, lattice_beta):
        self._lattice_beta = lattice_beta

    @property
    def lattice_gamma(self):
        return self._lattice_gamma

    @lattice_gamma.setter
    def lattice_gamma(self, lattice_gamma):
        self._lattice_gamma = lattice_gamma

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    @property
    def n_init(self):
        return self._n_init

    @n_init.setter
    def n_init(self, n_init):
        self._n_init = n_init

    @property
    def max_step(self):
        return self._max_step

    @max_step.setter
    def max_step(self, max_step):
        self._max_step = max_step

    @property
    def rand_seed(self):
        return self._rand_seed

    @rand_seed.setter
    def rand_seed(self, rand_seed):
        self._rand_seed = rand_seed
