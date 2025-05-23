import os
import time
import warnings
import argparse
import shutil

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import hyperopt as hy
from sko.PSO import PSO
from pymatgen.core import Structure, Lattice
import torch


from predict_energy_single import predict_energy
from utils.file_utils import check_and_rename_path
from utils.read_input import ReadInput
from utils.compound_utils import elements_info
from utils.algo_utils import hy_parameter_setting
from utils.print_utils import print_header, print_run_info
from utils.wyckoff_position.get_wyckoff_position import get_all_wyckoff_combination


class PredictStructure:
    @print_header
    def __init__(self, input_file_path='gnoa.in'):
        self.input_config = ReadInput(input_file_path)

        if not self.input_config.is_use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Model Loading Phase
        # self.gn_model = GNModel(is_use_gpu=self.input_config.is_use_gpu).model
        
        #self.nn_model = torch.load(self.input_config.gn_model_path)
        self.modelpath = self.input_config.gn_model_path
        
        
        self.compound = self.input_config.compound
        self.elements = self.input_config.elements
        self.elements_count = self.input_config.elements_count
        self.space_group = list(range(self.input_config.space_group[0], self.input_config.space_group[1] + 1))
        self.wyckoffs_dict, self.max_wyckoffs_count = get_all_wyckoff_combination(self.space_group, self.elements_count)
        self.total_atom_count = self.input_config.total_atom_count
        # self.total_atom_count = sum(self.elements_count)
        self.output_path = os.path.join(self.input_config.output_path, 'results')
        check_and_rename_path(self.output_path)
        self.structures_path = os.path.join(self.output_path, 'structures')
        check_and_rename_path(self.structures_path)
        self.temp_structures_path = os.path.join(self.output_path, 'temp_structures')
        check_and_rename_path(self.temp_structures_path)
        shutil.copy('./crysxpp_main/data/matbench_allcifs_unperturbed_testremoved/atom_init.json',
                    os.path.join(self.temp_structures_path, 'atom_init.json'))

        self.is_sko = None
        self.elements_info = elements_info

        self.step_number = 0
        self.structure_number = 0
        self.all_atoms = []
        self.start_time = time.time()

        self.find_stable_structure()

    def predict_structure_energy(self, kwargs):
        self.step_number += 1

        if self.is_sko:
            _dict = {'a': kwargs[0], 'b': kwargs[1], 'c': kwargs[2],
                     'alpha': kwargs[3], 'beta': kwargs[4], 'gamma': kwargs[5],
                     'sg': int(kwargs[6]), 'wp': kwargs[7]}
            for i in range(int((len(kwargs) - 8) / 3)):
                _dict['x' + str(i + 1)] = kwargs[6 + i * 3 + 0]
                _dict['y' + str(i + 1)] = kwargs[6 + i * 3 + 1]
                _dict['z' + str(i + 1)] = kwargs[6 + i * 3 + 2]
        else:
            _dict = kwargs

        try:
            tmp_structure_file_name = os.path.join(self.temp_structures_path, 'temp.cif')
            self.save_structure_file(self.all_atoms, _dict, file_name=tmp_structure_file_name)
            #time.sleep(0.1)            
            struc = Structure.from_file(tmp_structure_file_name)
            self.atomic_dist_and_volume_limit(struc)
            
            struc.to(fmt='cif', filename=tmp_structure_file_name)
            #time.sleep(0.1)
            result = predict_energy(self.modelpath, self.temp_structures_path)
            #print(self.step_number)
            #print(result)
            #print('I am not here')
            #print('I am here.')
            #print(result)

            self.structure_number += 1
            with open(os.path.join(self.output_path, 'energy_data.csv'), 'a+') as f:
                f.write(','.join([str(self.structure_number),
                                  str(self.step_number),
                                  str(result),
                                  str(_dict['sg']),
                                  str(_dict['wp']),
                                  str(time.time() - self.start_time)]) + '\n')

            structure_file_name = os.path.join(
                self.structures_path,
                '%s_%d_%f_%d_%d_%d.cif' % (self.compound, self.total_atom_count, result, self.structure_number, self.step_number, _dict['sg'])
            )
            shutil.copy(tmp_structure_file_name, structure_file_name)
        except Exception as e:
            print(e)
            result = 999

        if self.is_sko:
            return result
        else:
            return {'loss': result, 'status': hy.STATUS_OK}

    @print_run_info('Predict crystal structure')
    def find_stable_structure(self):
        with open(os.path.join(self.output_path, 'energy_data.csv'), 'w+') as f:
            f.writelines("number,step,energy,sg_number,wp_number,time\n")

        if self.input_config.algorithm in ['tpe', 'rand', 'anneal']:
            self.find_stable_structure_by_hyperopt()
        else:
            self.find_stable_structure_by_sko()

    def find_stable_structure_by_hyperopt(self):
        self.is_sko = False

        if self.total_atom_count % sum(self.elements_count) != 0:
            raise Exception("The parameter `atom_count` or `compound` setting error!")

        a = hy_parameter_setting('a', self.input_config.lattice_a)
        b = hy_parameter_setting('b', self.input_config.lattice_b)
        c = hy_parameter_setting('c', self.input_config.lattice_c)
        alpha = hy_parameter_setting('alpha', self.input_config.lattice_alpha)
        beta = hy_parameter_setting('beta', self.input_config.lattice_beta)
        gamma = hy_parameter_setting('gamma', self.input_config.lattice_gamma)
        sg = hy_parameter_setting('sg', self.input_config.space_group, ptype='int')
        wp = hy_parameter_setting('wp', [0, self.max_wyckoffs_count])
        pbounds = {'a': a, 'b': b, 'c': c,
                   'alpha': alpha, 'beta': beta, 'gamma': gamma,
                   'sg': sg, 'wp': wp}

        i_atoms = 0
        compound_times = self.total_atom_count / sum(self.elements_count)
        for j, a_j in enumerate(self.elements):
            for c_k in range(int(compound_times * self.elements_count[j])):
                self.all_atoms.append(a_j)
                i_atoms += 1
                pbounds['x' + str(i_atoms)] = hy.hp.uniform('x' + str(i_atoms), 0, 1)
                pbounds['y' + str(i_atoms)] = hy.hp.uniform('y' + str(i_atoms), 0, 1)
                pbounds['z' + str(i_atoms)] = hy.hp.uniform('z' + str(i_atoms), 0, 1)

        algorithm = self.input_config.algorithm
        n_init = self.input_config.n_init
        max_step = self.input_config.max_step
        rand_seed = self.input_config.rand_seed

        if algorithm == 'rand':
            print('using Random Search ...')
            algo = hy.rand.suggest
        elif algorithm == 'anneal':
            print('using Simulated Annealing ...')
            algo = hy.partial(hy.anneal.suggest)
        else:
            print('using Bayesian Optimization ...')
            algo = hy.partial(hy.tpe.suggest, n_startup_jobs=n_init)

        if rand_seed == -1:
            rand_seed = None
        else:
            rand_seed = np.random.default_rng(rand_seed)

        trials = hy.Trials()
        best = hy.fmin(fn=self.predict_structure_energy,
                       space=pbounds,
                       algo=algo,
                       max_evals=max_step,
                       trials=trials,
                       rstate=rand_seed  # 随机种子
                       )
        print(best)

    def find_stable_structure_by_sko(self):
        self.is_sko = True

        a = self.input_config.lattice_a
        b = self.input_config.lattice_b
        c = self.input_config.lattice_c
        alpha = self.input_config.lattice_alpha
        beta = self.input_config.lattice_beta
        gamma = self.input_config.lattice_gamma

        lb = [a[0], b[0], c[0], alpha[0], beta[0], gamma[0], 0, 0]
        ub = [a[1], b[1], c[1], alpha[1], beta[1], gamma[1], len(self.space_group), self.max_wyckoffs_count]

        compound_times = int(self.total_atom_count / sum(self.elements_count))
        #print(compound_times)
        for j, a_j in enumerate(self.elements):
            for c_k in range(compound_times * self.elements_count[j]):
                self.all_atoms.append(a_j)
                lb += [0, 0, 0]
                ub += [1, 1, 1]

        max_step = self.input_config.max_step
        n_init = self.input_config.n_init
        rand_seed = self.input_config.rand_seed
        if rand_seed != -1:
            np.random.seed(rand_seed)

        print('using PSO ...')
        pso = PSO(func=self.predict_structure_energy, n_dim=len(lb), pop=n_init, max_iter=max_step, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5, verbose=True)
        pso.run()
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

    def save_structure_file(self, all_atoms, struc_parameters, file_name):
        sg = struc_parameters['sg']
        wp_list = self.wyckoffs_dict[sg]
        wp = wp_list[int(struc_parameters['wp'] * len(wp_list) / self.max_wyckoffs_count)]

        atoms = []
        atom_positions = []
        count = 0
        for i, wp_i in enumerate(wp):
            for wp_i_j in wp_i:
                atoms += [self.elements[i]] * len(wp_i_j)

                for wp_i_j_k in wp_i_j:
                    count += 1
                    if 'x' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('x', str(struc_parameters['x' + str(count)]))
                    if 'y' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('y', str(struc_parameters['y' + str(count)]))
                    if 'z' in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace('z', str(struc_parameters['z' + str(count)]))
                    atom_positions.append(list(eval(wp_i_j_k)))

        if sg in [0, 1, 2]:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])
        elif sg in list(range(3, 15 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=struc_parameters['beta'], gamma=90)
        elif sg in list(range(16, 74 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif sg in list(range(75, 142 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=90)
        elif sg in list(range(143, 194 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['c'],
                                              alpha=90, beta=90, gamma=120)
        elif sg in list(range(195, 230 + 1)):
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['a'], c=struc_parameters['a'],
                                              alpha=90, beta=90, gamma=90)
        else:
            lattice = Lattice.from_parameters(a=struc_parameters['a'], b=struc_parameters['b'], c=struc_parameters['c'],
                                              alpha=struc_parameters['alpha'], beta=struc_parameters['beta'], gamma=struc_parameters['gamma'])

        structure = Structure(lattice, all_atoms, atom_positions)
        structure.to(fmt='cif', filename=file_name)

    def atomic_dist_and_volume_limit(self, struc: Structure):
        atom_radii = []
        for i in self.all_atoms:
            if self.elements_info[i][8] == -1:
                atom_radii.append(100.0 / 100.0)
            else:
                atom_radii.append(float(self.elements_info[i][8]) / 100.0)

        for i in range(self.total_atom_count - 1):
            for j in range(i + 1, self.total_atom_count):
                if struc.get_distance(i, j) < (atom_radii[i] + atom_radii[j]) * 0.4:
                    print('point 1')
                    raise Exception()

        atom_volume = [4.0 * np.pi * r ** 3 / 3.0 for r in atom_radii]
        sum_atom_volume = sum(atom_volume) / 0.55
        if not (sum_atom_volume * 0.4 <= struc.volume <= sum_atom_volume * 2.4):
            print('point 2')
            raise Exception()

        self.vacuum_size_limit(struc=struc.copy(), max_size=7.0)

    @staticmethod
    def vacuum_size_limit(struc: Structure, max_size: float = 10.0):
        def get_foot(p, a, b):
            p = np.array(p)
            a = np.array(a)
            b = np.array(b)
            ap = p - a
            ab = b - a
            result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
            return result

        def get_distance(a, b):
            return np.sqrt(np.sum(np.square(b - a)))

        struc.make_supercell([2, 2, 2], to_unit_cell=False)
        line_a_points = [[0, 0, 0], ]
        line_b_points = [[0, 0, 1], [0, 1, 0], [1, 0, 0],
                         [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, -1], [1, 0, -1], [1, -1, 0],
                         [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
        for a in line_a_points:
            for b in line_b_points:
                foot_points = []
                for p in struc.frac_coords:
                    f_p = get_foot(p, a, b)
                    foot_points.append(f_p)
                foot_points = sorted(foot_points, key=lambda x: [x[0], x[1], x[2]])

                # 转为笛卡尔坐标
                foot_points = np.asarray(np.mat(foot_points) * np.mat(struc.lattice.matrix))
                for fp_i in range(0, len(foot_points) - 1):
                    fp_distance = get_distance(foot_points[fp_i + 1], foot_points[fp_i])
                    if fp_distance > max_size:
                        print('point 3')
                        raise Exception()


if __name__ == '__main__':
    csp = PredictStructure(input_file_path='gnoa.in')
