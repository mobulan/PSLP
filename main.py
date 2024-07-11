import time

import torch.cuda

from eval import Evaluator
from scripts import _C
from utils import *


def main(write=True, search_mode=None):
	start = time.time()
	# write = write
	write = True
	# write = False
	terminal = True

	exp_name = 'Balanced'
	exp_scripts = 'pslp'
	log_name = 'PSLP_Balanced'

	config_list, log_path, write = build_experiments(exp_name, log_name, write, exp_scripts)
	results = []
	for f in config_list:
		config = _C.clone()
		config = set_config(config, f)
		if search_mode:
			terminal = False
			for key in search_mode.keys():
				assert config[key]
				config[key] = search_mode[key]
		result = evaluate(config, write, terminal)
		results.append(result)
	print(f'Total Datasets Time: {int((time.time() - start) // 60):d} min '
	      f'{int((time.time() - start) % 60):d} s')
	return results


def evaluate(config, write=None, terminal=True):
	T1 = time.time()
	device = set_device()
	set_seed(config.seed)

	evaluator = Evaluator(config, device)
	results = evaluator.full_evaluation(write, terminal)
	# print(f'running time: {time.time() - T1:.1f}')
	return results


def build_experiments(exp_name='unname', log_name='run', write=None, exp_scripts=''):
	config_list = []
	root = 'scripts'
	root = os.path.join(root, exp_scripts)
	file_names = get_file_names_in_subfolders(root)
	for file_name in file_names:
		config_list.append(file_name)
	config_list = [x for x in config_list if 'init' not in x and 'pycharm' not in x]

	os.makedirs(os.path.join('output', exp_name), exist_ok=True)
	log_path = os.path.join('output', exp_name, log_name +
	                        time.strftime(' %m-%d_%H-%M', time.localtime()) + '.txt')
	write = Log(log_path, write)

	return config_list, log_path, write


def grid_search():
	# search_params = {'alpha':0.0,'link_rate':0.0,'update_rate':0.0}
	best_params = [0.0, 0.0, 0.0]
	best_sum_acc = 0.0
	for a in torch.linspace(0.1, 0.9, 9):
		for b in torch.linspace(0.1, 1.0, 10):
			for c in torch.linspace(1, 10, 10):
				print(best_params)
				search_params = {'alpha': a, 'update_rate': b, 'link_rate': c}
				print(search_params)
				results = main(False, search_params)

				sum_acc = torch.stack(results, 0).sum()
				print(f'Best: {best_sum_acc:.3f} Acc: {sum_acc:.3f}')
				if sum_acc > best_sum_acc:
					best_sum_acc = sum_acc
					best_params = [a, b, c]


if __name__ == '__main__':
	# grid_search()
	main()
