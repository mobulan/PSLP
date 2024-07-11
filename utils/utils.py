import math
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def get_file_names_in_subfolders(folder_path):
	file_names = []
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			file_names.append(os.path.join(root, file))
	return file_names


def set_config(config, cfg_file=None):
	if cfg_file:
		config.defrost()
		config.merge_from_file(cfg_file)
		config.freeze()
	return config


def set_seed(seed=None):
	if seed is not None:
		np.random.seed(seed)
		torch.manual_seed(seed)


def set_device():
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	if torch.backends.mps.is_built():
		return 'mps'
	elif torch.cuda.is_available():
		return 'cuda'
	else:
		return 'cpu'



def warp_tqdm(data_loader, disable_tqdm=False):
	if disable_tqdm:
		tqdm_loader = data_loader
	else:
		tqdm_loader = tqdm(data_loader, total=len(data_loader), ascii=True)
	return tqdm_loader


def save_pickle(file, data):
	with open(file, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)
	try:
		# print('features from protoLP')
		if data.__len__() == 2:
			data = data[1]
		labels = [np.full(shape=len(data[key]), fill_value=key)
		          for key in data]
		data = [features for key in data for features in data[key]]
		dataset = dict()
		dataset['features'] = torch.tensor(np.stack(data, axis=0))
		dataset['labels'] = torch.tensor(np.concatenate(labels))
	except:
		dataset = data
	return dataset



def compute_confidence_interval(data):
	"""
	Compute 95% confidence interval
	:param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
	:return: the 95% confidence interval for this data.
	"""
	m = data.mean().item()
	pm = data.std().item() * 1.96 / math.sqrt(data.shape[0])
	return m, pm


if __name__ == '__main__':
	set_device()
