import os
import time

import torch

from models import CLASSIFIERS
from utils import Dataset_Name
from utils import Tasks
from utils import load_pickle
from utils import compute_confidence_interval


class Evaluator:
	def __init__(self, config, device):
		self.config = config
		self.runs = config.runs
		self.ways = config.ways
		self.queries = config.queries
		self.shots = config.shots
		self.device = device
		# self.device =
		self.mode = 'test'
		self.data_path = os.path.join(config.data_path, Dataset_Name[config.dataset])
		self.split_path = os.path.join('data', 'split', config.dataset)
		self.feature_path = os.path.join(config.feature_path, 'features', config.dataset)
		self.preload_tasks = os.path.join(config.feature_path, 'preload', config.dataset)
		self.model_name = config.model_name
		self.classifier = config.classifier
		self.dataset = config.dataset

		os.makedirs(self.preload_tasks, exist_ok=True)

	def full_evaluation(self, write, terminal=True):
		if terminal:
			print(
				f"=> Running: {self.classifier} {self.dataset} {self.model_name} {self.queries} {self.config.dirichlet}")

		extracted_features = self.model_extract(self.model_name, self.feature_path)
		results = torch.zeros(len(self.shots))

		for i, shot in enumerate(self.shots):
			if terminal:
				print(f' => Running {shot} Shot:')
			preload_features = os.path.join(self.preload_tasks, f'{self.model_name}_{shot}_{self.config.dirichlet}_{self.queries}.pth')
			task_gen = Tasks(self.runs, self.ways, shot, self.queries,
			                 extracted_features, preload_features, self.device)

			tasks = task_gen.generate_tasks(self.config.dirichlet)
			T1 = time.time()
			result = self.run_task(tasks, shot, terminal)
			# print(f'running time: {time.time() - T1:.2f}')
			mean, conf = compute_confidence_interval(result)
			if terminal:
				write.info(f"{self.dataset} \t{self.model_name} shot {shot} accuracy:\t "
				           f"{100 * mean:0.2f} +- {100 * conf:0.2f}")
			results[i] = mean
		return results

	def run_task(self, tasks, shot, terminal=True):
		# Extract support and query
		ys, yq = tasks['y_s'], tasks['y_q']
		xs, xq = tasks['x_s'], tasks['x_q']

		ys = ys.long()
		yq = yq.long()

		result = CLASSIFIERS[self.classifier](
			xs, xq, ys, yq, self.runs, self.ways, shot,
			self.queries, self.device, self.config, terminal)

		return result

	def model_extract(self, model_name, feature_path):

		# Load features from memory if previously saved ...
		save_dir = os.path.join(feature_path)
		filepath = os.path.join(save_dir, f'{model_name}.plk')
		if os.path.isfile(filepath):
			extracted_features = load_pickle(filepath)
			# print(" ==> Features loaded from {}".format(filepath))
			return extracted_features
		# ... otherwise just extract them
		else:
			raise Exception(f'no .pth file found at {filepath}')
