import math
import os

import numpy as np
import torch

from utils.utils import warp_tqdm


class Tasks():
	def __init__(self, runs, ways, shot, queries, extracted_features, preload, device):
		self.runs = runs
		self.ways = ways
		self.shot = shot
		self.queries = queries
		self.ns = ways * shot
		self.nq = ways * queries
		self.data, self.labels = None, None
		self.min_examples = 0
		self.preload = preload
		self.device = device
		if not os.path.isfile(preload):
			self.balance_samples(extracted_features, device)

	def generate_tasks(self, unbalanced=0, process_dim=40):
		"""
		task: a dict with:
			xs:support features     [runs, ways × shots, dim]
		    ys:support labels       [runs, ways × shots]
			xq:query features       [runs, ways × queries, dim]
			yq:query labels         [runs, ways × shots]
		"""

		if os.path.isfile(self.preload):
			task = torch.load(self.preload)
			# 如果成功读取字典形式，这里的Try就会失败，所以直接返回
			try:
				#如果读取失败，即字典没了，就会重新构建字典
				x_support, x_query = task[0][:, :self.ns], task[0][:, self.ns:]
				y_support, y_query = task[1][:, :self.ns], task[1][:, self.ns:]
				task = {'x_s': x_support, 'y_s': y_support,
				        'x_q': x_query, 'y_q': y_query}
			except:
				pass

		# 如果没有构建过任务
		else:
			x_support, x_query, y_support, y_query = [], [], [], []
			for i in warp_tqdm(range(self.runs)):
				xs, xq, ys, yq = self.single_task(self.data, unbalanced, )
				x_support.append(xs)
				x_query.append(xq)
				y_support.append(ys)
				y_query.append(yq)
			x_support, x_query = torch.stack(x_support, 0), torch.stack(x_query, 0)
			y_support, y_query = torch.stack(y_support, 0), torch.stack(y_query, 0)

			if process_dim:
				ns, nq = x_support.shape[1], x_query.shape[1]
				x = torch.cat((x_support, x_query), 1)
				x = preprocess(self.preload, x, process_dim)
				x_support, x_query = x[:, :ns], x[:, ns:]

			task = {'x_s': x_support, 'y_s': y_support,
			        'x_q': x_query, 'y_q': y_query}

			torch.save(task, self.preload)

		return task

	def single_task(self, data, alpha=2):
		classes = np.random.permutation(np.arange(data.shape[0]))[:self.ways]
		shuffle_indices = np.arange(self.min_examples)
		dataset = torch.zeros((self.ways, self.shot, data.shape[2]))
		label = torch.zeros(self.ways, self.shot, dtype=torch.int64)
		querySet = []
		labelSet = []

		if alpha == 0:
			for i in range(self.ways):
				shuffle_indices = np.random.permutation(shuffle_indices)
				dataset[i] = data[classes[i], shuffle_indices, :][:self.shot]
				querySet.append(data[classes[i], shuffle_indices, :][self.shot:self.shot + self.queries])
				label[i] = i
				labelSet.append(i * torch.ones(self.queries))
		else:
			alpha = alpha * np.ones(self.ways)
			pro = get_dirichlet_query_dist(alpha, 1, self.ways, self.queries * self.ways)[0]

			for i in range(self.ways):
				shuffle_indices = np.random.permutation(shuffle_indices)

				# dataset[i] = data[classes[i], shuffle_indices,
				#                   :][:cfg['shot']+self.queries]
				dataset[i] = data[classes[i], shuffle_indices, :][:self.shot]
				if pro[i] > data[classes[i], shuffle_indices, :].shape[0]:
					dist = pro[i] - data[classes[i], shuffle_indices, :].shape[0]
					query = data[classes[i], shuffle_indices, :][:pro[i]]
					query_extra = data[classes[i], shuffle_indices[:dist], :][:pro[i]]
					query = torch.cat((query, query_extra), dim=0)
				else:
					query = data[classes[i], shuffle_indices, :][:pro[i]]
				querySet.append(query)
				label[i] = i
				label_que = i * torch.ones(pro[i])
				labelSet.append(label_que)
		querys = torch.cat(querySet, dim=0).to(self.device)
		labels = torch.cat(labelSet, dim=0).to(self.device)
		supports = dataset.reshape(-1, data.shape[2]).to(self.device)  # [class11,class12...,class1shot, class21,]
		support_labels = label.reshape(-1)
		# dataset = torch.cat((dataset.reshape(-1, data.shape[2]), querys), dim=0)
		# labels = torch.cat((label.reshape(-1), labels), dim=0)
		return supports, querys, support_labels, labels

	def balance_samples(self, extracted_features, device='cpu'):
		extracted_features['labels'] = extracted_features['labels'].to(device)
		extracted_features['features'] = extracted_features['features'].to(device)
		# Computing the number of items per class in the extracted_features
		_min_examples = extracted_features["labels"].shape[0]
		for i in range(extracted_features["labels"].shape[0]):
			if torch.where(extracted_features["labels"] == extracted_features["labels"][i])[0].shape[0] > 0:
				_min_examples = min(_min_examples, torch.where(
					extracted_features["labels"] == extracted_features["labels"][i])[0].shape[0])
		# print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

		# Generating data tensors
		data = torch.zeros((0, _min_examples, extracted_features["features"].shape[1]), device=device)
		labels = extracted_features["labels"].clone()
		while labels.shape[0] > 0:
			indices = torch.where(extracted_features["labels"] == labels[0])[0]
			data = torch.cat([data, extracted_features["features"][indices, :]
			[:_min_examples].view(1, _min_examples, -1)], dim=0)
			indices = torch.where(labels != labels[0])[0]
			labels = labels[indices]
		self.data, self.labels = data, labels
		self.min_examples = _min_examples

	def reverse_feature(self, z_s, z_q, y_s, y_q, shot):
		z_s = z_s.view(self.runs, self.ways, shot, -1)
		z_q = z_q.view(self.runs, self.ways, self.queries, -1)
		features = torch.cat((z_s, z_q), 2)
		features = features.permute(0, 2, 1, 3).reshape(self.runs, self.ways * (shot + self.queries), -1)

		y_s, y_q = y_s.view(self.runs, self.ways, -1), y_q.view(self.runs, self.ways, -1)
		labels = torch.cat((y_s, y_q), 2)
		labels = labels.permute(0, 2, 1).reshape(self.runs, -1)
		return features, labels


def convert_prob_to_samples(prob, q_shot):
	global frac_clos
	prob = prob * q_shot
	for i in range(len(prob)):
		if sum(np.round(prob[i])) > q_shot:
			while sum(np.round(prob[i])) != q_shot:
				idx = 0
				for j in range(len(prob[i])):
					frac, whole = math.modf(prob[i, j])
					if j == 0:
						frac_clos = abs(frac - 0.5)
					else:
						if abs(frac - 0.5) < frac_clos:
							idx = j
							frac_clos = abs(frac - 0.5)
				prob[i, idx] = np.floor(prob[i, idx])
			prob[i] = np.round(prob[i])
		elif sum(np.round(prob[i])) < q_shot:
			while sum(np.round(prob[i])) != q_shot:
				idx = 0
				for j in range(len(prob[i])):
					frac, whole = math.modf(prob[i, j])
					if j == 0:
						frac_clos = abs(frac - 0.5)
					else:
						if abs(frac - 0.5) < frac_clos:
							idx = j
							frac_clos = abs(frac - 0.5)
				prob[i, idx] = np.ceil(prob[i, idx])
			prob[i] = np.round(prob[i])
		else:
			prob[i] = np.round(prob[i])
	return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
	alpha = np.full(n_ways, alpha)
	prob_dist = np.random.dirichlet(alpha, n_tasks)
	return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)


def QRreduction(datas):
	ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
	ndatas = ndatas.permute(0, 2, 1)
	return ndatas


def SVDreduction(ndatas, K, device='cpu'):
	if device == 'cpu':
		ndatas = ndatas.to('cpu')
	else:
		ndatas = ndatas.to('cuda')
	_, s, v = torch.linalg.svd(ndatas)
	v = v.mH
	ndatas = ndatas.matmul(v[:, :, :K])
	return ndatas


def center_norm(datas, center=True):
	if center:
		datas = datas - datas.mean(1, keepdim=True)
	datas = datas / torch.norm(datas, dim=2, keepdim=True)
	return datas


def preprocess(model_name, data, K=40):
	if 'resnet12' not in model_name:
		beta = 0.5
		data = torch.pow(data + 1e-6, beta)
	data = center_norm(data, False)
	data = SVDreduction(data, K)
	data = center_norm(data)
	return data
