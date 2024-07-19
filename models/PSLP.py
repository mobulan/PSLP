import torch
from torch import nn
from utils import warp_tqdm


@torch.no_grad()
def PrototypeSoftLabelPropagation(xs, xq, ys, yq, runs, ways, shot, queries, device, config, terminal=True):
	xs, xq, ys, yq = xs.to(device), xq.to(device), ys.to(device), yq.to(device)
	balance = False if config.dirichlet !=0 else True
	model = SoftLabelPropagation(runs, ways, shot, queries, ys,config.lam,config.alpha,config.link_rate,
						  config.update_rate,config.k_hop, config.epochs,terminal,balance).to(device)
	result = model(xs, xq, ys, yq)

	return result


class SoftLabelPropagation(nn.Module):
	def __init__(self, runs, ways, shot, queries, ys, lam=10,
				 alpha=0.7, link_rate=6, update_rate=0.6,
				 k_hop=4, epochs=10, terminal=True,balance=True):
		super().__init__()
		self.runs = runs
		self.ways = ways
		self.queries = queries
		self.shot = shot
		self.proto = None
		self.lam = lam
		self.ns = ways * shot
		self.qs = ways * queries
		self.ys = ys
		self.eps = 1e-8
		####
		self.alpha = alpha
		self.link_rate = link_rate
		self.update_rate = update_rate
		self.k_hop = k_hop
		####
		self.epochs = epochs
		self.terminal = terminal
		self.tau = 0.0001
		self.balance = balance

	def forward(self, xs, xq, ys, yq):
		features = torch.cat((xs, xq), dim=1)
		labels = torch.cat((ys, yq), dim=1)
		prob_query = self.preidct(features, labels)
		return prob_query

	def preidct(self, features, labels):
		device = features.device
		samples, dim = features.shape[1],features.shape[-1]
		Iden = torch.eye(samples, device=device).unsqueeze(0)
		for i in range(int(self.k_hop)):
			dist = self.get_distance(features,features)
			features, W = self.graph_convolution(features,dist,self.link_rate)
			W = self.build_graph(W, self.link_rate)
		self.proto = features[:, :self.ns].reshape(self.runs, self.ways, self.shot, -1).mean(dim=2)


		W = self.build_graph(W,self.link_rate)
		Inverse_W = torch.inverse(Iden - self.alpha * W)

		for i in warp_tqdm(range(int(self.epochs)), not self.terminal):
			# Build Graph
			# Get Y in the Algorithm
			Z = self.get_prob(features, self.proto,True)

			# Soft Lable Propagation
			Z = Inverse_W  @  Z

			# Normalize(if imbalance, we only use Z=Z/Z.sum(-1)）
			Z = compute_optimal_transport(Z, self.ns, self.ys, 1e-3, self.balance)
			# update Prototype
			self.update_prototype(Z, features, self.update_rate)

		# get final accuracy and return it
		Z = self.get_prob(features, self.proto)
		olabels = Z.argmax(dim=2)
		matches = labels.eq(olabels).float()
		acc_test = matches[:, self.ns:].mean(1)
		return acc_test

	def get_prob(self, features, proto, iter=False):
		# compute squared dist to centroids [n_runs][n_samples][n_ways]
		dist = torch.cdist(features, proto).pow(2)
		P = torch.zeros_like(dist)
		Pq = dist[:, self.ns:]
		Pq = torch.exp(- self.lam * Pq)
		Pq = compute_optimal_transport(Pq, 0, self.ys, 1e-3,self.balance)

		P[:, self.ns:] = Pq
		P[:, :self.ns].scatter_(2, self.ys.unsqueeze(2), 1)
		return P

	def graph_convolution(self,X, A,link_rate=6.):
		samples = X.shape[1]
		Iden = torch.eye(samples, device=X.device).unsqueeze(0)

		A = keep_top_k_row(A, link_rate)
		D = A.sum(-1).pow(-0.5)
		W = D.unsqueeze(1) * A * D.unsqueeze(-1)
		L = 0.5 * Iden + 0.5 * W
		G = torch.matrix_power(L, 2)
		X = G @ X
		return X, G

	def get_distance(self, features1,features2):
		# get pairwise distance of samples
		dist = torch.cdist(features1, features2).pow(2)
		dist = torch.exp(- dist * self.lam)
		return dist

	def build_graph(self,W, link_rate=0.4):
		samples = W.shape[1]
		Iden = torch.eye(samples, device=W.device).unsqueeze(0)

		# Set 0 to Diagnose.
		W = W * (1 - Iden)

		# Symmetrically normalize
		D = W.sum(-1).pow(-0.5)

		W = D.unsqueeze(-2) * W * D.unsqueeze(-1)

		return W
	def update_prototype(self, transport, features, alpha):
		new_proto = transport.permute(0, 2, 1).matmul(features).div(transport.sum(dim=1).unsqueeze(2))
		self.proto = (1-alpha) * self.proto + alpha * new_proto

def compute_optimal_transport(M, n_lsamples, labels, epsilon=1e-6,class_balance=True):
	# r : [runs, total samples], c : [runs, ways]
	# n samples, m ways
	n_runs, n, ways = M.shape
	r = torch.ones(n_runs, n, device=M.device)
	c = torch.ones(n_runs, ways, device=M.device) * n // ways
	u = torch.zeros(n_runs, n, device=r.device)
	P = M
	maxiters = 1000
	iters = 1

	# Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
	while torch.max(torch.abs(u - P.sum(2))) > epsilon:
		u = P.sum(2)
		P *= (r / u).view((n_runs, -1, 1))
		if class_balance:
			P *= (c / P.sum(1)).view((n_runs, 1, -1))
		P[:, :n_lsamples].fill_(0)
		P[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
		if iters == maxiters:
			break
		iters = iters + 1
	return P
def keep_top_k_row(matrix, k):
	batch_size, num_nodes, _ = matrix.shape
	values, indices = torch.topk(matrix, k, dim=-1)
	result = torch.zeros_like(matrix)
	result.scatter_(-1,indices,values)
	return result
