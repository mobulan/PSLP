from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = 'mini'
_C.model_name = 'resnet12'
_C.shots = [1, 5]
_C.ways = 5
_C.queries = 15
_C.runs = 10000
_C.lam = 10
_C.seed = 0
_C.alpha = 0.7
_C.link_rate = 6
_C.update_rate = 0.6
_C.dirichlet = 0
_C.k_hop = 2
_C.K = 40
_C.epochs = 10
_C.num_classes = 64
_C.resume = False
_C.evaluate = True
_C.model_path = ''
_C.data_path = f'data'
_C.feature_path = f'pretrained'
_C.classifier = 'pslp'