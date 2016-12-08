import numpy as np
from collections import namedtuple
from scipy.io import loadmat
from model import *

class Data(namedtuple('Data', 'features logits softmax labels')):
    pass

def from_logits_file(filename):
    raw_data = loadmat(filename)

    features = raw_data['features'] if 'features' in raw_data.keys() else np.array()
    logits = raw_data['logits']
    labels = raw_data['labels'][:, 0] - 1

    n_classes = logits.shape[1]
    exp_logits = np.exp(logits)
    coeff = np.sum(exp_logits, axis=1)
    softmax = exp_logits / coeff[:, np.newaxis]

    return Model(Data(features, logits, softmax, labels))

def load(filename, binary=False):
    cls = BinaryModel if binary else Model
    data = np.load(filename)
    return cls(namedtuple('GenericDict', data.keys())(**data))
