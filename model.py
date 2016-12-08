import os, sys
import math
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from scipy.optimize import linprog

# class Model(namedtuple('Model', 'logits softmax labels features')):
class Model(namedtuple('Model', 'data')):
    def num_samples(self):
        return len(self.data.labels)
    
    def accuracy(self):
        return self.accuracies().mean()

    def overconfidence(self):
        return self.confidences().mean() - self.accuracy()
    
    def calibration(self, delta=0.1):
        _, bin_sizes = self.bin_sizes(delta)
        return np.sum([size/self.num_samples() * calibration
                       for size, calibration in zip(bin_sizes, self.calibrations(delta))])
    
    def confidences(self):
        return np.max(self.data.softmax, axis=1)

    def predictions(self):
        return np.argmax(self.data.softmax, axis=1)

    def accuracies(self):
        return self.predictions() == self.data.labels
    
    def calibrations(self, delta=.1):
        bins, bin_accuracies = self.bin_accuracies(delta)
        return np.array([max(0, bin_accuracy - high_bin, low_bin - bin_accuracy)
            for low_bin, high_bin, bin_accuracy in zip(bins[0:-1], bins[1:], bin_accuracies)])
    
    def bin_accuracies(self, delta=.1):
        bins, all_hist = self.bin_sizes(delta)
        correct_hist, _ = np.histogram(self.confidences()[self.accuracies()], bins=bins)
        return bins, correct_hist / (all_hist + 1e-6)
    
    def bin_sizes(self, delta=.1):
        bins = np.arange(0, 1 + 1e-4, delta)
        all_hist, _ = np.histogram(self.confidences(), bins=bins)
        return bins, all_hist.astype(float)
    
    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'Overconfidence:\t%.3f' % self.overconfidence(),
            'Calib. (d=0.1):\t%.3f' % self.calibration(delta=0.1),
        ])

class BinaryModel(Model):
    def overconfidence(self):
        return self.data.softmax.mean() - self.data.label.mean()

    def confidences(self):
        return self.data.softmax

    def predictions(self):
        return self.data.softmax > 0.5

    def bin_accuracies(self, delta=.1):
        bins, all_hist = self.bin_sizes(delta)
        correct_hist, _ = np.histogram(self.confidences()[self.accuracies()], bins=bins)
        return bins, correct_hist / (all_hist + 1e-6)
