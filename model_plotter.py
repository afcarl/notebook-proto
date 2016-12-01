import numpy as np
from matplotlib import pyplot as plt

class ModelPlotter():
    @staticmethod  
    def plot_calibration(self, delta=.1):
        bins, bin_sizes = self.bin_sizes(delta)
        _, bin_accuracies = self.bin_accuracies(delta)
        f, (size_ax, acc_ax, calib_ax) = plt.subplots(1, 3, figsize=(12, 4))

        size_ax.bar(bins[:-1], bin_sizes, width=bins[1]-bins[0])
        size_ax.set_xlim([0, 1])
        size_ax.set_xlabel('Confidence')
        size_ax.set_ylabel('Bin Size')
        size_ax.set_title('Bin Size Histogram')
        
        acc_ax.bar(bins[:-1], bin_accuracies, width=bins[1]-bins[0])
        acc_ax.plot([0, 1], [0, 1], linestyle='--', color='red')
        acc_ax.set_xlim([0, 1])
        acc_ax.set_ylim([0, 1])
        acc_ax.set_xlabel('Confidence')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.set_title('Accuracy Histogram')
        
        deltas = 1 / np.logspace(0, math.log(50), 51)
        calib_ax.plot(deltas, [self.calibration(delta=delta) for delta in deltas])
        calib_ax.set_xlabel(r'$\delta$')
        calib_ax.set_ylabel(r'$\epsilon$')
        calib_ax.set_title('Calibration')
        
        return f

