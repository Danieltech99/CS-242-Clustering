import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import scale

sk_datasets = {}
sk_labels = {}

def plot_data_labels_dict(data_labels_dict, n_datasets, cmap="bwr"):
    fig, axes = plt.subplots(n_datasets, 2, figsize=(10, 24/5*n_datasets), sharex=True, sharey=True)
    for i, (category, (data, labels)) in enumerate(data_labels_dict.items()):
        axes[i][0].scatter(data[:, 0], data[:, 1], s=1)
        axes[i][1].scatter(data[:, 0], data[:, 1], s=1, c=labels, cmap=cmap)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

noise_categories = ["vhigh", "high", "med", "low", "vlow"]
noise_levels = [0.16, 0.14, 0.12, 0.1, 0.08]

circles_dict = {category:datasets.make_circles(n_samples=10000, noise=level, factor=0.4)
              for category, level in zip(noise_categories, noise_levels)}

plot_data_labels_dict(circles_dict, len(noise_categories))

for category, (data, labels) in circles_dict.items():
    sk_datasets["circles_" + category + "_noise"] = data
    sk_labels["circles_" + category + "_noise"] = labels