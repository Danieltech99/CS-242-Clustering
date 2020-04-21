import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from Testing.__init__ import DataSetCollection,DataSet

collection = DataSetCollection()
data, labels = collection.get_set("blobs", "vlow")
dataset = DataSet(data, labels) 
print(data.shape)   

np.random.seed(42)

# X_digits, y_digits = load_digits(return_X_y=True)
# data = scale(X_digits)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    print(linkage_matrix)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=0.9*max(linkage_matrix[:,2]), **kwargs)


iris = load_iris()
X = data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=50000, n_clusters=None)

model = model.fit(X)
print(model.labels_)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()