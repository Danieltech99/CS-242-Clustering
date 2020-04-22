# Evaluation metrics
from sklearn import metrics

class Metrics:
    # Params: label_true, label_pred
    def __init__(self, params):
        self._true = params['true']
        self._pred = params['pred']
    
    # Adjusted Rand Index
    # Pros: random labels have ARI close to 0, bounded [-1, 1], no assumption on cluster structure
    # Cons: requires ground truth
    def ari(self):
        return metrics.adjusted_rand_score(self._true, self._pred)
    
    # Mutual information based scores
    # Pros: random labels have MIs close to 0, [0, 1]
    # Cons: requires ground truth
    def mi(self):
        return metrics.adjusted_mutual_info_score(self._true, self._pred)

    # Homogeneity (each cluster contains only members of a given class)
    # Completeness (all members of a given class are assigned to the same cluster)
    # V-measure (the harmonic mean of homogeneity and completeness)
    # Pros: [0, 1] with 1 as perfect, intuitive qualitative interp, no assumption on cluster structure
    # Cons: not normalized to random labeling, if samples < 1000 or clusters > 10, use ARI isntead
    def homogenity(self):
        return metrics.homogeneity_score(self._true, self._pred)
    
    def completeness(self):
        return metrics.completeness_score(self._true, self._pred)
    
    def v_measure(self):
        return metrics.v_measure_score(self._true, self._pred)