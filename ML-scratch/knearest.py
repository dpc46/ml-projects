import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    
    def _predict(self, x):
        """
        Computes Eucledian distances for all data to the
        current data point.

        Finds the k nearest neighbours and predicts category
        based on the majority categorisation for k nearest neighbours
        """
        # call the euclidean_distance function between current vector and all X_train vectors
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get the indices of only the k nearest neighbours
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
