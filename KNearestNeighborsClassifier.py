import numpy as np
from usearch.index import search, MetricKind

class KNearestNeighborsClassifier():
    '''
    Classifier using the k-nearest neighbors algorithm.
    '''
    def __init__(self, n_neighbors = 5):
        '''
        Args:
            n_neighbors: The number of neighbors to count for
        '''
        self.n_neighbors = n_neighbors

    def fit(self, X:np.array, y:np.array):
        '''
        Fits data on a k-nearest neighbors model.

        Args:
            X: Training data
            y: Target values
        '''
        self.X_ = X
        self.y_ = y
        return

    def predict(self, X:np.array):
        '''
        Predict using the fitted k-nearest neighbors model.

        Args:
            X: Sample data
        '''
        preds = []
        # Iterate through every observation (row) in X
        for obs in X:
            # Get the indices of the k-nearest neighbors
            matches = search(self.X_, obs.reshape(1,-1), len(self.X_), MetricKind.L2sq, exact=True)
            neighbors = matches.keys[:self.n_neighbors]

            y_vals = self.y_[neighbors] # Get the target values of the k-nearest neighbors

            # Get the most frequently occurring target value in y_vals as the prediction for the observation
            unique_y = np.unique(y_vals)
            y_counts = dict(zip(unique_y, [0]*len(unique_y))) # Frequency counts of each unique y value (in y_vals)
            for val in y_vals:
                y_counts[int(val)] += 1
            preds.append(max(y_counts, key=y_counts.get))
        return np.array(preds)
    