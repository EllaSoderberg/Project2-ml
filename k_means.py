import numpy as np
from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, r_state=2):
    # Randomly choose what points to use as centers
    rnd = np.random.RandomState(r_state)
    i = rnd.permutation(X.shape[0])[:n_clusters]
    center_points = X.iloc[i]

    while True:
        # Finding the label of the points closest to the center points
        labels = pairwise_distances_argmin(X, center_points)

        # Find new centers from mean of points
        new_centers = np.array([X[labels == a].mean(0)for a in range(n_clusters)])

        # See if it matches
        if np.all(center_points == new_centers):
            break
        center_points = new_centers

    return center_points, labels




