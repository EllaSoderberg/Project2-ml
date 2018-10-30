import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

noah_dataset = pd.read_csv('MLHW2/datasets/data_noah.csv')
noah_dataset['pitch_type'].value_counts()
X = noah_dataset[['x', 'y']]
y_true = noah_dataset['pitch_type']


def noah_scatterplot(noah_dataset):
    FF_data = noah_dataset[noah_dataset.pitch_type == 'FF']
    CH_data = noah_dataset[noah_dataset.pitch_type == 'CH']
    CU_data = noah_dataset[noah_dataset.pitch_type == 'CU']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)
    ax3 = fig.add_subplot(111)

    ax1.scatter(FF_data['x'], FF_data['y'], color='DarkBlue', label='FF')
    ax2.scatter(CH_data['x'], CH_data['y'], color='DarkGreen', label='CH')
    ax3.scatter(CU_data['x'], CU_data['y'], color='Yellow', label='CU')

    plt.show()


def find_clusters(X, n_clusters, r_state=2):
    # Randomly choose what points to use as centers
    rng = np.random.RandomState(r_state)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X.iloc[i]

    while True:
        # Computing the minimum distance between the center and the set of points around it
        labels = pairwise_distances_argmin(X, centers)

        # Find new centers from mean of points
        new_centers = np.array([X[labels == i].mean(0)for i in range(n_clusters)])

        # See if it matches
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X, 3)

noah_scatterplot(noah_dataset)
plt.scatter(X['x'], X['y'], c=labels, s=50, cmap='viridis')
plt.show()




