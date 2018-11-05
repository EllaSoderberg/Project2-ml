import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
from accuracy import accuracy_of_clusters


noah_dataset = pd.read_csv('MLHW2/datasets/data_noah.csv')
print(noah_dataset[['speed', 'pitch_type']])
X = noah_dataset[['x', 'y']]
#X = noah_dataset[['speed', 'sz_bot']]
y_true = noah_dataset['pitch_type']
y_true_num = y_true.replace(['CH', 'CU', 'FF'], [0, 1, 2])
#y_true_num = y_true.replace(['FF', 'CH', 'CU'], [0, 1, 2])


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

    plt.legend()
    plt.show()


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

centers, labels = find_clusters(X, 3)

noah_scatterplot(noah_dataset)
plt.scatter(X['x'], X['y'], c=y_true_num, s=10, cmap='viridis')
#plt.scatter(X['speed'], X['sz_bot'], c=y_true_num, s=10, cmap='viridis')
plt.show()

plt.scatter(X['x'], X['y'], c=labels, s=10, cmap='viridis')
#plt.scatter(X['speed'], X['sz_bot'], c=labels, s=10, cmap='viridis')
plt.show()

print("Total accuracy: {}%".format(accuracy_score(list(y_true_num), list(labels))*100))
labelspd = pd.DataFrame(labels)
accuracy_of_clusters(labelspd, y_true, 3)



