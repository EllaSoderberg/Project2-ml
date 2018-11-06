import pandas as pd
import k_means
from accuracy import accuracy_of_clusters
from sklearn.metrics import accuracy_score
import visualization

noah_dataset = pd.read_csv('MLHW2/datasets/data_noah.csv')


def main(dataset, features, target, label_order, n_clusters, r_state=2, plot=False):
    X = dataset[features]
    y_true = dataset[target]
    y_true_num = y_true.replace(label_order, [0, 1, 2])
    centers, labels = k_means.find_clusters(X, n_clusters, r_state)
    print("Total accuracy: {}%".format(accuracy_score(list(y_true_num), list(labels)) * 100))
    labelspd = pd.DataFrame(labels)
    accuracy_of_clusters(labelspd, y_true, n_clusters)

    if plot:
        visualization.scatterplot(X, features[0], features[1], y_true_num)
        visualization.scatterplot(X, features[0], features[1], labels)
    return X, y_true, y_true_num

if __name__ == '__main__':
    main(noah_dataset, ['x', 'y'], 'pitch_type', ['CH', 'CU', 'FF'], 3, plot=True)
    main(noah_dataset, ['speed', 'sz_bot'], 'pitch_type', ['FF', 'CH', 'CU'], 3, plot=True)
