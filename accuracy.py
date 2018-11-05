import pandas as pd


def accuracy_of_clusters(labels, target, n):
    data = pd.concat([labels, target], axis=1)
    data.columns = ["labels", 'target']
    #print(data.head())
    categories = data.target.unique()
    associations = [0]*n
    # determine which categories each cluster represents the best
    for i in range(categories.size):
        max = 0
        for j in range(n):
            curr = data[(data['labels'] == j) & (data['target'] == categories[i])].size
            if curr > max:
                max = curr
                lab = j
        associations[lab] = categories[i]
    #print(associations)

    # calculating accuracy for each category
    for i in range(n):
        correct = data[(data['labels'] == i) & (data['target'] == associations[i])].size
        total = data[data['labels'] == i].size
        accu = float(correct)/total
        #print("corr",correct,"total",total,"acc",accu)
        accu *= 100
        print("Accuracy of {} is: {}%".format(associations[i], accu))

    return
