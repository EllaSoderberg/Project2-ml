import matplotlib.pyplot as plt


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


def scatterplot(dataset, x, y, labels):
    plt.scatter(dataset[x], dataset[y], c=labels, s=10, cmap='viridis')
    plt.show()
