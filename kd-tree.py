from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, value, left_child, right_child):
        self.value = value
        self.left_child = left_child
        self.right_child = right_child


def kdtree(points, axis=0):

    if len(points) == 0:  # Recursion ending condition
        return None

    points.sort(key=itemgetter(axis))
    median = len(points) // 2

    return Node(points[median], kdtree(points[:median], 1 - axis), kdtree(points[median + 1:], 1 - axis))


def plot_tree_node(tree, min_x, max_x, min_y, max_y, prev_node, branch, axis=0):

    cur_node = tree.value  # current tree's node
    left_branch = tree.left_child  # its left branch
    right_branch = tree.right_child  # its right branch

    ln_width = 2

    if axis == 0:

        if branch is not None and prev_node is not None:

            if branch:
                max_y = prev_node[1]
            else:
                min_y = prev_node[1]

        plt.plot([cur_node[0], cur_node[0]], [min_y, max_y], linestyle='-', color='red', linewidth=ln_width)

    elif axis == 1:

        if branch is not None and prev_node is not None:

            if branch:
                max_x = prev_node[0]
            else:
                min_x = prev_node[0]

        plt.plot([min_x, max_x], [cur_node[1], cur_node[1]], linestyle='-', color='blue', linewidth=ln_width)

    plt.plot(cur_node[0], cur_node[1], 'ko')

    if left_branch is not None:
        plot_tree_node(left_branch, min_x, max_x, min_y, max_y, cur_node, True, 1 - axis)

    if right_branch is not None:
        plot_tree_node(right_branch, min_x, max_x, min_y, max_y, cur_node, False, 1 - axis)


def prepare_plot(size, min_val, max_val, delta):
    plt.figure("K-d Tree", figsize=(size, size))
    plt.axis([min_val - delta, max_val + delta, min_val - delta, max_val + delta])

    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    plt.xticks([i for i in range(min_val - delta, max_val + delta, 1)])
    plt.yticks([i for i in range(min_val - delta, max_val + delta, 1)])


def plot_tree(tree, min_val, max_val, delta, axis):

    prepare_plot(5, min_val, max_val, delta)
    plot_tree_node(tree, min_val - delta, max_val + delta, min_val - delta, max_val + delta, None, None, axis)
    plt.title('K-D Tree')
    plt.show()
    plt.close()


def read_data_from_file(filename):
    file = open(filename, "r")
    line = file.readline()
    points = []
    while line:
        line = line.split()
        x = int(line[0])
        y = int(line[1])
        points.append((x, y))
        line = file.readline()
    # points = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    return points


def process_kd_tree(points):
    x_variance = np.var([point[0] for point in points])
    y_variance = np.var([point[1] for point in points])

    axis = 0 if x_variance >= y_variance else 1

    tree = kdtree(points, axis)

    min_val = 0
    max_val = max(max([point[0] for point in points]), max([point[1] for point in points])) + 1
    delta = 0

    plot_tree(tree, min_val, max_val, delta, axis)


filename = "..\Project2-ml\MLHW2\datasets\points.txt"
points = read_data_from_file(filename)
process_kd_tree(points)




