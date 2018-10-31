from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from sklearn.neighbors.kd_tree import KDTree

'''
class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))
'''
class Node:
    def __init__(self, location, left_child, right_child):
        self.location = location
        self.left_child = left_child
        self.right_child = right_child

def kdtree(points, axis=0):

    if len(points) == 0:
        return None

    # Sort point list and choose median as pivot element
    points.sort(key=itemgetter(axis))
    median = len(points) // 2  # choose median

    # Create node and construct subtrees
    return Node(points[median], kdtree(points[:median], 1 - axis), kdtree(points[median + 1:], 1 - axis))


file = open("..\Project2-ml\MLHW2\datasets\points.txt", "r")

line = file.readline()
points = []
x_list = []
y_list = []
while line:
    line = line.split()
    x = int(line[0])
    y = int(line[1])
    points.append((x, y))
    x_list.append(x)
    y_list.append(y)
    line = file.readline()

x_variance = np.var(x_list)
y_variance = np.var(y_list)
points = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]

kd_tree = kdtree(points)
n = 50        # number of points
min_val = 0   # minimal coordinate value
max_val = 10  # maximal coordinate value
delta = 0
# line width for visualization of K-D tree
line_width = [4., 3.5, 3., 2.5, 2., 1.5, 1., .5, 0.3]


def plot_tree(tree, min_x, max_x, min_y, max_y, prev_node, branch, depth=0):
    """ plot K-D tree
    :param tree      input tree to be plotted
    :param min_x
    :param max_x
    :param min_y
    :param max_y
    :param prev_node parent's node
    :param branch    True if left, False if right
    :param depth     tree's depth
    :return tree     node
    """

    cur_node = tree.location  # current tree's node
    left_branch = tree.left_child  # its left branch
    right_branch = tree.right_child  # its right branch

    # set line's width depending on tree's depth
    if depth > len(line_width) - 1:
        ln_width = line_width[len(line_width) - 1]
    else:
        ln_width = line_width[depth]

    k = len(cur_node)
    axis = depth % k

    # draw a vertical splitting line
    if axis == 0:

        if branch is not None and prev_node is not None:

            if branch:
                max_y = prev_node[1]
            else:
                min_y = prev_node[1]

        plt.plot([cur_node[0], cur_node[0]], [min_y, max_y], linestyle='-', color='red', linewidth=ln_width)

    # draw a horizontal splitting line
    elif axis == 1:

        if branch is not None and prev_node is not None:

            if branch:
                max_x = prev_node[0]
            else:
                min_x = prev_node[0]

        plt.plot([min_x, max_x], [cur_node[1], cur_node[1]], linestyle='-', color='blue', linewidth=ln_width)

    # draw the current node
    plt.plot(cur_node[0], cur_node[1], 'ko')

    # draw left and right branches of the current node
    if left_branch is not None:
        plot_tree(left_branch, min_x, max_x, min_y, max_y, cur_node, True, depth + 1)

    if right_branch is not None:
        plot_tree(right_branch, min_x, max_x, min_y, max_y, cur_node, False, depth + 1)


plt.figure("K-d Tree", figsize=(10., 10.))
plt.axis([min_val - delta, max_val + delta, min_val - delta, max_val + delta])

plt.grid(b=True, which='major', color='0.75', linestyle='--')
plt.xticks([i for i in range(min_val - delta, max_val + delta, 1)])
plt.yticks([i for i in range(min_val - delta, max_val + delta, 1)])

# draw the tree
plot_tree(kd_tree, min_val - delta, max_val + delta, min_val - delta, max_val + delta, None, None)

plt.title('K-D Tree')
plt.show()
plt.close()




