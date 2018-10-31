import numpy as np
from sklearn.neighbors.kd_tree import KDTree

class Node:
    def __init__(self, node , less = None, more = None):
        self.node = node
        self.less = less
        self.more = more

    def __repr__(self):
        return str(self.node) + str(self.less) + str(self.more)

file = open("..\Project2-ml\MLHW2\datasets\points.txt", "r")

line = file.readline()
points = []
while line:
    line = line.split()
    x = int(line[0])
    y = int(line[1])
    points.append([x, y])
    line = file.readline()

print(points)

n1 = Node([0, 1], points)
print(n1)
