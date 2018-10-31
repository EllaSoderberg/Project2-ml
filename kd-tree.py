file = open("..\Project2-ml\MLHW2\datasets\points.txt", "r")

line = file.readline()
while line:
    print(line)
    line = file.readline()

