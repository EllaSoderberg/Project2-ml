file = open("..\Project2-ml\MLHW2\datasets\points.txt", "r")

line = file.readline()
points = []
while line:
    line = line.split()
    x = int(line[0])
    y = int(line[1])
    points.append({x, y})
    line = file.readline()

print(points)
