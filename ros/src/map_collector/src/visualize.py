import os
import matplotlib.pyplot as plt
from math import pi
import math

def load_observations(path):
    if os.path.exists(path):
        pointsX = []
        pointsY = []
        pointsR = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                pointsX.append(float(line[4]))
                pointsY.append(float(line[5]))
                pointsR.append(float(line[6]))
        return (pointsX, pointsY, pointsR)
    else:
        print("File not found")
        return ([], [], [])

def angle_difference(a, b):
    return (a - b + pi) % (2 * pi) - pi

def filter_by_angle(xs, ys, rs, angle):
    mxs = []
    mys = []
    for (x,y,r) in zip(xs,ys, rs):
        angleDiff = abs(angle_difference(r, angle))
        if angleDiff <= 0.26: # 15 degrees
            mys.append(y)
            mxs.append(x)
    return (mxs, mys)


def required_points():
    xs = []
    ys = []

    stepSize = 0.2
    rangeX = (0, 3.6)
    rangeY = (0, 1.0)
    stepsX = int((rangeX[1] - rangeX[0]) / stepSize)
    stepsY = int((rangeY[1] - rangeY[0]) / stepSize)
    for j in range(stepsY):
        for i in range(stepsX):
            (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + j * stepSize)
            xs.append(x)
            ys.append(y)

    for j in range(5):
        for i in range(-2, 0):
            (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + j * stepSize)
            xs.append(x)
            ys.append(y)

    for i in range(4):
        (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + 5 * stepSize)
        xs.append(x)
        ys.append(y)

    for i in range(1,4):
        (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + 6 * stepSize)
        xs.append(x)
        ys.append(y)

    for i in range(10,18):
        (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + -1 * stepSize)
        xs.append(x)
        ys.append(y)

    for j in range(-3, -1):
        for i in range(12,17):
            (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + j * stepSize)
            xs.append(x)
            ys.append(y)


    return (xs,ys)


def find_missing(xs, ys, rs):
    rotations = [0, 3.14, 1.57, -1.57, 0.78, -0.78, 2.35, -2.35]
    expectedPoints = []
    missingPointsX = []
    missingPointsY = []
    for xd, yd in zip(*required_points()):        
        expectedPoints.append((xd,yd))
        
        isMissing = [True for _ in rotations]
        for (x,y,r) in zip(xs,ys, rs):
            dist = math.sqrt((x - xd) ** 2 + (y - yd) ** 2)
            if dist <= 0.1:
                for rdi, rd in enumerate(rotations):
                    angleDiff = abs(angle_difference(r, rd))
                    if angleDiff <= 0.26: # 15 degrees
                        isMissing[rdi] = False

        if any(isMissing):
            missingPointsX.append(xd)
            missingPointsY.append(yd)

    return (missingPointsX, missingPointsY)

    # Check for missing points

if __name__ == "__main__":
    from collector import PATH
    (x, y, r) = load_observations(PATH + "dataset/info.txt")
    plt.axis("equal")    
    plt.scatter(*find_missing(x, y, r), c='r')
    plt.scatter(x, y, c = 'b',edgecolors='none',s = 8)
    plt.show()
