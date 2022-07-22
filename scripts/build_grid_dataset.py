import h5py
from PIL import Image
from math import pi
import numpy as np
import math
import os


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
    for (x, y, r) in zip(xs, ys, rs):
        angleDiff = abs(angle_difference(r, angle))
        if angleDiff <= 0.26:  # 15 degrees
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
    for i in range(1, 4):
        (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + 6 * stepSize)
        xs.append(x)
        ys.append(y)
    for i in range(10, 18):
        (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + -1 * stepSize)
        xs.append(x)
        ys.append(y)
    for j in range(-3, -1):
        for i in range(12, 17):
            (x, y) = (rangeX[0] + i * stepSize, rangeY[0] + j * stepSize)
            xs.append(x)
            ys.append(y)
    return (xs, ys)


def find_missing(xs, ys, rs):
    rotations = [0, 3.14, 1.57, -1.57, 0.78, -0.78, 2.35, -2.35]
    expectedPoints = []
    missingPointsX = []
    missingPointsY = []
    for xd, yd in zip(*required_points()):
        expectedPoints.append((xd, yd))
        isMissing = [True for _ in rotations]
        for (x, y, r) in zip(xs, ys, rs):
            dist = math.sqrt((x - xd) ** 2 + (y - yd) ** 2)
            if dist <= 0.1:
                for rdi, rd in enumerate(rotations):
                    angleDiff = abs(angle_difference(r, rd))
                    if angleDiff <= 0.26:  # 15 degrees
                        isMissing[rdi] = False
        if any(isMissing):
            missingPointsX.append(xd)
            missingPointsY.append(yd)
    return (missingPointsX, missingPointsY)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Builds the .hdf5 dataset from a collection of collected images. Use 'compile_grid_dataset.py' to get the final compiled dataset.")
    parser.add_argument("path", help="Path to the folder containing info.txt file")
    parser.add_argument('--dataset-name', default="grid")
    args = parser.parse_args()
    path = args.path
    assert os.path.exists(os.path.join(path, "info.txt"))
    assert os.path.exists(os.path.join(path, "depths"))
    assert os.path.exists(os.path.join(path, "images"))

    header = []
    missing_files = []
    with open(os.path.join(path, "info.txt"), 'r') as fheader:
        for line in fheader:
            line = line.strip().split(" ")
            index = int(line[0])
            exists = os.path.exists(os.path.join(path, 'depths/%s.png' % index)) and os.path.exists(os.path.join(path, 'images/%s.png' % index))
            try:
                _ = Image.open(os.path.join(path, 'depths/%s.png' % index))
                _ = Image.open(os.path.join(path, 'images/%s.png' % index))
            except Exception:
                exists = False
            if not exists:
                print("Missing file %s" % index)
                missing_files.append(index)
            else:
                header.append((index, float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9])))

    nimages = len(header)
    if os.path.exists(os.path.join(path, "grid.hdf5")):
        os.remove(os.path.join(path, f"{args.dataset_name}.hdf5"))
    with h5py.File(os.path.join(path, f"{args.dataset_name}.hdf5"), 'w') as f:
        depths = f.create_dataset("depths", (nimages, 480, 640,), dtype='u2')
        images = f.create_dataset("images", (nimages, 480, 640, 3,), dtype='u1')
        positions = f.create_dataset("positions", (nimages, 6,), dtype='f4')
        for i, row in enumerate(header):
            index = row[0]
            depths[i, ...] = np.array(Image.open(os.path.join(path, 'depths/%s.png' % index))).astype(np.uint16)
            images[i, ...] = np.array(Image.open(os.path.join(path, 'images/%s.png' % index)))
            positions[i, ...] = row[1:7]
        f.flush()
