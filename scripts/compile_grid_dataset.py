import os
import h5py
from math import pi
import math
import os
from scipy.spatial import cKDTree
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa


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


def construct_grid(points, max_points=10):
    pointGrid = dict()
    pointData = points[:, 0:2]
    tree = cKDTree(pointData)
    for x, y in zip(*required_points()):
        pts = tree.query_ball_point((x, y), r=0.1)
        pts.sort(key=lambda i: np.linalg.norm(pointData[i, :] - np.array((x, y,), dtype=np.float32)))
        currentPosition = []
        pointGrid[(round(x / 0.2) + 2, round(y / 0.2) + 3)] = currentPosition
        for angleIndex, r in enumerate([0, 1.57, 3.14, -1.57]):
            currentPosition.append([])
            for i in pts:
                _, _, rp = tuple(points[i, 0:3])
                angleDiff = abs(angle_difference(r, rp))
                if angleDiff <= 0.26:  # 15 degrees
                    if len(currentPosition[angleIndex]) < max_points:
                        currentPosition[angleIndex].append(i)
    return pointGrid


def grid_to_numpy(grid, max_points=10):
    xmax = 0
    ymax = 0
    for (x, y) in grid.keys():
        xmax = max(x, xmax)
        ymax = max(y, ymax)
    array = np.ndarray((xmax + 1, ymax + 1, 4, max_points), dtype=np.int32)
    array.fill(-1)
    for (x, y), ps in grid.items():
        for r, ptidxs in enumerate(ps):
            for i, index in enumerate(ptidxs):
                array[x, y, r, i] = index
    return array


def find_missing(xs, ys, rs):
    rotations = [0, 3.14, 1.57, -1.57]  # 0.78, -0.78, 2.35, -2.35]
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


def create_augmented_dataset(f, images, depths):
    # Define our augmentation pipeline.
    image_aug = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ])
    depth_aug = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    aimg = f.create_dataset("augmented_images", (images.shape[0], 100,) + images.shape[1:], 'u1')
    adep = f.create_dataset("augmented_depths", (depths.shape[0], 100,) + depths.shape[1:], 'u2')

    for i in range(depths.shape[0]):
        print("augmenting image %s" % (i + 1))
        image = images[i, ...]
        dep = depths[i, ...]
        for j in range(100):
            aimg[i, j, ...] = image_aug(image=image)
            adep[i, j, ...] = depth_aug(image=dep)


def create_image_collection(newf, oldf, usedImages, indexMap, grid, height, width):
    oldImages = oldf['images']
    oldDepths = oldf['depths']
    group: h5py.Group = newf.create_group('%sx%s' % (height, width))
    grid = {key: [[indexMap[x] for x in k] for k in v] for key, v in grid.items()}
    imagesDataset = group.create_dataset("images", (len(usedImages), height, width, 3), 'u1')
    depthDataset = group.create_dataset("depths", (len(usedImages), height, width,), 'u2')

    for i, oldIndex in enumerate(usedImages):
        print("processing image %s" % (i + 1))
        imagesDataset[i, ...] = Image.fromarray(oldImages[oldIndex, ...]).resize((width, height))
        depthDataset[i, ...] = Image.fromarray(oldDepths[oldIndex, ...]).resize((width, height))

    create_augmented_dataset(group, imagesDataset, depthDataset)
    return grid, indexMap


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Script to compile the original .hdf5 dataset and add augmented images")
    parser.add_argument("path", "Path to the original .hdf5 dataset")
    args = parser.parse_args()
    path = os.path.splitext(args.path)[0]

    if os.path.exists(path + "_compiled.hdf5"):
        print("Removing existing compiled dataset")
        os.remove(path + "_compiled.hdf5")
    with h5py.File(path + ".hdf5", 'r') as f:
        with h5py.File(path + "_compiled.hdf5", 'w') as fcomp:
            positions = f['positions']
            grid = construct_grid(positions)
            usedImages = list(set((y for _, j in grid.items() for x in j for y in x)))
            usedImages.sort()
            indexMap = {e: i for i, e in enumerate(usedImages)}
            newPositions = [positions[i] for i in usedImages]
            grid = create_image_collection(fcomp, f, usedImages, indexMap, grid, 84, 84)
            grid = grid_to_numpy(grid)
            fcomp.create_dataset("grid", data=grid)
            fcomp.create_dataset("positions", data=newPositions)
            fcomp.flush()
