from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from common import Proxy
from visualize import load_observations, find_missing
from collector import PATH
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import os

def plot_maze(maze):
    minx = 0
    maxx = 0
    miny = 0
    maxy = 0
    for (x,y) in maze.keys():
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)

    shape = (maxx - minx + 1, maxy - miny + 1)
    width, height = shape

    fmaze = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(maze[(x + minx, y + miny)])

        if width == 1:
            row.append(0)
        fmaze.append(row)
    if height == 1:
        fmaze.append([0 for x in range(width)])

    print("shape: %s x %s" % shape)

    # plt.pcolormesh(fmaze)
    # plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    # plt.xticks([]) # remove the tick marks by setting to an empty list
    # plt.yticks([]) # remove the tick marks by setting to an empty list
    # plt.axes().invert_yaxis() #invert the y-axis so the first row of data is at the top
    # plt.show()
    mp = {
        0: "_",
        1: ".",
        2: "c",
        3: "o"
    }

    print("+" + "".join("-" for _ in range(len(fmaze[0]))) + "+")
    for r in fmaze:        
        print("|" + "".join([mp[x] for x in r]) + "|")
    print("+" + "".join("-" for _ in range(len(fmaze[0]))) + "+")

def visualize(navigator):
    oldCollect = navigator.collect
    
    def collect(*args, **kwargs):
        print("colll")
        plot_maze(navigator.maze)
        oldCollect(*args, **kwargs)
    navigator.collect = collect
    return navigator

    # class NavigatorProxy(Proxy):
    #     def __init__(self, *args, **kwargs):
    #         super(NavigatorProxy, self).__init__(*args, **kwargs)

    #     def __setattr__(self, name, value):
    #         if name == 'maze':
    #             plot_maze(value)
    #         return super(NavigatorProxy, self).__setattr__(name, value)

    #     def __getattribute__(self, name):
    #         if name == "collect":
    #             return collect
    #         return super(NavigatorProxy, self).__getattribute__(name)
        
    # return NavigatorProxy(navigator)


def visualize_points(navigator):
    oldCollect = navigator.collect
    pointsX = []
    pointsY = []
    pointsR = []

    plt.ion()
    plt.show()
    
    def collect(observation, *args, **kwargs):
        _, _, (px, py), r, (px2, py2, r2) = observation
        pointsX.append(px)
        pointsY.append(py)
        pointsR.append(r[2])
        print("collecting: [%s, %s]" % (px, py))

        #plt.scatter(*find_missing(pointsX, pointsY, pointsR), c='r')
        # plt.scatter(pointsX, pointsY)        
        # plt.draw()
        # plt.pause(0.001)

        oldCollect(observation, *args, **kwargs)
    navigator.collect = collect

    def _load_observations(path):
        (pointsX, pointsY, pointsR) = load_observations(PATH + "dataset/info.txt")
        pass

    navigator.load_observations = _load_observations
    return navigator

def visualize_remote(navigator):
    import rospy

    publisher = rospy.Publisher("/map_collector/points", Float32MultiArray, queue_size=10)

    oldCollect = navigator.collect
    pointsX = []
    pointsY = []
    pointsR = []

    plt.ion()
    plt.show()

    def vis():
        arr = Float32MultiArray()
        arr.data = pointsX + pointsY + pointsR
        arr.layout.dim.append(MultiArrayDimension())
        arr.layout.dim.append(MultiArrayDimension())
        arr.layout.dim[0].label = "dim"
        arr.layout.dim[0].size = 3
        arr.layout.dim[0].stride = 3 * len(pointsX)
        arr.layout.dim[1].label = "index"
        arr.layout.dim[1].size = len(pointsX)
        arr.layout.dim[1].stride = len(pointsX)
        publisher.publish(arr)

    navigator.visualize = vis
    
    def collect(observation, *args, **kwargs):
        _, _, (px, py), r, (px2, py2, r2) = observation
        pointsX.append(px)
        pointsY.append(py)
        pointsR.append(r[2])
        print("collecting: [%s, %s]" % (px, py))

        navigator.visualize()

        oldCollect(observation, *args, **kwargs)
    navigator.collect = collect

    def _load_observations(path):
        (xs, ys, rs) = load_observations(PATH + "dataset/info.txt")
        pointsX.extend(xs)
        pointsY.extend(ys)
        pointsR.extend(rs)
        pass

    navigator.load_observations = _load_observations
    return navigator
