import matplotlib
matplotlib.use("TkAgg")

import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray
from visualize import find_missing

class Server:
    def __init__(self, *args, **kwargs):
        rospy.Subscriber("/map_collector/points", Float32MultiArray, callback = self.read_points)
        plt.ion()
        plt.show()
        self.points = None

    def read_points(self, msg):
        points = list(msg.data)
        numPoints = int(len(points) / 3)
        self.points = points[:numPoints], points[numPoints:(numPoints+numPoints)], points[(numPoints+numPoints):]
        
    def start(self):
        while True:
            if self.points is None:
                continue
            (xs, ys, rs) = self.points
            plt.cla()
            plt.axis("equal")
            plt.scatter(*find_missing(xs, ys, rs), c='r')
            plt.scatter(xs, ys, c = 'b',edgecolors='none',s = 8)      
            plt.draw()
            plt.pause(1)

if __name__ == "__main__":
    node = rospy.init_node("visualize_client")
    server = Server()
    server.start()
