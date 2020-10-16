import rospy
import numpy as np
from controller import Controller
from wrappers import visualize
from collector import make_file_dataset
from collections import defaultdict
from navigator import Navigator


def main():
    node = rospy.init_node("map_collector")
    controller = Controller(node)
    controller.start()
    navigator = Navigator(controller)
    navigator = make_file_dataset(navigator)
    navigator = visualize(navigator)
    navigator.explore()

if __name__ == "__main__":
    main()