import rospy
import numpy as np
from controller import Controller
from wrappers import visualize_remote
from collector import make_file_dataset, PATH
from collections import defaultdict
from navigator import Navigator


if __name__ == "__main__":
    node = rospy.init_node("auto_collect")
    controller = Controller(node)
    controller.start()
    controller = visualize_remote(controller)
    controller = make_file_dataset(controller)
    controller.load_observations(PATH + "dataset/info.txt")
    controller.auto_collect()