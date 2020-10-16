#!/usr/bin/env python
from sensor_msgs.msg import Image
from std_msgs.msg import Int32,String

from controller import Controller
from robot_agent_msgs.msg import ComputeStepRequest
from convert import convert_image
import rospy
import argparse
import numpy as np
import os
from math import pi, sqrt
import random
import sys

class Puller(object):
    def __init__(self, *args, **kwargs):
        super(Puller, self).__init__(*args, **kwargs)
        self._target = None
        self._action = None
        self._observation = None

        rospy.Subscriber("robot_visual_navigation/set_goal", Image, self._set_target, queue_size = 10)
        rospy.Subscriber("robot_visual_navigation/action", Int32, self._set_action, queue_size = 10)
        rospy.Subscriber("camera/rgb/image_raw", Image, self._set_observation, queue_size = 10)
        self.reset_state_publisher = rospy.Publisher("robot_visual_navigation/reset_state", String, queue_size = 10)


    def _set_target(self, msg):
        self._target = msg
        s = String()
        s.data = "empty"
        self.reset_state_publisher.publish(s)

    def _set_observation(self, msg):
        self._observation = msg

    def _set_action(self, msg):
        self._action = msg.data

    def wait_for_target(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self._target is None:
            rate.sleep()
        return self._target

    def wait_for_observation(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self._observation is None:
            rate.sleep()
        return self._observation  

    def wait_for_action(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self._action is None:
            rate.sleep()
        action = self._action
        self._action = None
        return action

def load_target(x, y, r):
    import h5py
    with h5py.File(os.path.expanduser("~/datasets/turtle_room/grid_compiled.hdf5"),"r") as f:
        imageIndex = f["grid"][x, y, r, random.randrange(np.sum(f["grid"][x, y, r, :] != -1))]
        targetImage = f["84x84/images"][imageIndex]
        msg = convert_image(targetImage)
        return msg

def angle_difference(a, b):
    return (a - b + pi) % (2 * pi) - pi

def dist(pos, target):
    x, y, r = pos
    gx, gy, gr = target
    return sqrt((gx - x) ** 2 + (gy - y) ** 2)

def is_goal(pos, target):
    x, y, r = pos
    gx, gy, gr = target
    return ((gx - x) ** 2 + (gy - y) ** 2) < 0.3 ** 2 and abs(angle_difference(r, gr)) < 0.52


parser = argparse.ArgumentParser("robot_evaluator")
parser.add_argument("x", type=int)
parser.add_argument("y", type=int)
parser.add_argument("r", type=int)
parser.add_argument("--end", action="store_true")
args = parser.parse_args()

rospy.init_node("robot_evaluator")
rospy.loginfo("starting")
ctrl = Controller()
puller = Puller()
compute_step_client = rospy.Publisher("robot_visual_navigation/begin_compute_step", ComputeStepRequest, queue_size = 10)
reset_goal_client = rospy.Publisher('robot_visual_navigation/reset_state', String, queue_size = 10)

rospy.sleep(rospy.Duration(nsecs=100 * 1000000))
ctrl.start()
angle = 0
rospy.loginfo("running")

#reset agent service
rmsg = String()
rmsg.data = "empty"
reset_goal_client.publish(rmsg)
rospy.sleep(rospy.Duration(nsecs=500 * 1000000))
rospy.loginfo("goal cleared")

rospy.loginfo("setting target: %s, %s, %s" % (args.x, args.y, args.r))
targetmsg = load_target(args.x, args.y, args.r)

number_of_actions = 0
actions_taken = []
positions = []
real = ctrl.position_and_rotation()
positions.append(real)
while not rospy.is_shutdown():
    req = ComputeStepRequest()
    req.goal = targetmsg
    rospy.loginfo("waiting for observation")
    req.observation = puller.wait_for_observation()
    req.sender = "empty"
    compute_step_client.publish(req)

    rospy.loginfo("waiting for action")
    action = puller.wait_for_action()
    rospy.loginfo("received action %s" % action)
    
    if action == 0:
        ctrl.move_by(0.2)
    elif action == 1:
        ctrl.move_by(-0.2)
    elif action == 2:
        ctrl.rotate_by(1.57)
    elif action == 3:
        ctrl.rotate_by(-1.57)
    elif action == 4:
        rospy.sleep(rospy.Duration(secs=1))
    rospy.sleep(rospy.Duration(secs=0,nsecs=800 * 1000000))
    number_of_actions += 1
    actions_taken.append(action)

    # If the goal is reached, return
    phy = (args.x - 2) * 0.2, (args.y - 3) * 0.2, (args.r * 1.57 + pi) % (2 * pi) - pi
    real = ctrl.position_and_rotation()
    positions.append(real)
    if args.end:
        if action == 4:
            if is_goal(real, phy):
                rospy.loginfo("goal correctly signaled after %s steps" % number_of_actions)
            else:
                rospy.loginfo("goal incorrectly signaled after %s steps" % number_of_actions)
            rospy.loginfo("position: %.2f %.2f %.2f" % real)
            rospy.loginfo("goal %.2f %.2f %.2f" % phy)
            # write results
            with open("results.txt", "a") as f:
                f.write("(%s %s %s) -> (%s %s %s) (%s %s %s) = %s (%s) %s [%s] [%s]\n" % (tuple(real) + (args.x, args.y, args.r) + phy + (is_goal(real, phy), dist(real, phy), len(actions_taken), ", ".join(map(str, actions_taken)), ", ".join(map(str, positions)))))
                f.flush()

            sys.exit()

    elif is_goal(real, phy):
        rospy.loginfo("goal reached after %s steps" % number_of_actions)
        rospy.loginfo("position: %.2f %.2f %.2f" % real)
        rospy.loginfo("goal %.2f %.2f %.2f" % phy)

        # write results
        with open("results.txt", "a") as f:
            f.write("(%s %s %s) = %s [%s]\n" % (args.x, args.y, args.r, len(actions_taken), ", ".join(map(str, actions_taken))))
            f.flush()

        sys.exit()