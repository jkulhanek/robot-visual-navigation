#!/usr/bin/env python
from sensor_msgs.msg import Image
from std_msgs.msg import Int32,String

from controller import Controller
from robot_agent_msgs.msg import ComputeStepRequest
import rospy

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

rospy.init_node("robot_controller")

rospy.loginfo("starting")
ctrl = Controller()
puller = Puller()
compute_step_client = rospy.Publisher("robot_visual_navigation/begin_compute_step", ComputeStepRequest, queue_size = 10)
ctrl.start()
angle = 0
rospy.loginfo("running")

while not rospy.is_shutdown():
    req = ComputeStepRequest()
    rospy.loginfo("waiting for target")
    req.goal = puller.wait_for_target()
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
        rospy.sleep(rospy.Duration(1.0))
    rospy.sleep(rospy.Duration(0.5))