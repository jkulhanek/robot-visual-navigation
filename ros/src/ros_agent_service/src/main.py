#!~/envs/visual_navigation/bin/python
import torch
from agent import create_agent
import ros
import rospy
from convert import image_to_numpy
from std_msgs.msg import String, Int32
from robot_agent_msgs.msg import ComputeStepRequest
import cv2
import argparse


class Server:
    def __init__(self, model_name):
        self.agent = create_agent(model_name)

    def compute_step(self, req):
        observation = image_to_numpy(req.observation)
        goal = image_to_numpy(req.goal)
        observation = cv2.resize(observation, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        goal = cv2.resize(goal, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
        action = self.agent.act((observation, goal,))
        return Int32(action)

    def reset_state(self, req):
        print("state has been cleared")
        self.agent.reset_state()


def create_send_response(server, response_publisher):
    def send_response(request):
        print("begin_compute_step: received request")
        action = server.compute_step(request)
        response_publisher.publish(action)
        pass

    return send_response


def server_main(name):
    print("ros_agent_service: starting")
    print("model: %s" % name)
    rospy.init_node('ros_agent_server')
    server = Server(name)
    response_publisher = rospy.Publisher('robot_visual_navigation/action', Int32, queue_size=10)
    s1 = rospy.Subscriber('robot_visual_navigation/begin_compute_step', ComputeStepRequest,
                          create_send_response(server, response_publisher))
    print("robot_visual_navigation/begin_compute_step: subscriber started")
    s2 = rospy.Subscriber('robot_visual_navigation/reset_state', String, server.reset_state)
    print("robot_visual_navigation/reset_state: subscriber started")
    rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ros_agent_service")
    parser.add_argument("--end", action="store_true")
    args = parser.parse_args()
    server_main("turtlebot-end" if args.end else "turtlebot")
