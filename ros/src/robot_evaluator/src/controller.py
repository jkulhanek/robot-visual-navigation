import rospy
import tf
import numpy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point
from nav_msgs.msg import Odometry
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from convert import image_to_numpy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from math import floor, pi,isnan, copysign, sin, cos, asin, acos
import argparse
from collections import defaultdict
sign = lambda x: copysign(1, x)

R_SPEED = 0.2
M_SPEED = 0.08

def rotate_point(point, angle):
    return (cos(angle) * point[0] - sin(angle) * point[1],
        sin(angle) *  point[0] + cos(angle) * point[1])

class Controller:
    def __init__(self):
        self.imageTransform = tf.TransformListener()
        self.laserScanListener = rospy.Subscriber("/scan", LaserScan, callback = self._receiveLaserScan)
        self._velocityPublisher = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size = 10)
        self._odomSubscriber = rospy.Subscriber("/odom", Odometry, callback = self._receiveOdometry)
        self._laserScan = None
        self._pendingMovement = False
        self._image = None
        self.visualize = None
        self._startPose = None
        self._depth_image = None
        self._tangle = None
        self._positionTransform2 = None
        self._odom = None
        def collectImage(msg):
            self._image = msg
        def collectDepth(msg):
            self._depth_image = msg
        rospy.Subscriber("/camera/rgb/image_raw", Image, callback = collectImage)
        rospy.Subscriber("/camera/depth/image_raw", Image, callback = collectDepth)
        
    def _receiveLaserScan(self, msg):
        self._laserScan = msg

    def _receiveOdometry(self, msg):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        self._odom = ([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w])

    def _lookupOdometry(self):
        if not self._odom:
            raise tf.LookupException()
        return self._odom

    def start(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if not self._receiveSensorInput():
                continue
            # if self._pendingMovement:
            #     _, duration = rate.remaining()
            #     if self._baseController.wait_for_result(timeout = duration):
            #         self._pendingMovement = False
            # else:
            rate.sleep()
            return

    def collect(self, observation, posx, posy):
        print("Collecting")

    
    def _printPosition(self):
        print("position: %s, rotation: %s" % (self._position, self._rotation))

    def _receiveSensorInput(self):
        try:
            (trans, quat) = self._lookupOdometry()
            (trans2, quat2) = self.imageTransform.lookupTransform("/map", "/base_link",rospy.Time(0))
            if self._laserScan is None:
                return False
            self._rotation = euler_from_quaternion(quat)
            self._position = trans[:2]
            self._positionTransform2 = tuple(trans2[:2]) + (euler_from_quaternion(quat2)[2],)
            #self._position = rotate_point(trans, self._rotation[2])

            if not self._startPose:
                self._startPose = (self._position, self._rotation,)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False
        return True

    def move_to(self, target, rotate_final = True):
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            pass

        rotStart = self._rotation[2]
        pos = np.array(self._position, np.float32)
        target = np.array([target[0], target[1]], np.float32)
        direction = target - pos
        direction_angle = np.arctan2(direction[1], direction[0])
        distance = np.linalg.norm(direction)
        
        print("navigating to [%s, %s]" % (target[0], target[1]))
        self.rotate_to(direction_angle)
        self.move_by(distance)
        if rotate_final:
            self.rotate_to(rotStart)

    def move_by(self, distance):
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            pass

        start = np.array(self._position, np.float32)
        angle = self._rotation[2]
        ndir = np.array([cos(angle), sin(angle)], np.float32)
        target = start + distance * ndir
        while not rospy.is_shutdown():
            if not self._receiveSensorInput():
                continue

            diff = target - np.array(self._position, np.float32)
            dir_distance = np.dot(diff, ndir)
            if abs(dir_distance) <= 0.05:
                break
            
            msg = Twist()            
            speed = M_SPEED
            msg.linear.x = sign(dir_distance) * speed
            self._velocityPublisher.publish(msg)
            rate.sleep()
        self._printPosition()

    def rotate_to(self, angle):
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            pass
        # TODO: test this line
        #angle = (angle + self._startPose[1][2] + pi) % (2 * pi) - pi
        angle = (angle + pi) % (2 * pi) - pi
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown():
            if not self._receiveSensorInput():
                continue

            diff = self._rotation[2] - angle
            diff = (diff + pi) % (2 * pi) - pi
            if abs(diff) < 0.05:
                return
            
            msg = Twist()            
            speed = R_SPEED
            msg.angular.z = -sign(diff) * speed
            self._velocityPublisher.publish(msg)
            rate.sleep()

        self._printPosition()
        msg = Twist()
        self._velocityPublisher.publish(msg)

    def rotate_by(self, angle):
        if self._tangle is None:
            while not rospy.is_shutdown() and not self._receiveSensorInput():
                pass

            self._tangle = round(self._rotation[2] / 1.57) * 1.57

        self._tangle = (self._tangle + angle + pi) % (2 * pi) - pi
        self.rotate_to(self._tangle)        

    def is_occupied(self):
        assert self._laserScan is not None

        arange = self._laserScan.angle_max - self._laserScan.angle_min
        real_arange = min(arange, pi/6)
        pointCount = len(self._laserScan.ranges)
        stripPoints = int(floor((pointCount - pointCount * real_arange / arange) / 2))
        points = [0.2 if isnan(p) else p for p in self._laserScan.ranges[stripPoints:-stripPoints]]
        lbound = np.quantile(points, 0.1)
        return lbound < 0.5

    def observe(self):
        rospy.sleep(rospy.Duration(0.7))
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            pass
        assert self._image is not None
        assert self._depth_image is not None
        return [
            image_to_numpy(self._image),
            image_to_numpy(self._depth_image),
            self._position,
            self._rotation,
            self._positionTransform2
        ]

    def position(self):
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            continue

        return self._position

    def position_and_rotation(self):
        while not rospy.is_shutdown() and not self._receiveSensorInput():
            continue
        return tuple(self._position) + (self._rotation[2],)


if __name__ == "__main__":
    p = argparse.ArgumentParser("map_collector")
    p.add_argument("--angle", type = float, default=float("NaN"))
    p.add_argument("--goal", type = float, nargs = 2, metavar = ('x','y'), default=None)
    p.add_argument("--by", type = float, nargs = 2, action ="append", metavar = ('x','y'), default=None)
    p.add_argument("--jump", default=False, action="store_true")
    p.add_argument("--distance", type = float, default=float("NaN"))
    p = p.parse_args()
    rospy.init_node("map_collector")
    controller = Controller()
    controller.start()

    pos = controller.position()
    if p.by and len(p.by) > 0:
        for pp in p.by:
            controller.move_to(pp, rotate_final=False)
        

    if p.goal:
        controller.move_to(p.goal)
    if p.jump:
        controller.move_to(pos, rotate_final=False)

    if not isnan(p.angle):
        controller.rotate_to(p.angle)
    if not isnan(p.distance):
        controller.move_by(p.distance)