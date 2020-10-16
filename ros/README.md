1. Copy ws to turtlebot's home directory
2. Run the following in both your local ws and the turtlebot's ws
   a) cd to ws
   b) $ catkin_make
   c) $ source devel/setup.bash
3. Copy from cluster /home/kulhajon/models/turtlebot/weights.pth and /home/kulhajon/models/turtlebot-end/weights.pth to your local /home/<username>/models directory
4. Connect to turtlebot with ssh port forwarding $ ssh turtlebot@<ip> -L 5000:localhost:5000
5. From local terminal source visual_navigation environment $ source ~/envs/visual_navigation/bin/activate
6. In turtlebot's ssh run $ roslaunch robot_controller start.launch
7a. Start agent (from ws folder) with either $ python src/ros_agent_service/src/main.py
7b. or $ python src/ros_agent_service/src/main.py --end
8a. Navigate to http://localhost:5000/ and upload the goal
8b. or use $ python src/goal_picker/src/setgoal.py --image <path to image>
