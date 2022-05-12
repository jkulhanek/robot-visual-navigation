1. From your computer, download and extract the trained models:
   ```
   curl -L https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/turtlebot-models.tar.gz | tar -xz -C ~/.cache/robot-visual-navigation/models
   ```
2. Activate local environment with all python dependencies installed (follow README in the root of this repository).
3. Copy this repository to turtlebot.
4. Run the following in both your local repository and the turtlebot's copy of the repository
   ```
   cd ros
   catkin_make
   source devel/setup.bash
   ```
5. Connect to turtlebot with ssh port forwarding `ssh {username}@{turtlebot ip} -L 5000:localhost:5000`
6. In turtlebot's ssh run `roslaunch robot_controller start.launch`
7. On your local computer start the agent (from `ros` folder):
   `python src/ros_agent_service/src/main.py`
8a. Navigate to http://localhost:5000/ and upload the goal
8b. or use `python src/goal_picker/src/setgoal.py --image {path to image}`
