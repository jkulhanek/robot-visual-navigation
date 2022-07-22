## Using the trained model for the final navigation
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
8. a) Navigate to http://localhost:5000/ and upload the goal
8. b) or use `python src/goal_picker/src/setgoal.py --image {path to image}`


## Collecting the original dataset for training
Note that this tutorial may be incomplete. Let us know if you had to change something to get it working.
1. Activate local environment with all python dependencies installed (follow README in the root of this repository).
2. Copy this repository to turtlebot.
3. Run the following in both your local repository and the turtlebot's copy of the repository
   ```
   cd ros
   catkin_make
   source devel/setup.bash
   ```
4. Run the following command (in the background) that will start the service collecting the images and storing them to `/mnt/data/dataset`. You can change this path in `https://github.com/jkulhanek/robot-visual-navigation/blob/e2317b9b23f8c1f655770259f2e52dbf97db691d/ros/src/map_collector/src/collector.py#L7`
   ```
   roslaunch map_collector/launch/start.launch &
   ```

5. Prepare a list of coordinates for the robot to collect. Note that the robot will be collecting them only in the direction of its movement so we recommend to do a snake-like path through the grid to collect all images from all positions. Each position is specified by 
the `--by {x} {y}` argument. Run this command as many times as you want until you collect the entire dataset.
   ```
    rosrun map_collector controller.py \
        --by 1 2
        --by 2 2
        --by 3 2
        --goal 4 2
    ```

6. Copy the dataset from `/mnt/data/dataset` to your local computer, activate a python environment and run the following commands from the root of this repository to get the final `.hdf5` dataset:
    ```
    python scripts/build_grid_dataset.py {path to the dataset folder}
    python scripts/compile_grid_dataset.py {path to the dataset folder}.hdf5
    ```

    The final dataset should be located at `{path to the dataset folder}_compiled.hdf5`.
