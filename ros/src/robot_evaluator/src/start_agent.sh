#!/bin/bash
source ~/envs/visual_navigation/bin/activate
source ~/ws/devel/setup.sh
PYTHONPATH=$AGENT_PATH:$PYTHONPATH python ~/ws/src/ros_agent_service/src/main.py