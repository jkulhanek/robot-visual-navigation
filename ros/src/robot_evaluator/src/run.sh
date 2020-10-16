#!/bin/bash
set -e
LOC_FILE="locations.txt"
LINE=$1
LINE_TXT=$(echo "$a" | sed "${LINE}q;d" $LOC_FILE)
LINE_ARRAY=($LINE_TXT)
XREAL=${LINE_ARRAY[0]}
X=$(echo "(${LINE_ARRAY[0]}/0.2) + 2" | bc)
YREAL=${LINE_ARRAY[1]}
Y=$(echo "(${LINE_ARRAY[1]}/0.2) + 3" | bc)
RREAL=${LINE_ARRAY[2]}
R=$(echo "(((${LINE_ARRAY[2]}/1.57) % 4) + 4) % 4" | bc)
echo "starting from: $X $Y $R"
echo "reseting odometry roslaunch"
rostopic pub -1 /mobile_base/commands/reset_odometry std_msgs/Empty
echo "reseting state"
rostopic pub -1 /robot_visual_navigation/reset_state std_msgs/String "empty"

if [[ "$Y" -lt "3" ]]; then
    echo "navigating to negative y space"
    rosrun robot_controller controller.py --by "$XREAL" "0.0" --by "$XREAL" "$YREAL" --angle "$RREAL"
else
    echo "normal navigation"
    rosrun robot_controller controller.py --by "$XREAL" "$YREAL" --angle "$RREAL"
fi

echo "positioned on startup position"
echo "starting the experiment with id: ${LINE}"
rosrun robot_evaluator main.py "${LINE_ARRAY[3]}" "${LINE_ARRAY[4]}" "${LINE_ARRAY[5]}"
echo "finished - returning to roslaunch"