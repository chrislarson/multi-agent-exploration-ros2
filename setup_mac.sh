mamba deactivate
mamba activate ros_env
source install/setup.zsh
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/Users/chris/miniforge3/envs/ros_env/share/turtlebot3_gazebo/models