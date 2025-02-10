# Distributed Frontier Exploration

Launch two robots with slam-toolbox:

```sh
ros2 launch frontier_exploration unique_multi_tb3_simulation_launch.py
```

Launch frontier exploration for both robots:

```sh
ros2 run frontier_exploration frontier_exploration --ros-args -p namespace:=robot1
```

```sh
ros2 run frontier_exploration frontier_exploration --ros-args -p namespace:=robot2
```

Launch map merging node:

```sh
ros2 run frontier_exploration multi_robot_map_merger --ros-args -p robot_namespaces:="[robot1,robot2]" -p proximity_threshold:=5.0
```