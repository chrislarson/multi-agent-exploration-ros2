#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist, PoseStamped
from tf2_ros import Buffer
from tf2_msgs.msg import TFMessage
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import numpy as np
import math
from collections import deque
import time


def angle_diff(a, b):
    """Compute the smallest absolute difference between two angles a and b, result in [0, pi]."""
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)


class FrontierExplorationNode(Node):
    def __init__(self):
        super().__init__("frontier_exploration_node")

        # Declare parameters
        self.declare_parameter("namespace", "robot1")
        self.declare_parameter("frontier_method", "wavefront")  # 'wavefront' or 'naive'
        self.namespace = self.get_parameter("namespace").value
        self.frontier_method = self.get_parameter("frontier_method").get_parameter_value().string_value

        # Topics based on robot namespace
        self.map_topic = f"/{self.namespace}/map"
        self.frontiers_topic = f"/{self.namespace}/frontiers"
        self.tf_topic = f"/{self.namespace}/tf"
        self.tf_static_topic = f"/{self.namespace}/tf_static"
        self.nav_action: str = f"/{self.namespace}/navigate_to_pose"
        self.costmap_topic = f"/{self.namespace}/global_costmap/costmap"
        self.target_frontier_topic = f"/{self.namespace}/target_frontier"
        self.active_area_topic = f"/{self.namespace}/active_area"

        # ROS 2 Interfaces
        self.map_subscriber = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 10)

        # Publishers for visualization
        self.frontiers_publisher = self.create_publisher(MarkerArray, self.frontiers_topic, 10)
        self.active_area_publisher = self.create_publisher(Marker, self.active_area_topic, 10)
        self.target_frontier_publisher = self.create_publisher(Marker, self.target_frontier_topic, 10)

        # TF buffer without a default TransformListener
        self.tf_buffer = Buffer()

        # TF subscriptions
        self.tf_subscription = self.create_subscription(TFMessage, self.tf_topic, self.tf_callback, 100)
        self.tf_static_subscription = self.create_subscription(
            TFMessage, self.tf_static_topic, self.tf_static_callback, 100
        )

        # Costmap subscription
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, self.costmap_topic, self.costmap_callback, 10)
        self.costmap_data = None

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action)

        # Data storage
        self.map_data = None
        self.frontiers_t_minus_1 = set()
        # self.frontiers = set()
        self.visited = set()
        self.labeled_frontiers = []

        # State variables for navigation logic
        self.first_target_selected = False
        self.reached_first_target = False
        self.is_navigating = False
        self.current_target = None
        self.nav_visited = set()  # visited targets
        self.failed_nav = set()  # failed targets due to no progress

        # For progress checking
        self.max_distance_progress = None
        self.prev_feedback_time = None

        self.retries = 0
        self.MAX_RETRIES = 3

        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.end_time = 0
        self.run_time = 0

        self.FIRST_NAV_THRESHOLD = 30.0
        self.REST_NAV_THRESHOLD = 8.0
        self.no_progress_time_threshold = self.FIRST_NAV_THRESHOLD  # seconds

        self.get_logger().info(f"Frontier Exploration Node initialized for namespace: {self.namespace}.")

    # -----------------------------------------
    # ---------- CALLBACK FUNCTIONS -----------
    # -----------------------------------------

    def tf_callback(self, msg):
        for transform in msg.transforms:
            self.tf_buffer.set_transform(transform, "default_authority")

    def tf_static_callback(self, msg):
        for transform in msg.transforms:
            self.tf_buffer.set_transform_static(transform, "default_authority")

    def costmap_callback(self, msg):
        self.costmap_data = msg

    def map_callback(self, msg):
        """Callback for processing map updates."""
        self.map_data = msg
        self.get_logger().debug("Received map update.")
        if not self.first_target_selected or self.reached_first_target:
            self.recompute_frontiers()

    def feedback_callback(self, feedback_msg):
        # Extract distance_remaining from feedback
        # Assume feedback_msg.feedback has a field distance_remaining or something similar
        # Nav2 NavigateToPose feedback has `estimated_time_remaining`, `distance_remaining`
        # According to nav2 documentation: feedback_msg.feedback.distance_remaining

        distance_remaining = (
            feedback_msg.feedback.distance_remaining if hasattr(feedback_msg.feedback, "distance_remaining") else None
        )

        if distance_remaining is None:
            # If no distance info, just return
            return

        current_time = self.get_clock().now().seconds_nanoseconds()[0]

        # First tracking.
        if self.max_distance_remaining is None:
            # First feedback
            self.max_distance_progress = 0
            self.max_distance_remaining = distance_remaining
            self.prev_feedback_time = current_time
            return

        # If max path distance increased, increase it.
        if distance_remaining > self.max_distance_remaining:
            self.max_distance_remaining = distance_remaining
            self.get_logger().debug(f"Distance remaining increased {distance_remaining}")

        else:
            # Check progress.
            distance_progress = 0
            if self.max_distance_remaining > 0:
                distance_progress = (self.max_distance_remaining - distance_remaining) / self.max_distance_remaining
            if distance_progress > self.max_distance_progress:  # Traversed at least 0.1m
                # progress made
                self.max_distance_progress = distance_progress
                self.prev_feedback_time = current_time
            else:
                # no progress *or* path is just beginning to plan
                time_since_last_progress = current_time - self.prev_feedback_time
                if time_since_last_progress > 1:
                    self.get_logger().debug(
                        f"No progress made (Max: {self.max_distance_progress}, Curr: {distance_progress}) in 1 second. Will invalid in {self.no_progress_time_threshold - time_since_last_progress} seconds."
                    )
                if time_since_last_progress > self.no_progress_time_threshold:
                    self.get_logger().warn(
                        f"No progress in {self.no_progress_time_threshold} seconds, invalidating this target and picking a new one."
                    )
                    # Add to failed_nav
                    if self.current_target is not None:
                        key = (round(self.current_target[0], 1), round(self.current_target[1], 1))
                        self.failed_nav.add(key)
                    # Instead of canceling, send origin as new goal, so robot backtracks and can potentially find new path.
                    origin_x = self.map_data.info.origin.position.x
                    origin_y = self.map_data.info.origin.position.y
                    self.send_navigation_goal(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                    # Cancel the goal
                    # self.cancel_current_goal()

    # ----------------------------------------
    # --------- FRONTIER DETECTION -----------
    # ----------------------------------------

    def recompute_frontiers(self):
        if self.retries >= self.MAX_RETRIES:
            return

        """Recompute frontiers based on the selected method."""
        self.get_logger().info("Recomputing frontiers.")
        if self.frontier_method == "wavefront":
            robot_pose = self.get_current_pose()
            if robot_pose is None:
                self.get_logger().error("Robot pose unavailable. Cannot detect frontiers.")
                return
            robot_x, robot_y, _ = robot_pose

            width = self.map_data.info.width
            height = self.map_data.info.height
            resolution = self.map_data.info.resolution
            origin_x = self.map_data.info.origin.position.x
            origin_y = self.map_data.info.origin.position.y

            robot_i = int((robot_y - origin_y) / resolution)
            robot_j = int((robot_x - origin_x) / resolution)

            if not (0 <= robot_i < height and 0 <= robot_j < width):
                self.get_logger().error("Robot position is out of map bounds")
                return

            robotStartingCell = (robot_i, robot_j)
            if not self.reached_first_target:
                sensor_range = 2.0
            else:
                sensor_range = 5.0
            sensor_range_in_cells = int(sensor_range / resolution)

            min_i = max(0, robot_i - sensor_range_in_cells)
            max_i = min(height, robot_i + sensor_range_in_cells + 1)
            min_j = max(0, robot_j - sensor_range_in_cells)
            max_j = min(width, robot_j + sensor_range_in_cells + 1)

            active_area = []
            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    di = i - robot_i
                    dj = j - robot_j
                    if di * di + dj * dj <= sensor_range_in_cells * sensor_range_in_cells:
                        active_area.append((i, j))

            self.get_logger().info(f"Active area defined with {len(active_area)} cells.")
            self.publish_active_area(robot_x, robot_y, sensor_range)

            F_t, labeled_frontiers, self.visited = self.detect_frontiers_wavefront(
                self.frontiers_t_minus_1, set(active_area), self.visited, robotStartingCell
            )

            self.frontiers_t_minus_1 = F_t
            self.labeled_frontiers = labeled_frontiers
            # self.frontier_points = list(F_t)

            self.publish_frontiers(F_t, labeled_frontiers)
        else:
            self.get_logger().error(f"Unknown frontier detection method: {self.frontier_method}")
            return

        # if (not self.is_navigating) and (self.frontiers_t_minus_1 or self.labeled_frontiers):
        #     self.select_and_navigate_to_target()
        if self.frontiers_t_minus_1 or self.labeled_frontiers:
            self.select_and_navigate_to_target()

    def get_current_pose(self):
        base_frame = "base_footprint"
        map_frame = "map"
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame=map_frame,
                source_frame=base_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            orientation = trans.transform.rotation
            yaw = self.quaternion_to_yaw(orientation)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().error(f"Error getting current pose: {e}")
            return None

    def detect_frontiers_wavefront(self, F_t_minus_1, At, visited, robotStartingCell):
        self.get_logger().info("Computing frontiers using expanding wavefront.")
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution

        map_data = np.array(self.map_data.data).reshape((height, width))

        if not F_t_minus_1:
            self.get_logger().debug("No previous frontiers supplied, using robot starting position as queue.")
            queue = deque([robotStartingCell])
        else:
            queue = deque()
            self.get_logger().debug(f"Previous frontiers length: {len(F_t_minus_1)}")
            for cell in F_t_minus_1:
                if cell in At:
                    queue.append(cell)
                    if cell in visited:
                        visited.discard(cell)

        F_t = set(F_t_minus_1)

        self.get_logger().debug(f"Initial queue length: {len(queue)}.")
        while queue:
            c = queue.popleft()
            if c in visited:
                continue
            visited.add(c)

            i, j = c
            if map_data[i][j] == 0:
                is_frontier = False
                neighbors = self.get_neighbors(c, width, height)
                for neighbor in neighbors:
                    ni, nj = neighbor
                    if neighbor in At and neighbor not in visited:
                        queue.append(neighbor)
                    if map_data[ni][nj] == -1:
                        is_frontier = True
                if is_frontier:
                    F_t.add(c)
                else:
                    if c in F_t_minus_1:
                        F_t.discard(c)

        self.get_logger().info("Computed frontiers using expanding wavefront.")
        # self.frontiers = F_t
        clusters = self.kosaraju_labeling(F_t, map_data, width, height)

        labeled_frontiers = []
        for idx, cluster in enumerate(clusters):
            cluster_id = idx
            labeled_frontiers.append({"id": cluster_id, "points": cluster})

        return F_t, labeled_frontiers, visited

    def compute_cluster_centroids(self, labeled_frontiers):
        """Compute (x, y, cluster_id) for each frontier cluster centroid."""
        centroids = []
        if self.map_data is None:
            return centroids
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        for cluster in labeled_frontiers:
            cluster_id = cluster["id"]
            cluster_points = cluster["points"]
            cluster_array = np.array(cluster_points)
            avg_i, avg_j = cluster_array.mean(axis=0)
            x = origin_x + (avg_j + 0.5) * resolution
            y = origin_y + (avg_i + 0.5) * resolution
            centroids.append((x, y, cluster_id))
        return centroids

    def kosaraju_labeling(self, frontier_set, map_data, map_width, map_height):
        visited_local = set()
        finish_stack = []
        self.get_logger().debug("Computing labels using kosaraju.")

        def dfs_first_pass(u):
            stack = [u]
            while stack:
                v = stack[-1]
                if v not in visited_local:
                    visited_local.add(v)
                    pushed_neighbor = False
                    for neighbor in self.get_neighbors(v, map_width, map_height):
                        if neighbor in frontier_set and neighbor not in visited_local:
                            stack.append(neighbor)
                            pushed_neighbor = True
                            break
                    if not pushed_neighbor:
                        finish_stack.append(stack.pop())
                else:
                    stack.pop()

        def dfs_second_pass(u, component):
            stack = [u]
            while stack:
                v = stack.pop()
                if v not in visited_local:
                    visited_local.add(v)
                    component.append(v)
                    for neighbor in self.get_neighbors(v, map_width, map_height):
                        if neighbor in frontier_set and neighbor not in visited_local:
                            stack.append(neighbor)

        for point in frontier_set:
            if point not in visited_local:
                dfs_first_pass(point)

        visited_local.clear()
        clusters = []
        while finish_stack:
            point = finish_stack.pop()
            if point not in visited_local:
                component = []
                dfs_second_pass(point, component)
                clusters.append(component)

        self.get_logger().debug("Computed labels using kosaraju.")
        return clusters

    # ------------------------------------
    # --------------UTLITIES -------------
    # ------------------------------------

    def quaternion_to_yaw(self, orientation):
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def yaw_to_quaternion(self, yaw):
        half_yaw = yaw / 2.0
        qx = 0.0
        qy = 0.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        return qx, qy, qz, qw

    def get_neighbors(self, point, width, height):
        i, j = point
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    neighbors.append((ni, nj))
        return neighbors

    # ------------------------------------
    # -------VISUALIZATION PUBLISHERS ----
    # ------------------------------------

    def publish_frontiers(self, frontier_points, labeled_frontiers):
        marker_array = MarkerArray()

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 0
        # Publish all frontier points
        for point in frontier_points:
            i, j = point
            x = self.map_data.info.origin.position.x + (j + 0.5) * self.map_data.info.resolution
            y = self.map_data.info.origin.position.y + (i + 0.5) * self.map_data.info.resolution

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontier_points"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Light blue frontiers
            marker.color.r = 0.6
            marker.color.g = 0.8
            marker.color.b = 0.9
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1

        # Publish cluster centroids
        centroids = self.compute_cluster_centroids(labeled_frontiers)
        for x, y, cluster_id in centroids:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontier_clusters"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            # Blue frontier centroids
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1

        self.frontiers_publisher.publish(marker_array)

    def publish_active_area(self, robot_x, robot_y, sensor_range):
        """Publish the active area as a circle marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "active_area"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        num_points = 36
        for k in range(num_points + 1):
            theta = 2.0 * math.pi * (k / float(num_points))
            x = robot_x + sensor_range * math.cos(theta)
            y = robot_y + sensor_range * math.sin(theta)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            marker.points.append(point)

        self.active_area_publisher.publish(marker)

    def publish_target_position(self, x, y, qx, qy, qz, qw):
        """Publish the target frontier as a marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_frontier"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = float(qx)
        marker.pose.orientation.y = float(qy)
        marker.pose.orientation.z = float(qz)
        marker.pose.orientation.w = float(qw)
        marker.scale.x = 0.6
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        # Make this arrow yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.target_frontier_publisher.publish(marker)

    # -----------------------------------------
    # ---------- NAVIGATION LOGIC -------------
    # -----------------------------------------

    def select_and_navigate_to_target(self):
        robot_pose = self.get_current_pose()
        if robot_pose is None:
            self.get_logger().error("Cannot select target: Robot pose not available.")
            return
        robot_x, robot_y, robot_yaw = robot_pose

        rotation_penalty_factor = 2.0  # Adjust as needed
        # If you want different penalty factor after first nav success, you can make this dynamic

        if not self.first_target_selected:
            # First target: farthest frontier point (no rotation penalty)
            if not self.frontiers_t_minus_1:
                self.get_logger().warn("No frontier points available for the first target.")
                return
            candidates = []
            for i, j in self.frontiers_t_minus_1:
                x = self.map_data.info.origin.position.x + (j + 0.5) * self.map_data.info.resolution
                y = self.map_data.info.origin.position.y + (i + 0.5) * self.map_data.info.resolution
                dist = math.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
                candidates.append((dist, x, y))

            # Sort by distance descending, no penalty
            candidates.sort(key=lambda c: c[0], reverse=True)

            for dist, x, y in candidates:
                if self.is_goal_valid(x, y):
                    self.current_target = (x, y)
                    self.first_target_selected = True
                    self.get_logger().info(f"First target: farthest frontier point at ({x:.2f}, {y:.2f})")
                    heading = math.atan2(y - robot_y, x - robot_x)
                    qx, qy, qz, qw = self.yaw_to_quaternion(heading)
                    self.send_navigation_goal(x, y, qx, qy, qz, qw)
                    return

            self.get_logger().info("No valid frontier point targets found.")
        else:
            # Subsequent targets: nearest centroid WITH rotation penalty
            if not self.labeled_frontiers:
                self.get_logger().warn("No labeled frontiers available for subsequent targets.")
                return

            centroids = self.compute_cluster_centroids(self.labeled_frontiers)
            if not centroids:
                self.get_logger().warn("No centroids computed; cannot select subsequent target.")
                return

            candidates = []
            for cx, cy, cid in centroids:
                dist = math.sqrt((cx - robot_x) ** 2 + (cy - robot_y) ** 2)
                # Compute heading difference
                heading = math.atan2(cy - robot_y, cx - robot_x)
                heading_diff = angle_diff(heading, robot_yaw)
                # Apply rotation penalty
                effective_distance = dist + rotation_penalty_factor * heading_diff
                candidates.append((effective_distance, cx, cy, cid))

            # Sort by effective_distance (smaller is better)
            candidates.sort(key=lambda c: c[0])

            for eff_dist, x, y, cid in candidates:
                if self.is_goal_valid(x, y):
                    self.current_target = (x, y)
                    self.get_logger().info(
                        f"Subsequent target: centroid at ({x:.2f}, {y:.2f}) with effective distance {eff_dist:.2f}"
                    )
                    heading = math.atan2(y - robot_y, x - robot_x)
                    qx, qy, qz, qw = self.yaw_to_quaternion(heading)
                    self.send_navigation_goal(x, y, qx, qy, qz, qw)
                    return

            self.get_logger().info("No valid centroid targets found for subsequent navigation.")
            self.retries = self.retries + 1

            if self.retries >= self.MAX_RETRIES:
                self.end_time = self.get_clock().now().seconds_nanoseconds()[0]
                self.run_time = self.end_time - self.start_time
                self.get_logger().info(
                    f"Frontier exploration using {self.frontier_method} COMPLETE in ${self.run_time} seconds. Returning to origin."
                )
                origin_x = self.map_data.info.origin.position.x
                origin_y = self.map_data.info.origin.position.y
                self.send_navigation_goal(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def is_goal_valid(self, x, y):
        """Check if a goal is valid:
        1. Not previously visited
        2. Not in a high cost area (cost > 90)
        3. Not previously failed due to no progress
        """
        # Round coordinates for consistent matching
        key = (round(x, 1), round(y, 1))
        if key in self.nav_visited:
            self.get_logger().debug(f"Invalid: already visited ({key}).")
            return False
        if key in self.failed_nav:
            self.get_logger().debug(f"Invalid: previously failed nav ({key}).")
            return False

        # Check costmap if available
        if self.costmap_data is not None:
            costmap = self.costmap_data
            width = costmap.info.width
            height = costmap.info.height
            resolution = costmap.info.resolution
            origin_x = costmap.info.origin.position.x
            origin_y = costmap.info.origin.position.y

            map_x = int((x - origin_x) / resolution)
            map_y = int((y - origin_y) / resolution)

            if not (0 <= map_x < width and 0 <= map_y < height):
                self.get_logger().debug("Invalid: goal out of costmap bounds.")
                return False

            cost = costmap.data[map_y * width + map_x]
            if cost > 90:
                self.get_logger().debug(f"Invalid: high cost ({cost}).")
                return False

        return True

    def send_navigation_goal(self, x, y, qx, qy, qz, qw):
        self.publish_target_position(x, y, qx, qy, qz, qw)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.is_navigating = True
        # Reset progress tracking
        self.max_distance_progress = None
        self.max_distance_remaining = None
        self.prev_feedback_time = None

        self.get_logger().info(f"Sending navigation goal to ({x:.2f}, {y:.2f})")
        self.nav_client.wait_for_server()
        # Provide feedback_callback here
        send_goal_future = self.nav_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    # def cancel_current_goal(self):
    #     if self.is_navigating:
    #         # Cancel the goal
    #         # We must first get the goal handle from the future.
    #         # However, we did not store the goal handle from the result of send_goal_async().
    #         # We can store it in goal_response_callback.
    #         self.get_logger().warn("Cancelling current goal due to no progress.")
    #         # We'll store the goal_handle in self.goal_handle when response callback is called
    #         if hasattr(self, "goal_handle") and self.goal_handle is not None:
    #             cancel_future = self.goal_handle.cancel_goal_async()
    #             cancel_future.add_done_callback(self.cancel_done_callback)
    #         else:
    #             self.get_logger().warn("No goal_handle found to cancel.")
    #     else:
    #         self.get_logger().warn("No navigation in progress to cancel.")

    # def cancel_done_callback(self, future):
    #     cancel_result = future.result()
    #     if cancel_result is not None:
    #         self.get_logger().info("Goal canceled successfully. Recomputing frontiers.")
    #     else:
    #         self.get_logger().warn("Failed to cancel the goal.")
    #     # self.is_navigating = False
    #     # self.recompute_frontiers()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal was rejected by the action server.")
            self.is_navigating = False
            return

        self.goal_handle = goal_handle  # store goal_handle for cancellation if needed
        self.get_logger().info("Goal accepted by the action server.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result()
        self.is_navigating = False
        self.goal_handle = None

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Successfully reached the frontier!")
            # Mark current target as visited
            if self.reached_first_target == False:
                self.reached_first_target = True
            if self.no_progress_time_threshold == self.FIRST_NAV_THRESHOLD:
                # After getting to first frontier, change threshold to 10 seconds
                self.no_progress_time_threshold = self.REST_NAV_THRESHOLD
            if self.current_target is not None:
                key = (round(self.current_target[0], 1), round(self.current_target[1], 1))
                self.nav_visited.add(key)
            self.recompute_frontiers()

        elif result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Goal aborted by the action server.")
            if self.current_target is not None:
                key = (round(self.current_target[0], 1), round(self.current_target[1], 1))

        elif result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn("Goal canceled.")
            if self.current_target is not None:
                key = (round(self.current_target[0], 1), round(self.current_target[1], 1))
                self.failed_nav.add(key)
            self.recompute_frontiers()
        else:
            self.get_logger().warn(f"Navigation failed with status {result.status}.")
            if self.current_target is not None:
                key = (round(self.current_target[0], 1), round(self.current_target[1], 1))
                self.failed_nav.add(key)
            self.recompute_frontiers()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
