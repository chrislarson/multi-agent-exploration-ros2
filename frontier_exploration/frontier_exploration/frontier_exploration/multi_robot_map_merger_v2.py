#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose2D
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import math
import os

from slam_toolbox.srv import SerializePoseGraph, DeserializePoseGraph

# This node is an attempt to introduce posegraph sharing to enable decentralized merging.


class MultiRobotMapMerger(Node):
    def __init__(self):
        super().__init__("multi_robot_map_merger")

        # Declare params
        self.declare_parameter("local_namespace", "robot1")
        self.declare_parameter("other_robot_namespaces", ["robot2"])
        self.declare_parameter("active_area_radius", 5.0)
        self.declare_parameter("pose_graph_directory", "/tmp")

        self.local_namespace = self.get_parameter("local_namespace").value
        self.other_robot_namespaces = self.get_parameter("other_robot_namespaces").value
        self.active_area_radius = self.get_parameter("active_area_radius").value
        self.pose_graph_directory = self.get_parameter("pose_graph_directory").value

        # Track poses as (x, y, yaw)
        self.local_pose = None
        self.other_poses = {ns: None for ns in self.other_robot_namespaces}

        # Track merging status
        self.have_merged_with_robot = {ns: False for ns in self.other_robot_namespaces}

        # For asynchronous merging states
        self.current_merge_robot = None
        self.current_step = None
        self.local_posegraph_file = ""
        self.other_posegraph_file = ""

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribe to local robot pose
        self.local_pose_sub_stamped = self.create_subscription(
            PoseStamped, f"/{self.local_namespace}/pose", self.pose_callback_factory(is_local=True), qos
        )
        self.local_pose_sub_cov = self.create_subscription(
            PoseWithCovarianceStamped, f"/{self.local_namespace}/pose", self.pose_callback_factory(is_local=True), qos
        )

        # Subscribe to other robots' poses
        self.other_pose_subs = []
        for ns in self.other_robot_namespaces:
            sub_stamped = self.create_subscription(
                PoseStamped, f"/{ns}/pose", self.pose_callback_factory(is_local=False, other_ns=ns), qos
            )
            self.other_pose_subs.append(sub_stamped)

            sub_cov = self.create_subscription(
                PoseWithCovarianceStamped, f"/{ns}/pose", self.pose_callback_factory(is_local=False, other_ns=ns), qos
            )
            self.other_pose_subs.append(sub_cov)

        # Create service clients
        self.local_save_cli = self.create_client(
            SerializePoseGraph, f"/{self.local_namespace}/slam_toolbox/serialize_map"
        )
        self.local_load_cli = self.create_client(
            DeserializePoseGraph, f"/{self.local_namespace}/slam_toolbox/deserialize_map"
        )

        self.other_save_clis = {}
        self.other_load_clis = {}
        for ns in self.other_robot_namespaces:
            save_cli = self.create_client(SerializePoseGraph, f"/{ns}/slam_toolbox/serialize_map")
            load_cli = self.create_client(DeserializePoseGraph, f"/{ns}/slam_toolbox/deserialize_map")
            self.other_save_clis[ns] = save_cli
            self.other_load_clis[ns] = load_cli

        self.timer = self.create_timer(1.0, self.check_for_proximity_and_merge)

        self.get_logger().info("Multi Robot Map Merger initialized.")

    def pose_callback_factory(self, is_local, other_ns=None):
        def callback(msg):
            if isinstance(msg, PoseStamped):
                x = msg.pose.position.x
                y = msg.pose.position.y
                orientation = msg.pose.orientation
            elif isinstance(msg, PoseWithCovarianceStamped):
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                orientation = msg.pose.pose.orientation
            else:
                self.get_logger().error("Unknown pose message type received.")
                return

            yaw = self.quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)

            if is_local:
                self.local_pose = (x, y, yaw)
                self.get_logger().debug(f"Local pose updated: ({x:.2f}, {y:.2f}, yaw={yaw:.2f}rad)")
            else:
                self.other_poses[other_ns] = (x, y, yaw)
                self.get_logger().debug(f"Pose from {other_ns} updated: ({x:.2f}, {y:.2f}, yaw={yaw:.2f}rad)")

        return callback

    def check_for_proximity_and_merge(self):
        if self.local_pose is None:
            return
        for ns in self.other_robot_namespaces:
            other_pose = self.other_poses[ns]
            if other_pose is None:
                continue
            dist = self.distance(self.local_pose[:2], other_pose[:2])
            self.get_logger().info(f"Distance to {ns}: {dist:.2f}m")

            in_area = dist < self.active_area_radius
            if in_area and not self.have_merged_with_robot[ns]:
                self.get_logger().info(
                    f"Robot {ns} is now within {self.active_area_radius}m of {self.local_namespace}. Initiating map merge."
                )
                self.start_map_merging(ns)
                self.have_merged_with_robot[ns] = True
            elif not in_area and self.have_merged_with_robot[ns]:
                self.get_logger().info(f"Robot {ns} left {self.local_namespace}'s active area. Resetting merge flag.")
                self.have_merged_with_robot[ns] = False

    def start_map_merging(self, other_ns):
        # State
        self.current_merge_robot = other_ns
        self.current_step = "SAVE_LOCAL"
        self.local_posegraph_file = os.path.join(self.pose_graph_directory, f"{self.local_namespace}_posegraph")
        self.other_posegraph_file = os.path.join(self.pose_graph_directory, f"{other_ns}_posegraph")

        # Ensure services are up:
        if not self.wait_for_service(self.local_save_cli, f"{self.local_namespace} serialize_map"):
            return
        if not self.wait_for_service(self.other_save_clis[other_ns], f"{other_ns} serialize_map"):
            return
        if not self.wait_for_service(self.local_load_cli, f"{self.local_namespace} deserialize_map"):
            return
        if not self.wait_for_service(self.other_load_clis[other_ns], f"{other_ns} deserialize_map"):
            return

        self.get_logger().info("Attempting to perform map merging...")
        self.get_logger().info("Saving local pose graph...")
        req = SerializePoseGraph.Request()
        req.filename = self.local_posegraph_file
        future = self.local_save_cli.call_async(req)
        future.add_done_callback(self.save_local_done_callback)

    def save_local_done_callback(self, future):
        self.get_logger().info("Local pose graph save callback triggered.")
        result = future.result()
        if result is None:
            self.get_logger().error("serialize_map call returned None result.")
            return
        if result.result != 0:  # 0 = RESULT_SUCCESS
            self.get_logger().error(f"Failed to save local pose graph, result code: {result.result}")
            return
        self.get_logger().info(f"Local pose graph saved successfully (result={result.result}).")

        # Move to next step: save other robot pose graph
        self.current_step = "SAVE_OTHER"
        self.get_logger().info(f"Saving {self.current_merge_robot}'s pose graph...")
        req = SerializePoseGraph.Request()
        req.filename = self.other_posegraph_file
        future2 = self.other_save_clis[self.current_merge_robot].call_async(req)
        future2.add_done_callback(self.save_other_done_callback)

    def save_other_done_callback(self, future):
        self.get_logger().info("Other robot pose graph save callback triggered.")
        result = future.result()
        if result is None:
            self.get_logger().error("serialize_map call for other robot returned None.")
            return
        if result.result != 0:
            self.get_logger().error(f"Failed to save {self.current_merge_robot}'s pose graph, code: {result.result}")
            return
        self.get_logger().info(f"{self.current_merge_robot}'s pose graph saved successfully (result={result.result}).")

        # Load other robot's pose graph locally
        self.current_step = "LOAD_OTHER_LOCALLY"
        self.get_logger().info(f"Loading {self.current_merge_robot}'s pose graph locally...")
        req = DeserializePoseGraph.Request()
        req.filename = self.other_posegraph_file

        # Use START_AT_GIVEN_POSE = 2 and set initial pose using local_pose
        req.match_type = 2
        init_pose = Pose2D()
        init_pose.x = self.local_pose[0]
        init_pose.y = self.local_pose[1]
        init_pose.theta = self.local_pose[2]  # Yaw
        req.initial_pose = init_pose

        future3 = self.local_load_cli.call_async(req)
        future3.add_done_callback(self.load_other_locally_done_callback)

    def load_other_locally_done_callback(self, future):
        self.get_logger().info("Load other robot pose graph locally callback triggered.")
        result = future.result()
        if result is None:
            self.get_logger().error("deserialize_map local call returned None.")
            return
        self.get_logger().info(
            f"Successfully loaded {self.current_merge_robot}'s pose graph locally (result={result})."
        )

        # Next: load local pose graph into the other robot
        self.current_step = "LOAD_LOCAL_IN_OTHER"
        self.get_logger().info(f"Loading local pose graph into {self.current_merge_robot}...")
        req = DeserializePoseGraph.Request()
        req.filename = self.local_posegraph_file
        # Also use START_AT_GIVEN_POSE = 2 and set initial pose based on other robot's pose or local?
        # Let's assume we want to align from the other's perspective as well, or just set same (x,y,yaw)
        # TODO: Possibly use the other robot's pose or local.
        req.match_type = 2
        init_pose = Pose2D()
        init_pose.x = self.other_poses[self.current_merge_robot][0]
        init_pose.y = self.other_poses[self.current_merge_robot][1]
        init_pose.theta = self.other_poses[self.current_merge_robot][2]
        req.initial_pose = init_pose

        future4 = self.other_load_clis[self.current_merge_robot].call_async(req)
        future4.add_done_callback(self.load_local_in_other_done_callback)

    def load_local_in_other_done_callback(self, future):
        self.get_logger().info("Load local pose graph in other robot callback triggered.")
        result = future.result()
        if result is None:
            self.get_logger().error("deserialize_map other call returned None.")
            return
        self.get_logger().info(f"Successfully loaded local pose graph into {self.current_merge_robot}. Merge complete.")

        # Reset states
        self.current_merge_robot = None
        self.current_step = None

    def wait_for_service(self, client, name):
        self.get_logger().info(f"Waiting for {name} service...")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f"{name} service not available.")
            return False
        self.get_logger().info(f"{name} service is available.")
        return True

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        # Convert quaternion to yaw
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotMapMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down multi_robot_map_merger node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
