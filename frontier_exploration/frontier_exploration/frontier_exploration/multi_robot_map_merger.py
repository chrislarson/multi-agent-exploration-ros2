import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


def merge_occupancy_grids(map1, map2):
    """Merge two occupancy grids into a single grid."""
    # Extract map data and dimensions
    data1 = np.array(map1.data).reshape(map1.info.height, map1.info.width)
    data2 = np.array(map2.data).reshape(map2.info.height, map2.info.width)

    # Determine new dimensions to fit both maps
    new_width = max(map1.info.width, map2.info.width)
    new_height = max(map1.info.height, map2.info.height)

    # Create new grids with unexplored cells (-1) to fit new dimensions
    padded_data1 = np.full((new_height, new_width), -1, dtype=np.int8)
    padded_data2 = np.full((new_height, new_width), -1, dtype=np.int8)

    # Calculate offsets based on origins
    offset_x1 = int((map1.info.origin.position.x - map2.info.origin.position.x) / map1.info.resolution)
    offset_y1 = int((map1.info.origin.position.y - map2.info.origin.position.y) / map1.info.resolution)

    # Place map1 in the padded grid
    padded_data1[
        max(0, offset_y1) : max(0, offset_y1) + map1.info.height,
        max(0, offset_x1) : max(0, offset_x1) + map1.info.width,
    ] = data1

    # Place map2 in the padded grid
    padded_data2[
        max(0, -offset_y1) : max(0, -offset_y1) + map2.info.height,
        max(0, -offset_x1) : max(0, -offset_x1) + map2.info.width,
    ] = data2

    # Merge the maps (average values for overlapping regions)
    merged_data = np.where(
        padded_data1 == -1, padded_data2, np.where(padded_data2 == -1, padded_data1, (padded_data1 + padded_data2) // 2)
    )
    merged_data = merged_data.flatten()

    # Create the merged map message
    merged_map = OccupancyGrid()
    merged_map.header = map1.header  # Use map1's header (adjust if needed)
    merged_map.info = map1.info  # Copy map1's metadata
    merged_map.info.width = new_width
    merged_map.info.height = new_height
    merged_map.data = merged_data.tolist()

    return merged_map


class MultiRobotMapMerger(Node):
    def __init__(self):
        super().__init__("multi_robot_map_merger")

        # Declare parameters
        self.declare_parameter("robot_namespaces", ["robot1", "robot2"])
        self.declare_parameter("proximity_threshold", 3.0)

        # Set default values
        self.robot_namespaces = self.get_parameter("robot_namespaces").get_parameter_value().string_array_value
        self.proximity_threshold = self.get_parameter("proximity_threshold").get_parameter_value().double_value

        # Initialize pose and map dictionaries
        self.robot_poses = {ns: None for ns in self.robot_namespaces}
        self.robot_maps = {ns: None for ns in self.robot_namespaces}

        # Subscribe to pose and map topics
        for namespace in self.robot_namespaces:

            # Subscribe to PoseStamped
            self.create_subscription(
                PoseStamped, f"{namespace}/pose", lambda msg, ns=namespace: self.pose_stamped_callback(msg, ns), 10
            )

            # Subscribe to PoseWithCovarianceStamped
            self.create_subscription(
                PoseWithCovarianceStamped,
                f"{namespace}/pose",
                lambda msg, ns=namespace: self.pose_with_covariance_callback(msg, ns),
                10,
            )

            self.create_subscription(
                OccupancyGrid, f"{namespace}/map", lambda msg, ns=namespace: self.map_callback(msg, ns), 10
            )

    def pose_stamped_callback(self, msg, namespace):
        self.get_logger().info(f"Got pose.")
        position = msg.pose.position
        self.robot_poses[namespace] = (position.x, position.y)
        self.check_proximity()

    def pose_with_covariance_callback(self, msg, namespace):
        self.get_logger().info(f"Got pose with covariance.")
        position = msg.pose.pose.position
        self.robot_poses[namespace] = (position.x, position.y)
        self.check_proximity()

    def map_callback(self, msg, namespace):
        self.robot_maps[namespace] = msg
        self.log_map_metadata(namespace, msg)

    def check_proximity(self):
        """Check proximity between robots and trigger map merging."""
        for ns1, pose1 in self.robot_poses.items():
            for ns2, pose2 in self.robot_poses.items():
                if ns1 != ns2 and pose1 and pose2:
                    distance = ((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) ** 0.5
                    self.get_logger().info(f"Robots {ns1} and {ns2} are ({distance:.2f}m) apart.")
                    if distance < self.proximity_threshold:
                        self.get_logger().info(
                            f"Robots {ns1} and {ns2} are within range ({distance:.2f}m). Merging maps."
                        )
                        self.merge_maps(ns1, ns2)

    # def merge_maps(self, ns1, ns2):
    #     map1 = self.robot_maps[ns1]
    #     map2 = self.robot_maps[ns2]

    #     if map1 and map2:
    #         # Validate maps
    #         valid1, message1 = self.validate_map(map1)
    #         valid2, message2 = self.validate_map(map2)
    #         if not valid1:
    #             self.get_logger().warn(f"Map from {ns1} is invalid: {message1}")
    #             return
    #         if not valid2:
    #             self.get_logger().warn(f"Map from {ns2} is invalid: {message2}")
    #             return

    #         # Merge valid maps
    #         merged_map = merge_occupancy_grids(map1, map2)

    #         # Configure QoS for compatibility
    #         qos_profile = QoSProfile(
    #             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
    #         )

    #         for namespace in self.robot_namespaces:
    #             self.get_logger().info(f"Publishing merged map to {namespace}.")
    #             publisher = self.create_publisher(OccupancyGrid, f"{namespace}/map", qos_profile)
    #             merged_map.header.frame_id = f"{namespace}/map"
    #             merged_map.header.stamp = self.get_clock().now().to_msg()
    #             publisher.publish(merged_map)

    def merge_maps(self, ns1, ns2):
        map1 = self.robot_maps[ns1]
        map2 = self.robot_maps[ns2]

        if map1 and map2:
            valid1, message1 = self.validate_map(map1)
            valid2, message2 = self.validate_map(map2)
            if not valid1:
                self.get_logger().warn(f"Map from {ns1} is invalid: {message1}")
                return
            if not valid2:
                self.get_logger().warn(f"Map from {ns2} is invalid: {message2}")
                return

            # Merge the maps
            merged_map = merge_occupancy_grids(map1, map2)

            # Configure QoS for compatibility
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
            )

            # Publish to a single merged map topic
            publisher = self.create_publisher(OccupancyGrid, "/merged_map", qos_profile)
            merged_map.header.frame_id = "map"
            merged_map.header.stamp = self.get_clock().now().to_msg()
            publisher.publish(merged_map)

            self.get_logger().info("Published merged map to /merged_map.")

    def validate_map(self, occupancy_grid):
        """Validate that the occupancy grid metadata matches the data size."""
        expected_size = occupancy_grid.info.height * occupancy_grid.info.width
        actual_size = len(occupancy_grid.data)
        if expected_size != actual_size:
            return False, f"Expected size {expected_size}, but got {actual_size}."
        return True, "Valid map."

    def log_map_metadata(self, namespace, occupancy_grid):
        self.get_logger().info(
            f"Map from {namespace}: width={occupancy_grid.info.width}, height={occupancy_grid.info.height}, "
            f"data size={len(occupancy_grid.data)}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = MultiRobotMapMerger()

    node.get_logger().info(
        f"Starting map merger with namespaces: {node.robot_namespaces} and proximity threshold: {node.proximity_threshold}."
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down MultiRobotMapMerger...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
