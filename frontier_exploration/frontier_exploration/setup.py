from setuptools import find_packages, setup
import os
from glob import glob

package_name = "frontier_exploration"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.py"))),
        (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
        (os.path.join("share", package_name, "urdf"), glob(os.path.join("urd", "*.urdf"))),
        (os.path.join("share", package_name, "worlds"), glob(os.path.join("worlds", "*.model"))),
        (os.path.join("share", package_name, "params"), glob(os.path.join("params", "*.yaml"))),
        (os.path.join("share", package_name, "maps"), glob(os.path.join("maps", "*"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chris",
    maintainer_email="larson07@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frontier_exploration = frontier_exploration.frontier_exploration:main",
            "multi_robot_map_merger = frontier_exploration.multi_robot_map_merger:main",
        ],
    },
)
