from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """
    Launch the TerraSeg ROS2 node, loading defaults from ``config/terraseg.yaml``.

    Override individual parameters at launch time, e.g.::

        ros2 launch terraseg_ros2 terraseg.launch.py \\
            checkpoint_path:=/path/to/best.pth variant:=S

    Returns:
        LaunchDescription : Composable launch description with one ``terraseg_node`` action.
    """
    default_config = Path(get_package_share_directory("terraseg_ros2")) / "config" / "terraseg.yaml"

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=str(default_config),
        description="Path to the parameter YAML file.",
    )

    node = Node(
        package="terraseg_ros2",
        executable="terraseg_node",
        name="terraseg_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, node])
