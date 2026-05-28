#!/usr/bin/env bash
# build.sh -- build the TerraSeg ROS2 node inside a container with venv-backed Python.
#
# IMPORTANT: source this script, do not execute it.
#
#     source build.sh             # correct: env stays in your shell
#     ./build.sh                  # wrong: env is lost when the subshell exits
#                                 #   (ros2 command will be "not found" afterwards)
#
# The colcon-generated entry-point script's shebang otherwise points at the
# container's system Python, which cannot import `terraseg`. We patch the
# shebang to the venv's Python so the node sees both the TerraSeg library and
# the ROS2 message types.
#
# Re-source this script after every code or pyproject change.

set -eo pipefail

# Detect execution vs sourcing. BASH_SOURCE[0] equals $0 when executed; they
# differ when sourced. If we were executed, abort with a clear explanation so
# the env-loss footgun does not bite the next user.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: build.sh was executed, not sourced." >&2
    echo "  Run it with 'source build.sh' so that ROS2 + venv stay active in your shell." >&2
    echo "  Otherwise the ros2 command will not be found after the build finishes." >&2
    exit 1
fi

source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
source .venv/bin/activate
rm -rf build install log
colcon build --packages-select terraseg_ros2 --symlink-install
sed -i "1c#\!$PWD/.venv/bin/python3" install/terraseg_ros2/lib/terraseg_ros2/terraseg_node
source install/setup.bash
echo "TerraSeg ROS2 build done. ROS2 + venv + install/ are all active in this shell."
echo "Now: ros2 launch terraseg_ros2 terraseg.launch.py"
