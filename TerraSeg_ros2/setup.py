from setuptools import find_packages, setup

package_name = "terraseg_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/terraseg.launch.py"]),
        ("share/" + package_name + "/config", ["config/terraseg.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ted Lentsch",
    maintainer_email="t.lentsch@tudelft.nl",
    description="ROS2 node for online LiDAR ground segmentation with TerraSeg.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "terraseg_node = terraseg_ros2.terraseg_node:main",
        ],
    },
)
