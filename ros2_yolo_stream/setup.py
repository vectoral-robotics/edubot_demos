import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ros2_yolo_stream"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stefan Siegler",
    maintainer_email="dev@siegler.one",
    description="Lightweight ROS 2 camera stream object detection demo using OpenCV DNN and YOLO ONNX.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_stream_node = ros2_yolo_stream.yolo_stream_node:main",
        ],
    },
)
