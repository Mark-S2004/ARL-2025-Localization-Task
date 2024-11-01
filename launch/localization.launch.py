from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import ExecuteProcess

package_name = "kalman_filter"


def generate_launch_description():
    return LaunchDescription([
        Node(
            package=f"{package_name}",
            executable="Noise",
        ),
        Node(
            package=f"{package_name}",
            executable="KalmanFilter",
        ),
        ExecuteProcess(cmd=[["chmod 0700 /run/user/1000/"]], shell=True),
        Node(
            package="plotjuggler",
            executable="plotjuggler",
        ),
    ])
