import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from numpy.linalg import inv
from numpy.typing import NDArray
from rclpy.node import Node
from sympy import Matrix, pretty


class KalmanFilter(Node):
    def __init__(self):
        super().__init__("kalman_filter_node")
        # Initialize kalman variables
        self.state_vars_prev_estimated = np.array([
            [0.0],
            [0.0],
        ])  # state variables = [[x], [y]]
        self.P_prev_estimated = np.array([
            [0.1, 0.0],
            [0.0, 0.1],
        ])  # Estimated Uncertainity of previous step
        self.u_prev = np.array([[0.0], [0.0]])  # input control = [[x_dot], [y_dot]]
        self.stamp_prev = 0  # timestamp of previous spin
        self.F = np.array([[1, 0], [0, 1]])  # prediction matrix
        self.Q = np.array([[0, 0], [0, 0]])  # Process noise/covariance
        self.H = np.array([[1, 0], [0, 1]])

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(
            Odometry, "/odom_noise", self.odom_callback, 1
        )
        self.create_subscription(Twist, "/cmd_vel", self.set_control, 1)

        # publish the estimated reading
        self.estimated_pub = self.create_publisher(Odometry, "/odom_estimated", 1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        position = msg.pose.pose.position
        y = np.array([
            [position.x],
            [position.y],
        ])  # measurement values [[measured_x], [measured_y]]
        covariance = msg.pose.covariance
        R = np.array([[covariance[0], 0], [0, covariance[7]]])

        # Prediction step
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-09
        state_vars_pred = self.predict_current_state_vars(timestamp)
        P_pred = self.F.dot(self.P_prev_estimated).dot(self.F.T) + self.Q

        # Correction step
        kalman_gain = P_pred.dot(self.H.T).dot(
            inv(self.H.dot(P_pred).dot(self.H.T) + R)
        )
        state_vars_corrected = state_vars_pred + kalman_gain.dot(
            y - self.H.dot(state_vars_pred)
        )
        P_corrected = (1 - kalman_gain.dot(self.H)).dot(P_pred)

        # publish the estimated reading
        pub_msg = msg
        pub_msg.pose.pose.position.x = state_vars_corrected[0][0]
        pub_msg.pose.pose.position.y = state_vars_corrected[1][0]
        self.estimated_pub.publish(pub_msg)

        # Debugging
        self.get_logger().info(
            f"""
At {timestamp} seconds:
{pretty(Matrix(self.state_vars_prev_estimated))} = state_vars_prev_estimated
{pretty(Matrix(self.u_prev))} = u_prev
{pretty(Matrix(state_vars_pred))} = state_vars_pred

{pretty(Matrix(self.F))} = F
{pretty(Matrix(self.P_prev_estimated))} = P_prev_estimated
{pretty(Matrix(self.Q))} = Q
{pretty(Matrix(P_pred))} = P_pred

{pretty(Matrix(self.H))} = H
{pretty(Matrix(R))} = R
{pretty(Matrix(kalman_gain))} = kalman_gain

{pretty(Matrix(y))} = y
{pretty(Matrix(state_vars_corrected))} = state_vars_corrected
{pretty(Matrix(P_corrected))} = P_corrected
------------------------------------------------------------------------------------------------
            """,
            throttle_duration_sec=5,
        )

        self.state_vars_prev_estimated = state_vars_corrected
        self.P_prev_estimated = P_corrected

    def set_control(self, msg: Twist):
        self.u_prev[0][0] = msg.linear.x
        self.u_prev[1][0] = msg.linear.y

    def predict_current_state_vars(self, timestamp: float) -> NDArray[np.float64]:
        """Predict the current position from previous estimated position and control input (linear velocity)

        Args:
            timestamp (float): current timestamp
            estimated_state_vars_prev (NDArray[np.float64]): previous corrected state variables
            u_prev (NDArray[np.float64]): input control from our control system

        Returns:
            NDArray[np.float64]: predicted state variables [[x_k], [y_k]]
        """
        x_prev, y_prev = self.state_vars_prev_estimated.flatten()
        x_dot_prev, y_dot_prev = self.u_prev.flatten()
        delta_t = timestamp - self.stamp_prev
        self.stamp_prev = timestamp

        return np.array([
            [x_prev + delta_t * x_dot_prev],
            [y_prev + delta_t * y_dot_prev],
        ])


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
