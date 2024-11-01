import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables

        
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        
        
        # Prediction step

        
        # Update step

        
        #publish the estimated reading

        pass    

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
