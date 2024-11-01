from rclpy.node import Node
import rclpy
from nav_msgs.msg import Odometry
import numpy as np
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
class Noise(Node):
    def __init__(self):
        super().__init__("Noise_odom")
        self.noise=self.create_publisher(Odometry,"/odom_noise",1)
        self.create_subscription(Odometry,'/odom',callback=self.Noise,qos_profile=1)
        self.nosied_Reading=Odometry()
    def Noise(self,msg:Odometry):
        groundTruthReading=msg
        #add gaussian noise to the ground truth reading
        self.nosied_Reading.pose.pose.position.x=groundTruthReading.pose.pose.position.x+np.random.normal(0,0.1)
        self.nosied_Reading.pose.pose.position.y=groundTruthReading.pose.pose.position.y+np.random.normal(0,0.1)
        #publish the noisy reading
        self.noise.publish(self.nosied_Reading)
        
def main(args=None):
    rclpy.init(args=args)
    node=Noise()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()