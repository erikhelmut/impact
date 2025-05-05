import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import WrenchStamped

import numpy as np
from resense_hex21.resense import sensor


class ResenseNode(Node):

    def __init__(self, serial_ports, frequencyMode):
        """
        This node is responsible for reading data from the Resense Hex 21 sensors and publishing it as WrenchStamped messages.

        :param serial_ports: serial ports to which the sensors are connected
        :param frequencyMode: frequence mode of the sensor
        """
        
        super().__init__("resense_node")

        # create dictionary for frequency modes
        self.frequencyModes = {0:1000, 1:500, 2:100}

        self.sensors = []
        # making sure that serial_port is readable before initializing sensor
        for serial_port in serial_ports:
            print("\nMaking sure that {} is readable...".format(serial_port))
            #os.system("sudo chmod a+rw {}".format(serial_port))
            
            print("Connecting to sensor electronics at {}...".format(serial_port))
            self.sensors.append(sensor.HEXSensor(serial_port))
            self.sensors[-1].connect()

        # declare parameters for QoS settings
        self.declare_parameter("rs_hex21.qos.reliability", "reliable")
        self.declare_parameter("rs_hex21.qos.history", "keep_last")
        self.declare_parameter("rs_hex21.qos.depth", 10)

        # get QoS profile
        qos_profile = self.get_qos_profile("rs_hex21.qos")

        # create publisher for all connected sensors
        self.pub_ = []
        for i in range(len(serial_ports)):
            self.pub_.append(self.create_publisher(WrenchStamped, "resense_{}".format(i), qos_profile))

        timer_period = 1 / self.frequencyModes[frequencyMode]
        self.timer = self.create_timer(timer_period, self.get_current_wrench)

    
    def get_qos_profile(self, base_param_name):
        """
        Helper function to retrieve and validate QoS settings.
        
        :param base_param_name: base name of the QoS parameters
        :return: QoSProfile object
        """
        
        # get the parameter values
        reliability_param = self.get_parameter(f"{base_param_name}.reliability").value
        history_param = self.get_parameter(f"{base_param_name}.history").value
        depth_param = self.get_parameter(f"{base_param_name}.depth").value

        # normalize to lowercase to avoid mismatches
        reliability_param = str(reliability_param).lower()
        history_param = str(history_param).lower()

        self.get_logger().info(f"QoS settings: reliability={reliability_param}, history={history_param}, depth={depth_param}")

        # convert to QoS enums with fallback
        if reliability_param == "best_effort":
            reliability = QoSReliabilityPolicy.BEST_EFFORT
        elif reliability_param == "reliable":
            reliability = QoSReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(f"Unknown reliability: {reliability_param}, defaulting to RELIABLE")
            reliability = QoSReliabilityPolicy.RELIABLE

        if history_param == "keep_last":
            history = QoSHistoryPolicy.KEEP_LAST
        elif history_param == "keep_all":
            history = QoSHistoryPolicy.KEEP_ALL
        else:
            self.get_logger().warn(f"Unknown history: {history_param}, defaulting to KEEP_LAST")
            history = QoSHistoryPolicy.KEEP_LAST

        # depth should be an int, just check type or cast
        try:
            depth = int(depth_param)
        except (ValueError, TypeError):
            self.get_logger().warn(f"Invalid depth: {depth_param}, defaulting to 10")
            depth = 10

        # return the QoSProfile
        return QoSProfile(
            reliability=reliability,
            history=history,
            depth=depth
        )


    def get_current_wrench(self):
        """
        Callback function to publish the current wrench of all connected sensors.

        :return: None
        """
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        # read data and create messages for all connected sensors
        msg = []
        for i in range(len(self.sensors)):

            # read data from sensor
            recording = self.sensors[i].record_sample()
            
            # create message
            msg.append(WrenchStamped())
            msg[i].header = header
            msg[i].wrench.force.x = recording.force.x
            msg[i].wrench.force.y = recording.force.y
            msg[i].wrench.force.z = recording.force.z
            msg[i].wrench.torque.x = recording.torque.x
            msg[i].wrench.torque.y = recording.torque.y
            msg[i].wrench.torque.z = recording.torque.z

        # publish messages
        for i in range(len(self.sensors)):
            self.pub_[i].publish(msg[i])


def main(args=None):
    """
    Ros node for the Resense Hex 21 sensor.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("""
        RESENSE HEX 21      
         ____________
        /            \ 
       |--------------|
       |              |
       |              |
        \____________/
           |     |
        Forces  Torques
        """)

        print("\nResense ROS node is running... Press <ctrl+c> to stop the node. \nResults are published on topic resense_<sensor_number> \n")

        rclpy.init(args=args)

        resense_node = ResenseNode(["/dev/ttyACM0"], 2)

        rclpy.spin(resense_node)

    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        resense_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()