import os

import rclpy
from rclpy.node import Node
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

        # create publisher for all connected sensors
        self.pub_ = []
        for i in range(len(serial_ports)):
            self.pub_.append(self.create_publisher(WrenchStamped, "resense_{}".format(i), 10))

        timer_period = 1 / self.frequencyModes[frequencyMode]
        self.timer = self.create_timer(timer_period, self.get_current_wrench)


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