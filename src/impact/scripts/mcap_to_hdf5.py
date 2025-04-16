import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

from aruco_msgs.msg import ArUcoDistStamped

import numpy as np
import h5py


class RosbagReader():

    def __init__(self, input_bag: str):
        """
        Rosbag reader class to read messages from a bag file.

        :param input_bag: path to the bag file
        :return: None
        """
        
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(
            rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )

        self.topic_types = self.reader.get_all_topics_and_types()


    def __del__(self):
        """
        Destructor to close the reader object.
        
        :return: None
        """

        del self.reader


    def typename(self, topic_name):
        """
        Get the message type of a topic.

        :param topic_name: name of the topic
        :return: message type of the topic
        :raises ValueError: if the topic is not in the bag
        """
        
        for topic_type in self.topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        
        raise ValueError(f"topic {topic_name} not in bag")


    def read_messages(self):
        """
        Generator function to read messages from the bag file.

        :return: topic name, message, timestamp
        """

        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            msg_type = get_message(self.typename(topic))
            msg = deserialize_message(data, msg_type)
            
            yield topic, msg, timestamp


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", help="input bag path (folder or filepath) to read from"
    )
    args = parser.parse_args()

    # create reader object
    rr = RosbagReader(args.input)

    # create hdf5 file
    hdf5_filename = "../hdf5_data/" + args.input.split("/")[-1].split(".")[0] + ".hdf5"
    hdf5_file = h5py.File(hdf5_filename, "w")

    # create hdf5 groups
    gs_mini_grp = hdf5_file.create_group("gelsight_mini")
    gs_mini_img = []
    gs_mini_timestamps = []

    realsense_d405_grp = hdf5_file.create_group("realsense_d405")
    realsense_d405_color_img = []
    realsense_d405_depth_img = []
    realsense_d405_aruco_dist = []
    realsense_d405_timestamps = []

    feats_grp = hdf5_file.create_group("feats")
    feats_fz = []
    feats_fz_dist = []
    feats_timestamps = []

    for topic, msg, timestamp in rr.read_messages():
        #if isinstance(msg, ArUcoDistStamped):
        #    print(f"{topic} [{timestamp}]: '{msg.distance}'")

        #if topic == "/actuated_umi_motor_state":
        #    print(f"{topic} [{timestamp}]: '{msg.position}'")

        if topic == "/gelsight_mini_image":
            gs_mini_img.append(msg.data)
            gs_mini_timestamps.append(timestamp)

        if topic == "/feats_fz":
            print(f"{topic} [{timestamp}]: '{msg.f}'")
            feats_fz.append(msg.f)
            feats_fz_dist.append(msg.fd)
            feats_timestamps.append(timestamp)

    # save dataset
    gs_mini_grp.create_dataset(
        "gs_mini_img",
        data=np.array(gs_mini_img, dtype=np.uint8),
    )
    gs_mini_grp.create_dataset(
        "gs_mini_timestamps",
        data=np.array(gs_mini_timestamps, dtype=np.float32),
    )


    feats_grp.create_dataset(
        "feats_fz",
        data=np.array(feats_fz, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fz_dist",
        data=np.array(feats_fz_dist, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_timestamps",
        data=np.array(feats_timestamps, dtype=np.float32),
    )

    # close hdf5 file
    hdf5_file.close()


if __name__ == "__main__":

    main()