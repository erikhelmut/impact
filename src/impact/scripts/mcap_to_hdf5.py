import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

import copy
import h5py
import numpy as np


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


def convert_timestamp_to_float(timestamp):
    """
    Helper function to convert a timestamp to a float.

    :param timestamp: timestamp to convert
    :return: converted timestamp
    """

    return timestamp.sec + timestamp.nanosec * 1e-9


def time_filter_list(ref_timestamps, target_timestamps, target_list, target_list_name, delta_t=0.1):
    """
    Filter a list based on the closest timestamps in another list.

    :param ref_timestamps: reference timestamps
    :param target_timestamps: target timestamps
    :param target_list: target list to filter
    :param target_list_name: name of the target list
    :return: filtered list
    """

    filtered_list = []

    warn = False
    
    for ts in ref_timestamps:
        # find the closest timestamp in target_timestamps
        closest_ts = min(target_timestamps, key=lambda x: abs(x - ts))
        index = target_timestamps.index(closest_ts)

        # append the corresponding value to the filtered list
        if abs(closest_ts - ts) < delta_t:
            filtered_list.append(target_list[index])
        else:
            filtered_list.append(None)
            warn = True
    
    if warn:
        print(f"Warning: {target_list_name} timestamps are not aligned with gs_mini timestamps. Some values are set to None.")

    return filtered_list


def fill_none_values(seq):
    """
    Fill None values in a list with the first and last non-None values.
    Interpolate None values in the middle.
    
    :param seq: list to fill
    :return: filled list
    """

    # replace None values at the start with the first non-None value
    for i in range(len(seq)):
        if seq[i] is not None:
            seq[:i] = [seq[i]] * i
            break

    # replace None values at the end with the last non-None value
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] is not None:
            seq[i+1:] = [seq[i]] * (len(seq) - i - 1)
            break

    # interpolate None values in the middle
    i = 0
    while i < len(seq):
        if seq[i] is None:
            start = i - 1
            # find the next non-None value
            j = i
            while j < len(seq) and seq[j] is None:
                j += 1
            if j < len(seq):
                # interpolate between seq[start] and seq[j]
                step = (seq[j] - seq[start]) / (j - start)
                for k in range(1, j - start):
                    seq[start + k] = seq[start] + step * k
            i = j
        else:
            i += 1

    return seq


def main():
    """
    MCAP to HDF5 converter script.

    :return: None
    """

    # parse arguments
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

    feats_grp = hdf5_file.create_group("feats")
    feats_fx = []; feats_fy = []; feats_fz = []
    feats_fdx = []; feats_fdy = []; feats_fdz = []
    feats_fx_timestamps = []; feats_fy_timestamps = []; feats_fz_timestamps = []

    realsense_d405_grp = hdf5_file.create_group("realsense_d405")
    realsense_d405_color_img = []; realsense_d405_depth_img = []; realsense_d405_aruco_dist = []
    realsense_d405_color_img_timestamps = []; realsense_d405_depth_img_timestamps = []; realsense_d405_aruco_dist_timestamps = []


    # iterate over messages
    for topic, msg, _ in rr.read_messages():

        if topic == "/gelsight_mini_image":
            gs_mini_img.append(msg.data)
            gs_mini_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/feats_fx":
            feats_fx.append(msg.f)
            feats_fdx.append(msg.fd)
            feats_fx_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/feats_fy":
            feats_fy.append(msg.f)
            feats_fdy.append(msg.fd)
            feats_fy_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/feats_fz":
            feats_fz.append(msg.f)
            feats_fdz.append(msg.fd)
            feats_fz_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/realsense_d405_color_image":
            realsense_d405_color_img.append(msg.data)
            realsense_d405_color_img_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/realsense_d405_depth_image":
            realsense_d405_depth_img.append(msg.data)
            realsense_d405_depth_img_timestamps.append(convert_timestamp_to_float(msg.header.stamp))

        elif topic == "/realsense_d405_aruco_distance":
            realsense_d405_aruco_dist.append(msg.distance)
            realsense_d405_aruco_dist_timestamps.append(convert_timestamp_to_float(msg.header.stamp))


    # close reader object
    del rr

    # remove first and last 10 points from gs_mini_img
    gs_mini_img = gs_mini_img[10:-10]
    gs_mini_timestamps = gs_mini_timestamps[10:-10]

    # filter lists based on timestamps of gs_mini
    feats_fx = time_filter_list(gs_mini_timestamps, feats_fx_timestamps, feats_fx, "feats_fx")
    feats_fdx = time_filter_list(gs_mini_timestamps, feats_fx_timestamps, feats_fdx, "feats_fdx")
    feats_fy = time_filter_list(gs_mini_timestamps, feats_fy_timestamps, feats_fy, "feats_fy")
    feats_fdy = time_filter_list(gs_mini_timestamps, feats_fy_timestamps, feats_fdy, "feats_fdy")
    feats_fz = time_filter_list(gs_mini_timestamps, feats_fz_timestamps, feats_fz, "feats_fz")
    feats_fdz = time_filter_list(gs_mini_timestamps, feats_fz_timestamps, feats_fdz, "feats_fz_dist")
    feats_timestamps = copy.deepcopy(gs_mini_timestamps)

    realsense_d405_color_img = time_filter_list(gs_mini_timestamps, realsense_d405_color_img_timestamps, realsense_d405_color_img, "realsense_d405_color_img")
    realsense_d405_depth_img = time_filter_list(gs_mini_timestamps, realsense_d405_depth_img_timestamps, realsense_d405_depth_img, "realsense_d405_depth_img")
    realsense_d405_aruco_dist = time_filter_list(gs_mini_timestamps, realsense_d405_aruco_dist_timestamps, realsense_d405_aruco_dist, "realsense_d405_aruco_dist")
    realsense_d405_timestamps = copy.deepcopy(gs_mini_timestamps)

    # fill None values in lists
    realsense_d405_aruco_dist = fill_none_values(realsense_d405_aruco_dist)

    # save datasets
    gs_mini_grp.create_dataset(
        "gs_mini_img",
        data=np.array(gs_mini_img, dtype=np.uint8),
    )
    gs_mini_grp.create_dataset(
        "gs_mini_timestamps",
        data=np.array(gs_mini_timestamps, dtype=np.float32),
    )

    feats_grp.create_dataset(
        "feats_fx",
        data=np.array(feats_fx, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fdx",
        data=np.array(feats_fdx, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fy",
        data=np.array(feats_fy, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fdy",
        data=np.array(feats_fdy, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fz",
        data=np.array(feats_fz, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_fz_dist",
        data=np.array(feats_fdz, dtype=np.float32),
    )
    feats_grp.create_dataset(
        "feats_timestamps",
        data=np.array(feats_timestamps, dtype=np.float32),
    )

    realsense_d405_grp.create_dataset(
        "realsense_d405_color_img",
        data=np.array(realsense_d405_color_img, dtype=np.uint8),
    )
    realsense_d405_grp.create_dataset(
        "realsense_d405_depth_img",
        data=np.array(realsense_d405_depth_img, dtype=np.uint8),
    )
    realsense_d405_grp.create_dataset(
        "realsense_d405_aruco_dist",
        data=np.array(realsense_d405_aruco_dist, dtype=np.float32),
    )
    realsense_d405_grp.create_dataset(
        "realsense_d405_timestamps",
        data=np.array(realsense_d405_timestamps, dtype=np.float32),
    )


    # close hdf5 file
    hdf5_file.close()


if __name__ == "__main__":

    main()