import os
import sys; sys.path.append("/home/erik/impact/src/imitator/lerobot")
import argparse

import h5py
import numpy as np
import cv2 
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES


def convert_hdf5_to_lerobot_dataset(hdf5_paths, output_dataset_dir, task_name="mystery_mission", fps=25, use_videos=False):
    """
    Converts a list of HDF5 files (each an episode) to a LeRobotDataset.
    Main logic of the method was created by Google Gemini.

    :param hdf5_paths: list of paths to HDF5 files
    :param output_dataset_dir: directory where the LeRobotDataset (e.g., 'my_repo_id/') will be created
    :param task_name: task name for the dataset
    :param fps: frames per second
    :param use_videos: whether to store image sequences as videos (.mp4) or embed images in Parquet files
    :return: LeRobotDataset object
    """

    output_dataset_dir = Path(output_dataset_dir)
    repo_id = output_dataset_dir.name  # use directory name as repo_id

    # LeRobotDataset will add 'index', 'episode_index', 'frame_index', 'timestamp',
    # 'task_index', 'is_first', 'is_last', etc. if DEFAULT_FEATURES are included
    # for images, specify 'image' or 'video' as dtype
    # the 'task' key in `add_frame` will be handled and converted to 'task_index'
    # you don't need 'task' in the initial `features` dict for `create`
    features = {
        **DEFAULT_FEATURES,  # recommended to include
        "observation.state": {"shape": (9,), "dtype": "float32", "names": ["feats_fz", "aruco_dist", "optitrack_trans_x", "optitrack_trans_y", "optitrack_trans_z", "optitrack_rot_x", "optitrack_rot_y", "optitrack_rot_z", "optitrack_rot_w"]},
        "observation.image": {
            "shape": (96, 96, 3),
            "dtype": "video" if use_videos else "image",
            "names": ["height", "width", "channel"]
        },
        "action": {"shape": (9,), "dtype": "float32", "names": ["feats_fz", "aruco_dist", "optitrack_trans_x", "optitrack_trans_y", "optitrack_trans_z", "optitrack_rot_x", "optitrack_rot_y", "optitrack_rot_z", "optitrack_rot_w"]},
    }
    
    # if there is no reward/done in the HDF5 files, remove them from DEFAULT_FEATURES
    if "reward" in features: del features["reward"]
    if "done" in features: del features["done"]
    if "is_terminal" in features: del features["is_terminal"]  # often same as done

    # create/clean output directory
    if output_dataset_dir.exists():
        
        print(f"Warning: Output directory {output_dataset_dir} already exists.")
        user_input = input(f"Do you want to remove it and create a new dataset? (yes/NO): ")
        
        if user_input.lower() == "yes":
            shutil.rmtree(output_dataset_dir)
            print(f"Removed {output_dataset_dir}.")
        else:
            print("Exiting. Please remove the directory manually or choose a different one.")
            return None
    
    output_dataset_dir.parent.mkdir(parents=True, exist_ok=True) # ensure parent exists

    # create LeRobotDataset - the 'root' for `create` is the actual directory for this dataset
    print(f"Creating LeRobotDataset '{repo_id}' at: {output_dataset_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,  # this is mostly for naming/identification if pushed to Hub
        root=output_dataset_dir,  # the direct path where this dataset will live
        fps=fps,
        features=features,
        use_videos=use_videos,
    )
    print("LeRobotDataset object created.")

    # process HDF5 files
    for hdf5_file_path_str in hdf5_paths:
        
        hdf5_file_path = Path(hdf5_file_path_str)
        print(f"Processing episode from HDF5: {hdf5_file_path.name}")
        
        try:
            with h5py.File(hdf5_file_path, "r") as f:
                # custom hdf5 file structure
                feats_fz = f["feats/feats_fz"][:]
                aruco_dist = f["realsense_d405/realsense_d405_aruco_dist"][:]
                raw_img_h5 = f["realsense_d405/realsense_d405_color_img"][:]
                optitrack_trans_x = f["optitrack/optitrack_trans_x"][:]
                optitrack_trans_y = f["optitrack/optitrack_trans_y"][:]
                optitrack_trans_z = f["optitrack/optitrack_trans_z"][:]
                optitrack_rot_x = f["optitrack/optitrack_rot_x"][:]
                optitrack_rot_y = f["optitrack/optitrack_rot_y"][:]
                optitrack_rot_z = f["optitrack/optitrack_rot_z"][:]
                optitrack_rot_w = f["optitrack/optitrack_rot_w"][:]

                # reshape if necessary and ensure they are 1D time series first
                feats_fz = feats_fz.reshape(-1, 1) if feats_fz.ndim == 1 else feats_fz
                aruco_dist = aruco_dist.reshape(-1, 1) if aruco_dist.ndim == 1 else aruco_dist
                optitrack_trans_x = optitrack_trans_x.reshape(-1, 1) if optitrack_trans_x.ndim == 1 else optitrack_trans_x
                optitrack_trans_y = optitrack_trans_y.reshape(-1, 1) if optitrack_trans_y.ndim == 1 else optitrack_trans_y
                optitrack_trans_z = optitrack_trans_z.reshape(-1, 1) if optitrack_trans_z.ndim == 1 else optitrack_trans_z
                optitrack_rot_x = optitrack_rot_x.reshape(-1, 1) if optitrack_rot_x.ndim == 1 else optitrack_rot_x
                optitrack_rot_y = optitrack_rot_y.reshape(-1, 1) if optitrack_rot_y.ndim == 1 else optitrack_rot_y
                optitrack_rot_z = optitrack_rot_z.reshape(-1, 1) if optitrack_rot_z.ndim == 1 else optitrack_rot_z
                optitrack_rot_w = optitrack_rot_w.reshape(-1, 1) if optitrack_rot_w.ndim == 1 else optitrack_rot_w

                # convert raw_img_h5 from bgr to rgb
                raw_img_h5 = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in raw_img_h5], dtype=np.uint8)

                # resize images to a smaller size
                resized_images_np = np.array([
                    cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                    for frame in raw_img_h5
                ], dtype=np.uint8)

                # determine trajectory length (T-1 transitions)
                # all source arrays for state/action/image must have at least T frames
                T = min(feats_fz.shape[0], aruco_dist.shape[0], resized_images_np.shape[0], optitrack_trans_x.shape[0],
                        optitrack_trans_y.shape[0], optitrack_trans_z.shape[0], optitrack_rot_x.shape[0], optitrack_rot_y.shape[0],
                        optitrack_rot_z.shape[0], optitrack_rot_w.shape[0])

                if T < 2:  # need at least one state and one next_state (action)
                    print(f"  Skipping {hdf5_file_path.name} (too short, T={T}). Needs at least 2 frames for one transition.")
                    continue

                num_transitions = T - 1

                # prepare data for LeRobotDataset
                # state_t: current state
                # action_t: action taken at state_t (here, defined as next_state features)
                # image_t: image at state_t
                #current_state_data = np.concatenate([feats_fz[:num_transitions], aruco_dist[:num_transitions], optitrack_trans_x[:num_transitions], optitrack_trans_y[:num_transitions], optitrack_trans_z[:num_transitions], optitrack_rot_x[:num_transitions], optitrack_rot_y[:num_transitions], optitrack_rot_z[:num_transitions], optitrack_rot_w[:num_transitions]], axis=1)
                #action_data = np.concatenate([feats_fz[1:T], aruco_dist[1:T], optitrack_trans_x[1:T], optitrack_trans_y[1:T], optitrack_trans_z[1:T], optitrack_rot_x[1:T], optitrack_rot_y[1:T], optitrack_rot_z[1:T], optitrack_rot_w[1:T]], axis=1)  # next state as action
                #image_data_for_episode = resized_images_np[:num_transitions]

                placeholder = np.zeros(feats_fz.shape)

                current_state_data = np.concatenate([feats_fz[:num_transitions], aruco_dist[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions], placeholder[:num_transitions]], axis=1)
                action_data = np.concatenate([feats_fz[1:T], aruco_dist[1:T], placeholder[1:T], placeholder[1:T], placeholder[1:T], placeholder[1:T], placeholder[1:T], placeholder[1:T], placeholder[1:T]], axis=1)  # next state as action
                image_data_for_episode = resized_images_np[:num_transitions]

                print(f"  Episode length (transitions): {num_transitions}")

                for i in range(num_transitions):
                    frame_for_lerobot = {
                        "observation.state": current_state_data[i].astype(np.float32),
                        # `add_frame` expects PIL.Image or HWC np.array for images
                        "observation.image": image_data_for_episode[i],
                        "action": action_data[i].astype(np.float32),
                        "task": task_name  # task description for this episode/frame
                    }
                    dataset.add_frame(frame_for_lerobot)

                dataset.save_episode()  # saves the buffered frames as one episode
                print(f"  Saved episode from {hdf5_file_path.name} to LeRobotDataset.")
                # `save_episode` also clears the buffer for the next episode

        except Exception as e:
            print(f"Error processing {hdf5_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Skip to next file on error

    print(f"\nDataset conversion complete. Output at: {output_dataset_dir}")
    
    return dataset


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobotDataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the LeRobotDataset.")
    parser.add_argument("--hdf5_dir", type=str, required=True, help="Paths to folder storing the HDF5 files.")
    args = parser.parse_args()

    # iterate over alle hdf5 files in the directory
    hdf5_files = []
    
    for filename in os.listdir(args.hdf5_dir):
        if filename.endswith(".hdf5"):
            hdf5_files.append(os.path.join(args.hdf5_dir, filename))

    # create the LeRobotDataset
    created_dataset = convert_hdf5_to_lerobot_dataset(
        hdf5_paths=hdf5_files,
        output_dataset_dir=args.output_dir,
        fps=25,
        use_videos=False  # set to True if you prefer .mp4 videos for images
    )

    if created_dataset:
        print(f"\nSuccessfully created dataset: {created_dataset.repo_id}")
        print(f"Root path: {created_dataset.root}")
        print(f"Total episodes: {created_dataset.num_episodes}")
        print(f"Total frames: {created_dataset.num_frames}")
        print(f"Features: {list(created_dataset.features.keys())}")

        # you can now load it like any other LeRobotDataset
        # repo_id is the path to the dataset directory
        # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        # loaded_ds = LeRobotDataset(repo_id, episodes=[0, 2, 5, ...])
        # print(f"First item: {loaded_ds[0]}")