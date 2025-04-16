import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

import os
import yaml

import cv2
import numpy as np
import torch

from feats.dataloader import normalize, unnormalize
from feats.unet import UNet
from feats_msgs.msg import ForceDistStamped


class FEATS:

    def __init__(self):
        """
        This class is used to make predictions using the FEATS model. It loads the model and the configuration file, and sets the device to GPU if available.

        :return: None
        """
        
        # specify device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # define package path
        self.package_path = os.path.dirname(__file__).split("install")[0] + "src/feats/"

        # load config file
        self.config = yaml.load(open(self.package_path + "config/predict_config.yaml", "r"),  Loader=yaml.FullLoader)

        # load model
        self.model = UNet(enc_chs=self.config["enc_chs"], dec_chs=self.config["dec_chs"], out_sz=self.config["output_size"])
        self.model.load_state_dict(torch.load(self.package_path + self.config["model"], map_location=torch.device("cpu"), weights_only=True))
        self.model.eval().to(self.device)


    def make_prediction(self, img):
        """
        Make prediction using the FEATS model.

        :param img: current image of the gelsight mini
        :return: pred_grid_x, pred_grid_y, pred_grid_z: predicted force distributions
        """

        # store data in dictionary
        data = {}
        data["gs_img"] = img
        
        # normalize data
        data = normalize(data, self.package_path + self.config["norm_file"])

        # convert to torch tensor
        gs_img = torch.from_numpy(data["gs_img"]).float()
        gs_img = gs_img.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        # load calibration file
        if self.config["calibration_file"] is not None:
            calibration = np.load(self.config["calibration_file"])
            rows, cols = 240, 320
            M = np.float32([[1, 0, calibration[0]], [0, 1, calibration[1]]])

        # prepare input data
        if self.config["calibration_file"] is not None:
            inputs_prewarp = data["gs_img"]
            inputs_warp = cv2.warpAffine(inputs_prewarp,  M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            inputs = torch.from_numpy(inputs_warp).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        else:
            inputs = gs_img

        # get model prediction
        outputs = self.model(inputs)

        # unnormalize the outputs
        outputs_transf = outputs.squeeze(0).permute(1, 2, 0)
        pred_grid_x = unnormalize(outputs_transf[:, :, 0], "grid_x", self.package_path + self.config["norm_file"])
        pred_grid_y = unnormalize(outputs_transf[:, :, 1], "grid_y", self.package_path + self.config["norm_file"])
        pred_grid_z = unnormalize(outputs_transf[:, :, 2], "grid_z", self.package_path + self.config["norm_file"])

        # convert to numpy
        pred_grid_x = pred_grid_x.cpu().detach().numpy()
        pred_grid_y = pred_grid_y.cpu().detach().numpy()
        pred_grid_z = pred_grid_z.cpu().detach().numpy()

        return pred_grid_x, pred_grid_y, pred_grid_z


class FEATSNode(Node):

    def __init__(self):
        """
        This node subscribes to the current image of the gelsight mini and makes predictions using the FEATS model. It publishes the predicted force distribution as an image.

        :return: None
        """
        
        super().__init__("feats_node")

        # initialize FEATS model
        self.feats = FEATS()

        # define color limits
        self.clim_x = (-0.029, 0.029)
        self.clim_y = (-0.029, 0.029)
        self.clim_z = (-0.17, 0.0)

        # create subscriber to get current image of the gelsight mini
        gelsight_subscriber_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.gelsight_subscriber = self.create_subscription(Image, "gelsight_mini_image", self.pub_prediction, gelsight_subscriber_qos_profile)
        self.gelsight_subscriber  # prevent unused variable warning

        # create publisher for FEATS
        feats_publisher_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.feats_x_publisher_ = self.create_publisher(ForceDistStamped, "feats_fx", feats_publisher_qos_profile)
        self.feats_y_publisher_ = self.create_publisher(ForceDistStamped, "feats_fy", feats_publisher_qos_profile)
        self.feats_z_publisher_ = self.create_publisher(ForceDistStamped, "feats_fz", feats_publisher_qos_profile)
        self.feats_x_image_publisher_ = self.create_publisher(Image, "feats_fx_image", feats_publisher_qos_profile)
        self.feats_y_image_publisher_ = self.create_publisher(Image, "feats_fy_image", feats_publisher_qos_profile)
        self.feats_z_image_publisher_ = self.create_publisher(Image, "feats_fz_image", feats_publisher_qos_profile)


    def pub_prediction(self, msg):
        """
        Publish the prediction of the FEATS model.

        :param msg: current image of the gelsight mini
        :return: None
        """

        # convert ros image message to numpy array
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        # make prediction with FEATS model
        pred_grid_x, pred_grid_y, pred_grid_z = self.feats.make_prediction(img)

        # calculate total forces
        f_x = np.sum(pred_grid_x)
        f_y = np.sum(pred_grid_y)
        f_z = np.sum(pred_grid_z)
        
        # create Image message for x
        pred_grid_x_img = (pred_grid_x - self.clim_x[0]) / (self.clim_x[1] - self.clim_x[0])
        pred_grid_x_img = np.clip(pred_grid_x_img, 0, 1) * 255
        pred_grid_x_img = np.uint8(pred_grid_x_img)
        pred_grid_x_img = cv2.applyColorMap(pred_grid_x_img, cv2.COLORMAP_VIRIDIS)
        pred_grid_x_img = cv2.resize(pred_grid_x_img, (320, 240), interpolation=cv2.INTER_NEAREST)

        pred_grid_x_msg = Image()
        pred_grid_x_msg.header = msg.header
        pred_grid_x_msg.data = pred_grid_x_img.tobytes()
        pred_grid_x_msg.height = pred_grid_x_img.shape[0]
        pred_grid_x_msg.width = pred_grid_x_img.shape[1]
        pred_grid_x_msg.encoding = "bgr8"
        pred_grid_x_msg.step = pred_grid_x_img.shape[1] * 3
        pred_grid_x_msg.header.frame_id = "feats_fx_image"
        self.feats_x_image_publisher_.publish(pred_grid_x_msg)
        
        # create Image message for y
        pred_grid_y_img = (pred_grid_y - self.clim_y[0]) / (self.clim_y[1] - self.clim_y[0])
        pred_grid_y_img = np.clip(pred_grid_y_img, 0, 1) * 255
        pred_grid_y_img = np.uint8(pred_grid_y_img)
        pred_grid_y_img = cv2.applyColorMap(pred_grid_y_img, cv2.COLORMAP_VIRIDIS)
        pred_grid_y_img = cv2.resize(pred_grid_y_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        pred_grid_y_msg = Image()
        pred_grid_y_msg.header = msg.header
        pred_grid_y_msg.data = pred_grid_y_img.tobytes()
        pred_grid_y_msg.height = pred_grid_y_img.shape[0]
        pred_grid_y_msg.width = pred_grid_y_img.shape[1]
        pred_grid_y_msg.encoding = "bgr8"
        pred_grid_y_msg.step = pred_grid_y_img.shape[1] * 3
        pred_grid_y_msg.header.frame_id = "feats_fy_image"
        self.feats_y_image_publisher_.publish(pred_grid_y_msg)

        # create Image message for z
        pred_grid_z_img = (pred_grid_z - self.clim_z[0]) / (self.clim_z[1] - self.clim_z[0])
        pred_grid_z_img = np.clip(pred_grid_z_img, 0, 1) * 255
        pred_grid_z_img = np.uint8(pred_grid_z_img)
        pred_grid_z_img = cv2.applyColorMap(pred_grid_z_img, cv2.COLORMAP_VIRIDIS)
        pred_grid_z_img = cv2.resize(pred_grid_z_img, (320, 240), interpolation=cv2.INTER_NEAREST)

        pred_grid_z_msg = Image()
        pred_grid_z_msg.header = msg.header
        pred_grid_z_msg.data = pred_grid_z_img.tobytes()
        pred_grid_z_msg.height = pred_grid_z_img.shape[0]
        pred_grid_z_msg.width = pred_grid_z_img.shape[1]
        pred_grid_z_msg.encoding = "bgr8"
        pred_grid_z_msg.step = pred_grid_z_img.shape[1] * 3
        pred_grid_z_msg.header.frame_id = "feats_fz_image"
        self.feats_z_image_publisher_.publish(pred_grid_z_msg)
        
        # create ForceDistStamped message
        feats_x_msg = ForceDistStamped()
        feats_x_msg.header = msg.header
        feats_x_msg.force = "f_x"
        feats_x_msg.f = f_x.item()
        feats_x_msg.fd = pred_grid_x.flatten()
        self.feats_x_publisher_.publish(feats_x_msg)

        feats_y_msg = ForceDistStamped()
        feats_y_msg.header = msg.header
        feats_y_msg.force = "f_y"
        feats_y_msg.f = f_y.item()
        feats_y_msg.fd = pred_grid_y.flatten()
        self.feats_y_publisher_.publish(feats_y_msg)

        feats_z_msg = ForceDistStamped()
        feats_z_msg.header = msg.header
        feats_z_msg.force = "f_z"
        feats_z_msg.f = f_z.item()
        feats_z_msg.fd = pred_grid_z.flatten()
        self.feats_z_publisher_.publish(feats_z_msg)


def main(args=None):
    """
    ROS node for making live predictions using the FEATS model.

    :param args: arguments for the ROS node
    :return: None
    """

    try:

        print("FEATS Node is running... Press <ctrl> <c> to stop. \nPredicted force distributions are being published on topic /feats_fX or /feats_fX_image. \n")

        rclpy.init(args=args)

        feats_node = FEATSNode()

        rclpy.spin(feats_node)

    finally:

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        feats_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()
