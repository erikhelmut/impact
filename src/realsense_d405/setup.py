from setuptools import find_packages, setup

package_name = 'realsense_d405'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/data_collection_qos.yaml']),
        ('share/' + package_name + '/config', ['config/inference_qos.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='RealSense D405 ROS2 Jazzy Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'realsense_d405_node = realsense_d405.realsense_d405_node:main',
            'detect_aruco_node = realsense_d405.detect_aruco_node:main',
        ],
    },
)
