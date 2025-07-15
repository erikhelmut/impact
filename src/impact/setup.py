import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'impact'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='IMPACT ROS2 Jazzy Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'force_control = impact.force_control:main',
            'gripper_control = impact.gripper_control:main',
            'binary_control = impact.binary_control:main',
            'force_trajectory_publisher = impact.force_trajectory_publisher:main',
        ],
    },
)
