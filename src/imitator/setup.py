from setuptools import find_packages, setup

package_name = 'imitator'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/inference_qos.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='IMITATOR ROS2 Jazzy Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'imitator_node = imitator.imitator_node:main',
            'imitator_node_grape_force = imitator.imitator_node_grape_force:main',
            'imitator_node_grape_gripper = imitator.imitator_node_grape_gripper:main',
            'imitator_node_fdz = imitator.imitator_node_fdz:main',
        ],
    },
)
