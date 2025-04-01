from setuptools import find_packages, setup

package_name = 'feats'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='FEATS ROS2 Jazzy Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'feats_node = feats.feats_node:main',
        ],
    },
)
