from setuptools import find_packages, setup

package_name = 'impact'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'actuated_umi_node = impact.actuated_umi.actuated_umi_node:main',
            'realsense_d405_node = impact.realsense_d405.realsense_d405_node:main',
            'detect_aruco_node = impact.realsense_d405.detect_aruco_node:main',
        ],
    },
)
