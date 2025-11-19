import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'object_detector_husky'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pt')),
    ],
    zip_safe=True,
    maintainer='emil',
    maintainer_email='emil@example.com',
    description='YOLO GNN Refiner - Computer vision package for bounding box refinement using GNN',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'object_detector_husky = object_detector_husky.object_detector_husky:main',
        ],
    },
)
