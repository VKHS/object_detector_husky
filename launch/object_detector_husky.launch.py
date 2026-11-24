# Copyright 2025-2026 Abdolvahab Khalili Sadaghiani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch the YOLO -> GNN box refiner of object_detector_husky."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Declare the launch arguments and start the refiner executable."""
    bundled_weights = PathJoinSubstitution(
        [FindPackageShare('object_detector_husky'), 'weights', 'weights.pt'])

    arguments = [
        DeclareLaunchArgument(
            'mode', default_value='train_detect',
            description='Run mode: train, detect or train_detect'),
        DeclareLaunchArgument(
            'model', default_value=bundled_weights,
            description='YOLO weights (.pt); default: the bundled COCO YOLOv8n'),
        DeclareLaunchArgument(
            'train_dir',
            default_value='/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017',
            description='Folder with the training images'),
        DeclareLaunchArgument(
            'train_annot',
            default_value='/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017',
            description='Folder with the YOLO .txt labels of the training images'),
        DeclareLaunchArgument(
            'test_dir', default_value='/proj/vahabkhalili/users/x_abdkh/test_results',
            description='Folder with the images to detect on'),
        DeclareLaunchArgument(
            'out_dir', default_value='/proj/vahabkhalili/users/x_abdkh/test_results_COCO',
            description='Output folder (config, cache, checkpoint, annotated images)'),
        DeclareLaunchArgument(
            'epochs', default_value='5',
            description='Number of training epochs'),
        DeclareLaunchArgument(
            'device', default_value='cuda',
            description='Device: cuda or cpu'),
    ]

    refiner = Node(
        package='object_detector_husky',
        executable='object_detector_husky',
        name='object_detector_husky',
        output='screen',
        emulate_tty=True,
        arguments=[
            '--mode', LaunchConfiguration('mode'),
            '--model', LaunchConfiguration('model'),
            '--train_dir', LaunchConfiguration('train_dir'),
            '--train_annot', LaunchConfiguration('train_annot'),
            '--test_dir', LaunchConfiguration('test_dir'),
            '--out_dir', LaunchConfiguration('out_dir'),
            '--epochs', LaunchConfiguration('epochs'),
            '--device', LaunchConfiguration('device'),
        ],
    )

    return LaunchDescription(arguments + [refiner])
