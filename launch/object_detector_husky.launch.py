import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch-fil för yolo_gnn_refiner.
    
    Detta startar yolo_gnn_refiner med konfigurerbara argument.
    """
    
    # Launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value=TextSubstitution(text='train_detect'),
        description='Körläge: train, detect, eller train_detect'
    )
    
    model_arg = DeclareLaunchArgument(
        'model',
        default_value=TextSubstitution(text='yolov8n.pt'),
        description='Sökväg till YOLO-modell (.pt-fil)'
    )
    
    train_dir_arg = DeclareLaunchArgument(
        'train_dir',
        default_value=TextSubstitution(text='/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017'),
        description='Sökväg till träningsbilder'
    )
    
    train_annot_arg = DeclareLaunchArgument(
        'train_annot',
        default_value=TextSubstitution(text='/proj/vahabkhalili/users/x_abdkh/coco_data_auto/train2017'),
        description='Sökväg till träningsannoteringar'
    )
    
    test_dir_arg = DeclareLaunchArgument(
        'test_dir',
        default_value=TextSubstitution(text='/proj/vahabkhalili/users/x_abdkh/test_results'),
        description='Sökväg till testbilder'
    )
    
    out_dir_arg = DeclareLaunchArgument(
        'out_dir',
        default_value=TextSubstitution(text='/proj/vahabkhalili/users/x_abdkh/test_results_COCO'),
        description='Utdatamapp för resultat'
    )
    
    epochs_arg = DeclareLaunchArgument(
        'epochs',
        default_value=TextSubstitution(text='5'),
        description='Antal träningsepoker'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value=TextSubstitution(text='cuda'),
        description='Enhet att köra på: cuda eller cpu'
    )
    
    # Node som kör yolo_gnn_refiner
    yolo_gnn_refiner_node = Node(
        package='yolo_gnn_refiner',
        executable='yolo_gnn_refiner',
        name='yolo_gnn_refiner',
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
        ]
    )
    
    return LaunchDescription([
        mode_arg,
        model_arg,
        train_dir_arg,
        train_annot_arg,
        test_dir_arg,
        out_dir_arg,
        epochs_arg,
        device_arg,
        yolo_gnn_refiner_node,
    ])

