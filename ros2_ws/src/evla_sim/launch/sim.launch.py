"""Gazebo＋SO-101＋コントローラ＋bridge＋把持判定ノードを起動する。docs/track2/31 Step 4

例: ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf
起動時は world を動かしておき（コントローラの起動に時間の進行が要るため）、
記録・評価のノードが pause してロックステップに切り替える。
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("evla_sim"))
    world = LaunchConfiguration("world")
    robot_description = ParameterValue(Command(["xacro ", str(share / "urdf" / "so101_evla.urdf.xacro")]),
                                       value_type=str)  # 【穴①】xacro の出力を YAML として解釈させず、文字列のまま渡す指定

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(get_package_share_directory("ros_gz_sim")) / "launch" / "gz_sim.launch.py")),
        launch_arguments={"gz_args": [world, " -r"]}.items(),  # -r：起動直後から時間を進める
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": robot_description, "use_sim_time": True}])
    spawn = Node(package="ros_gz_sim", executable="create",
                 arguments=["-topic", "robot_description", "-name", "so101"])  # 【穴②】ロボットのモデルを、このトピックの URDF から読む指定
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        parameters=[{"config_file": str(share / "config" / "bridge.yaml"), "use_sim_time": True}],
        arguments=["/world/evla/control@ros_gz_interfaces/srv/ControlWorld",
                   "/world/evla/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
    )
    spawners = [Node(package="controller_manager", executable="spawner", arguments=[name])
                for name in ("joint_state_broadcaster", "arm_controller", "gripper_controller")]
    judge = Node(package="evla_nodes", executable="grasp_judge_node", name="grasp_judge",
                 parameters=[str(share / "config" / "grasp.yaml")])
    # ロボットが出現してからコントローラを起動する
    after_spawn = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=spawners))  # 【穴③】spawn のプロセスが終わったら spawners を起動するイベント

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="/tmp/evla_world_0.sdf"),
        gz, rsp, spawn, bridge, after_spawn, judge,
    ])
