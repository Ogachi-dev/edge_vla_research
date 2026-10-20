from setuptools import setup

package_name = "evla_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "grasp_judge_node = evla_nodes.grasp_judge_node:main",  # 【穴①】「モジュール:関数」の形（ros2 run evla_nodes grasp_judge_node で起動できるようにする）
            "episode_runner = evla_nodes.episode_runner:main",
            "policy_node = evla_nodes.policy_node:main",
            "eval_runner = evla_nodes.eval_runner:main",
        ],
    },
)
