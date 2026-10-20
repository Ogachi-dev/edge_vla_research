"""エピソードを「区間」の列として表すデータ型。docs/track2/33_moveit_demo_gen.md 4.1

demo_generator（計画）と episode_runner（実行・記録）の間で受け渡す。JSON にして meta.json にも残す。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evla.planning.randomize import SceneSample, TrajectoryParams


@dataclass(frozen=True)
class JointTrajectoryData:
    joint_names: tuple[str, ...]
    times_s: tuple[float, ...]                 # 区間の開始からの時刻（time_from_start）
    positions_rad: tuple[tuple[float, ...], ...]

    @property
    def duration_s(self) -> float:
        return self.times_s[-1] if self.times_s else 0.0

    def sample(self, t: float) -> tuple[float, ...]:
        """時刻 t の目標位置を線形補間で返す（範囲外は端の値）。記録の action の参考値に使える。"""
        import numpy as np

        pts = np.asarray(self.positions_rad)
        return tuple(float(np.interp(t, self.times_s, pts[:, j])) for j in range(pts.shape[1]))


@dataclass(frozen=True)
class ArmSegment:
    name: str                                  # "approach", "descend", "lift", "transfer", "lower", "retreat", "home"
    trajectory: JointTrajectoryData
    kind: str = "arm"


@dataclass(frozen=True)
class GripperSegment:
    name: str                                  # "close", "open"
    target_rad: float
    duration_s: float
    kind: str = "gripper"


@dataclass(frozen=True)
class WaitSegment:
    name: str
    duration_s: float
    kind: str = "wait"


Segment = ArmSegment | GripperSegment | WaitSegment


@dataclass(frozen=True)
class EpisodePlan:
    episode_seed: int
    scene: SceneSample
    params: TrajectoryParams
    segments: tuple[Segment, ...]
    perturbations: tuple[dict[str, Any], ...] = ()

    @property
    def duration_s(self) -> float:
        total = 0.0
        for s in self.segments:
            total += s.trajectory.duration_s if isinstance(s, ArmSegment) else s.duration_s  # 【穴①】腕の区間かどうか（腕だけ長さの持ち方が違う）
        return total

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
