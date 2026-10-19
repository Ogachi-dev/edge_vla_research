"""シミュレーションと方策の間でやり取りする共通データ型。docs/track1/21_libero_setup.md 4"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Observation:
    images: dict[str, npt.NDArray[np.uint8]]   # {"image": (H, W, 3), "image2": (H, W, 3)}。方策に渡す向き（180°回転済み）
    state: npt.NDArray[np.float32]             # (8,)。LIBEROなら [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    task: str                                  # 自然言語の指示文
    sim_step: int = 0                          # エピソード開始からのステップ数（待機ステップを除く）


@dataclass(frozen=True)
class StepResult:
    obs: Observation
    success: bool
    done: bool          # 成功、または最大ステップ到達
    info: dict[str, Any] = field(default_factory=dict)
