"""設定から方策を作る。docs/track1/22_policy_interface.md 4"""

from __future__ import annotations

from evla.policy.base import Policy, PolicyConfig


def make_policy(cfg: PolicyConfig) -> Policy:
    """cfg.runtime に応じて TorchSmolVLA / OVSmolVLA / VlaCppPolicy を返す。
    import はこの関数の中で遅延して行う（使わないランタイムのライブラリを読み込まない）。"""
    if cfg.runtime == "torch":
        from evla.policy.torch_policy import TorchSmolVLA

        return TorchSmolVLA(cfg)  # 【穴①】設定を渡してインスタンスを作る
    if cfg.runtime == "ov":
        from evla.policy.ov_policy import OVSmolVLA

        return OVSmolVLA(cfg)
    if cfg.runtime == "vlacpp":
        from evla.policy.vlacpp_policy import VlaCppPolicy

        return VlaCppPolicy(cfg)
    raise ValueError(f"未知のランタイム: {cfg.runtime}")
