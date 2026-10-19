import numpy as np
import pytest

from evla.dataset.quality import check_dataset, check_episode, gate_summary
from evla.dataset.raw_episode import RawEpisodeWriter, list_complete_episodes, load_raw_episode

pytest.importorskip("cv2")


def write_ep(root, name, n=10, success=True, black=False, gap=False):
    w = RawEpisodeWriter(root / name)
    for i in range(n):
        t = i / 30 + (0.1 if gap and i > 5 else 0)
        img = np.zeros((48, 64, 3), np.uint8) if black else np.full((48, 64, 3), 128, np.uint8)
        w.add_frame(t, {"top": img, "wrist": img}, np.zeros(6), np.zeros(6), np.zeros(7))
    w.add_event({"type": "success"})
    w.finish({"success": success})


def test_write_and_read(tmp_path):
    write_ep(tmp_path, "ep_000000")
    (tmp_path / "ep_000001").mkdir()   # meta.json の無い（途中で落ちた）エピソード
    eps = list_complete_episodes(tmp_path)
    assert [p.name for p in eps] == ["ep_000000"]
    ep = load_raw_episode(eps[0])
    assert ep.n_frames == 10 and ep.meta["n_frames"] == 10
    assert ep.load_image("top", 3).shape == (48, 64, 3)
    assert check_episode(ep).ok


def test_quality_detects_problems(tmp_path):
    write_ep(tmp_path, "ep_000000", black=True)
    write_ep(tmp_path, "ep_000001", gap=True)
    write_ep(tmp_path, "ep_000002", success=False)
    df = check_dataset(tmp_path)
    assert "black_image" in df.loc[0, "problems"]
    assert "frame_gap" in df.loc[1, "problems"]
    g = gate_summary(df)
    assert g["n_success"] == 2
