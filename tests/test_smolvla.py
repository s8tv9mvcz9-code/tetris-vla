"""SmolVLA アダプタのテスト。

lerobot が入っていない環境ではモジュール全体を skip する
(コア PoC は lerobot 無しで完結するため、必須依存にしていない)。
重みのダウンロードを伴うテストは `-m slow` を付けた時だけ走る。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="lerobot 未導入 (pip install 'lerobot[smolvla]')")
pytest.importorskip("torch")

from tetris_vla.parachute import (  # noqa: E402
    ParachuteConfig,
    ParachuteWorld,
    Thrust,
    make_terrain,
    render_flight,
    spawn_lander,
    to_png,
)
from tetris_vla.render import RenderConfig  # noqa: E402
from tetris_vla.smolvla_pilot import (  # noqa: E402
    ACTION_DIM,
    IMG_SIZE,
    STATE_DIM,
    NoTarget,
    OracleTarget,
    decode_action,
    encode_action,
    encode_image,
    encode_state,
    image_from_uint8,
    task_for,
)


def _obs(seed: int = 0, ticks: int = 10):
    from tetris_vla.parachute import FlightObservation

    cfg = ParachuteConfig()
    rc = RenderConfig()
    board = make_terrain(cfg, seed)
    L = spawn_lander(cfg, "O", seed, board)
    w = ParachuteWorld(cfg, board, L)
    for _ in range(ticks):
        w.step(Thrust.COAST)
    img = render_flight(w, cfg, rc)
    return FlightObservation(png=to_png(img), image=img, kind=L.kind, rot=L.rot,
                             x=L.x, y=L.y, vx=L.vx, vy=L.vy, fuel=L.fuel,
                             tick=w.tick, t=w.t, width=cfg.width, height=cfg.height,
                             cfg=cfg, render_cfg=rc)


# --- 行動のエンコード / デコード -----------------------------------------


@pytest.mark.parametrize("act", list(Thrust))
def test_action_roundtrip(act: Thrust) -> None:
    """離散コマンド -> 連続ベクトル -> 離散コマンド が一致すること。"""
    v = encode_action(act)
    assert v.shape == (ACTION_DIM,)
    assert decode_action(v) == act


def test_decode_action_is_dead_band_around_zero() -> None:
    """小さい出力は噴射しない。燃料を無駄にしないための不感帯。"""
    assert decode_action(np.zeros(ACTION_DIM)) == Thrust.COAST
    assert decode_action(np.array([0.2, 0, 0, 0, 0, 0])) == Thrust.COAST
    assert decode_action(np.array([0.9, 0, 0, 0, 0, 0])) == Thrust.RIGHT
    assert decode_action(np.array([-0.9, 0, 0, 0, 0, 0])) == Thrust.LEFT


def test_vent_only_when_not_thrusting_sideways() -> None:
    assert decode_action(np.array([0.0, 0.9, 0, 0, 0, 0])) == Thrust.VENT
    assert decode_action(np.array([0.9, 0.9, 0, 0, 0, 0])) == Thrust.RIGHT


# --- 観測のエンコード -----------------------------------------------------


def test_state_matches_the_declared_dim_and_is_bounded() -> None:
    obs = _obs()
    s = encode_state(obs, target_col=3)
    assert s.shape == (STATE_DIM,)
    assert s.dtype == np.float32
    assert np.all(np.abs(s) <= 1.5), s


def test_state_last_dim_carries_the_commanded_target() -> None:
    """最後の次元が「上位からの指令」。ここを差し替えて指令元を変える。"""
    obs = _obs()
    left = encode_state(obs, target_col=0)
    right = encode_state(obs, target_col=obs.width - 1)
    none = encode_state(obs, target_col=None)
    assert left[-1] < none[-1] < right[-1]
    assert np.allclose(left[:-1], right[:-1]), "目標以外の次元は変わらない"
    assert none[-1] == 0.0


def test_image_is_padded_to_the_expected_tensor() -> None:
    import torch

    t = encode_image(_obs())
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (3, IMG_SIZE, IMG_SIZE)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_uint8_storage_path_matches_direct_encoding() -> None:
    """教師データは uint8 で持つ (メモリ節約)。復元経路が一致すること。"""
    import torch

    obs = _obs()
    direct = encode_image(obs)
    viaint = image_from_uint8(np.asarray(obs.image.convert("RGB"), dtype=np.uint8))
    assert torch.allclose(direct, viaint)


# --- 言語条件付け ---------------------------------------------------------


def test_task_string_carries_the_target_column() -> None:
    """VLM の判断は自然言語の指示として VLA に渡る (階層型の接点)。"""
    assert "column 7" in task_for(7)
    assert task_for(None) != task_for(0)


# --- 目標の供給元 ---------------------------------------------------------


def test_oracle_target_matches_the_reference_solution() -> None:
    from tetris_vla.parachute import target_column
    from tetris_vla.render import decode

    obs = _obs(seed=4)
    tgt, latency, note = OracleTarget()(obs)
    terrain = decode(obs.image, obs.width, obs.height, obs.render_cfg).settled
    assert tgt == target_column(terrain, obs.kind, obs.rot)
    assert latency >= 0 and "参照解" in note


def test_no_target_source_returns_nothing() -> None:
    tgt, latency, note = NoTarget()(_obs())
    assert tgt is None and latency == 0.0


def test_vlm_target_only_queries_on_its_own_schedule() -> None:
    """上位 VLM は every_ticks ごとにしか呼ばれない = 遅さが償却される。"""
    from tetris_vla.smolvla_pilot import VLMTarget

    src = VLMTarget(every_ticks=20)
    obs = _obs(ticks=1)
    obs.tick = 5
    src._next_tick = 100          # まだ問い合わせ時刻ではない
    tgt, latency, note = src(obs)
    assert tgt is None and latency == 0.0 and note == ""
    assert src.calls == 0, "スケジュール外では VLM を呼ばない"


# --- 重みを要するテスト (既定では skip) ----------------------------------


@pytest.mark.slow
def test_policy_builds_and_returns_action_chunks() -> None:
    """1 回の推論で n_action_steps 手ぶんが返る = VLA の action chunking。"""
    import torch

    from tetris_vla.smolvla_pilot import build_policy

    policy, pre, post, cfg, dev = build_policy(chunk_size=50)
    assert cfg.n_action_steps == 50
    obs = _obs()
    batch = {"observation.state": torch.from_numpy(encode_state(obs, 3)),
             "observation.images.camera1": encode_image(obs),
             "task": task_for(3)}
    policy.reset()
    with torch.inference_mode():
        first = post(policy.select_action(pre(batch)))
        assert tuple(first.shape)[-1] == ACTION_DIM
        # 続く 49 手はキューから出るだけ (再推論しない)
        for _ in range(49):
            policy.select_action(pre(batch))
        assert len(policy._queues["action"]) == 0
