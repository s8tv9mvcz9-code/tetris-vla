"""tetris-vla — localVLA を使ったマルチエージェント・テトリス PoC。

各落下ピースが独立したエージェントであり、盤面スナップショット画像だけを頼りに
自分の収まり先を決める。他のピースの意図は見えず、位置だけが見える。
"""

from .engine import Action, ActivePiece, Engine, EngineConfig, Snapshot
from .render import RenderConfig, decode, render, to_ascii
from .agents import (
    HeuristicAgent,
    Intent,
    MockBackend,
    MockConfig,
    Persona,
    PERSONAS,
    RandomAgent,
    VLAAgent,
)
from .runtime import RuntimeConfig, Session

__version__ = "0.1.0"

__all__ = [
    "Action", "ActivePiece", "Engine", "EngineConfig", "Snapshot",
    "RenderConfig", "decode", "render", "to_ascii",
    "HeuristicAgent", "Intent", "MockBackend", "MockConfig", "Persona", "PERSONAS",
    "RandomAgent", "VLAAgent",
    "RuntimeConfig", "Session",
]
