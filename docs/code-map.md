# コード地図

このリポジトリを読む / 一部だけ取り込む / 別環境へ移すときの索引。
上から順に読む必要はなく、目的の行を引いて該当ファイルへ飛べばよい。

## 依存と前提

| 項目 | 値 |
|---|---|
| Python | 3.10 以上 (`X \| None` 記法を使用) |
| 必須依存 | `numpy` / `Pillow` / `httpx` |
| 任意依存 | `torch` + `lerobot` (学習モデルを使うときだけ) |
| OS 依存 | **なし。** 下記「環境に触る箇所」以外は純粋な Python |
| ネットワーク | 既定では使わない。実機モデルを指定したときだけ HTTP |
| 乱数 | シミュレータと解析解・mock は `random.Random(seed)` 経由で完全再現。**学習モデルを挟んだ経路だけ再現しない** (下記) |

`numpy` / `Pillow` / `httpx` だけで、シミュレータ・制御層・可視化はすべて動く。
学習モデルを使わない構成なら `torch` は不要。

## 環境に触る箇所

移植時に確認するのはこの 3 つだけで、他はどこで動かしても同じ結果になる。

| 場所 | 何をしている | 別環境での扱い |
|---|---|---|
| `smolvla_pilot.pick_device()` | `mps` → `cuda` → `cpu` の順に選ぶ | 順序を変えるか、`device=` で明示指定 |
| `pinball_agents.VLMStrategist.__init__` | `http://localhost:11434` (Ollama) を既定にする | `host=` で差し替え |
| `smolvla_pilot.build_policy()` | `lerobot/smolvla_base` を取りに行く | `pretrained=None` で重みなし構築 |

`pick_device` 以外に OS 分岐は無い。ファイルパスはすべて `pathlib` 経由で、
シェル呼び出しは可視化スクリプトの `git log` (失敗しても続行) だけ。

## モジュール

| ファイル | 役割 | 外に出している主なもの |
|---|---|---|
| `tetris_vla/pinball.py` | ピンボール題材のシミュレータ。物理・ラダー・シーケンサ | `PinballWorld` `PinballConfig` `LadderPLC` `Rung` `build_ladder` `Sequencer` `SeqState` `Jig` `Kicker` |
| `tetris_vla/pinball_agents.py` | 4 層の結線と各層の実装、盤面描画 | `run_stack` `StackConfig` `ScriptedPaddle` `MockVLAPaddle` `SmolVLAPaddle` `HeuristicStrategist` `VLMStrategist` `render_pinball` |
| `tetris_vla/pinballviz.py` | 走行記録 → 自己完結 HTML / SVG | `pinball_html` `pinball_svg` `ladder_diagram` `takt_chart` `st_sequencer_code` |
| `tetris_vla/engine.py` | テトリス題材のコア。SRS / 7-bag / 重力 / lock | `Engine` `Piece` |
| `tetris_vla/render.py` | 盤面 ⇄ PNG の厳密な相互変換 | `render` `decode` `to_png` |
| `tetris_vla/agents.py` | 探索ソルバ、ペルソナ、VLA バックエンド | `HeuristicAgent` `VLAAgent` |
| `tetris_vla/runtime.py` | 論理 tick クロック、単一サーバ推論待ち行列 | `Runtime` |
| `tetris_vla/smolvla_pilot.py` | 学習モデルの構築と behavior cloning | `build_policy` `train_bc` `BCSample` `pick_device` |
| `tetris_vla/parachute.py` / `flock.py` | 別題材 (降下制御 / 群れ) | 各 `*_world` |
| `scripts/build_site.py` | ドキュメントと成果物を静的サイトに組む | `main` `md2html` `check_links` |

## ピンボール題材の構造

制御は下から上へ 4 層で、`run_stack` が別々の周期で回す。

    ラダー (毎 tick)      LadderPLC.scan        決定的。ゲート開閉と安全インタロック
    シーケンサ (毎 tick)  Sequencer.step        故障検知 → 修理段取り → 復帰
    VLA (25 tick)         paddle_ctl.plan       パドル 1 自由度。1 回で 25 手ぶん
    VLM (150 tick)        strategist.__call__   狙う工程と修理可否の助言

差し替えは `run_stack(world, paddle_ctl, strategist, cfg)` の引数で行う。
必要なインタフェースは 2 つだけ。

```python
class PaddleController:                 # VLA の位置
    def plan(self, world, target_seg: int | None) -> tuple[list[float], float, dict]:
        """returns (行動列, レイテンシ秒, 記録用の情報)"""

class Strategist:                       # VLM の位置
    def __call__(self, world) -> tuple[dict, float, str, str, str | None]:
        """returns ({"target_seg": int, "repair_ok": bool, ...}, レイテンシ秒,
                    プロンプト, 生出力, エラー)"""
```

`ScriptedPaddle` (解析解) と `HeuristicStrategist` はモデル無しで動くので、
差し替え先の参照実装として読むのが早い。

### 時間の扱い

tick は論理時間で、実時間とは結びつけない。`StackConfig.slowmo` が
「推論の実時間を何 tick ぶんの陳腐化として扱うか」の換算率で、
`force_drift_ticks` を使えばモデル無しで陳腐化だけを注入できる。

工程のタクト (球が戻る周期) は実測 61 tick。助言がこれ以内なら遅れは無害で、
超えると均等性から崩れる。

### 決定性

シミュレータ・ラダー・シーケンサ・解析解パドル・mock は、同じ入力なら必ず同じ出力になる。
`test_stack_is_deterministic` と `test_ladder_is_deterministic` が固定している。

**学習モデルを挟むと再現しない。** 同じ seed・同じ盤面で 2 回走らせて 691.7 / 668.8、
推論の種を固定しても 925.0 / 802.1 で揃わない。サンプリングだけでなく演算自体に由来する。
比較するときは 1 回の走行ではなく seed を増やした分布で見ること。

### 隠れ状態

`Jig` は `durability` / `clearance` / `wear_per_hit` を持つが、
`Jig.telemetry()` が返すのは `hits` / `fit_ng` / `broken` / `repairing` だけ。
**制御層へ渡してよいのは telemetry の中身に限る。**
答え合わせ用の真値は走行結果の `jig_truth` にだけ入る。

## 別プロジェクトへ持っていく単位

| 持っていくもの | 必要なファイル | 依存 |
|---|---|---|
| ラダーの実行意味論だけ | `pinball.py` の `Rung` / `LadderPLC` / `build_ladder` | なし (標準ライブラリのみ) |
| ラダー図と ST 生成 | 上記 + `pinballviz.py` の `ladder_diagram` / `st_ladder_code` / `st_sequencer_code` | なし |
| シミュレータ一式 | `pinball.py` + `pinball_agents.py` | `numpy` `Pillow` |
| 可視化まで | 上記 + `pinballviz.py` | 同上 |
| 学習まで | 上記 + `smolvla_pilot.py` | `torch` `lerobot` |

`Rung` / `LadderPLC` は他に何も参照していないので、ファイルから切り出して
そのまま持っていける。ST 生成 (`st_sequencer_code`) は `Sequencer.step` の
構文木を読むので、ステートマシンの書き方を
`if self.state is XxxState.NAME:` の形に揃えれば別の実装にも当たる。

## 走らせ方

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q                      # モデル不要。全部通る
.venv/bin/python scripts/pinball_demo.py --strategy heuristic --out results/pinball
.venv/bin/python scripts/pinball_matrix.py --seeds 10     # 制御層の差し替え比較
python3 scripts/build_site.py _site                       # 標準ライブラリのみ
```

`pinball_demo.py --strategy vlm` と `--paddle smolvla` だけがモデルを要求する。
それ以外はモデル無しで完結する。

## テストの読み方

`tests/test_pinball.py` は仕様書として書いてある。
挙動を変えたときにどれが落ちるかで、何を壊したかが分かる。

| テスト | 固定している仕様 |
|---|---|
| `test_plc_evaluates_rungs_in_order` | ラングの評価順が意味を持つこと |
| `test_durability_is_hidden_from_telemetry` | 隠れ状態が制御層へ漏れないこと |
| `test_the_task_is_a_usable_instrument` | 題材の変動係数が小さいまま保たれること |
| `test_st_is_generated_from_the_implementation_not_hardcoded` | ST が実装から導出され続けること |
| `test_kicker_only_catches_balls_through_the_saucer` | 吸収射出が腕前を代替しないこと |

---

## rtsdk/ — 型で干渉を消す側 (Rust)

`rtsdk/` は Python 本体とは別系統の PoC で、依存も走らせ方も独立している
(`cd rtsdk && cargo test`)。ピンボール題材で測った「層ごとに別のクロックが同居する」
話を、測定ではなく**型**の側に置けるかを見るためのもの。

| クレート | 中身 | 依存 |
|---|---|---|
| `pulse-trace` | 事象の型と固定長レコーダ (`no_std`, 確保なし) | なし |
| `vtime` | 仮想時間モナド。Δt 注入と WCET の型レベル加算 | `pulse-trace` |
| `zoneguard` | 型状態のアーム、ゾーンごとに 1 枚の排他トークン | 上記 |
| `legacy-bridge` | C++ 資産の `unsafe` 包み + 決定的な離散事象スケジューラ | 上記 + `cc` |
| `sim-harness` | 空間セマンティクス DSL、シミュレータ、TDD ランナー | 上記 |
| `ui-pulse` | パルスグラフ (egui/Wasm)。重ね合わせ・因果・タイムトラベル | `pulse-trace` + `egui` |

移植の単位としては `pulse-trace` / `vtime` / `zoneguard` の 3 つが `no_std` で完結していて、
そのまま他のプロジェクトへ持っていける。Python 側の制御をこの基板へ移す段取りは
[rtsdk/README.md](../rtsdk/README.md) の最後の節にある。
