# エージェント作業宣言 (co-working note)

このリポジトリでは **複数の Claude Code セッションが同時に作業している**。
作業を始める前にこのファイルを読むこと。

---

## ⚠ 最優先: origin へ push できない (人間の対応が必要)

**GitHub の認証情報が切れている。** そのため `main` はローカルに積み上がる一方で、
`origin/main` に反映されていない。

```
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

- `gh auth status` → `The token in default is invalid.`
- `credential.helper` は `osxkeychain` だが、有効な資格情報が入っていない
- 読み取り (`git ls-remote`) は public repo なので通る。**push だけが通らない**

セッション A が 11:13 の commit を最後に止まっているのは、レートリミットではなく
**これが原因の可能性が高い**。エージェントは認証情報を入力できないので、
人間が次を実行する必要がある:

```
gh auth login -h github.com
```

その後 `git push origin main`。

---

## セッション B — ビジュアル層の強化 (進行中)

**担当範囲: 描画層のみ。ロジック・エージェント・学習には触らない。**

### 所有ファイル

| ファイル | 状態 |
|---|---|
| `tetris_vla/pinballviz.py` | **編集中** |

### 完了したもの (commit `f6d8ad6` / `a953537` に含まれている)

`build_ladder()` の 6 ラングは、以前は HTML の表に条件式のテキストとして出ているだけで、
**ラダー図そのものが描かれていなかった**。以下を追加済み:

1. `ladder_diagram()` — 母線 / `─┤├─` / `─┤/├─` / `─( )─` を描く本物のラダー図
2. 通電状態のオーバーレイ — `ladder_trace` のビットで ON の経路を緑に光らせる
   (`trace_bits_at()` / `find_scan()` で「見どころ」のスキャンを自動選択)
3. `layer_stack_diagram()` — 4 層の発火タイミングを実寸で刻んだ図

### 作業中 (未コミット)

4. `pinball_svg()` の強化 — 列レーン、凡例、静的表示時の座標欠落修正
5. シーケンサのフロー図 / ラダーの全体像 / VLM コンテキストとの紐付け

既存の `ladder_timing_chart()` と HTML の表は **消さずに残す**
(表は `comment` 列を持ち、図では表現できない情報があるため)。

### git について

このセッションは **commit / push を自分から行わない**。

> セッション A へ: commit する場合は `git add -A` ではなく
> **パス指定で自分の担当ファイルだけを stage** してほしい。
> 11:11 の `f6d8ad6` では、こちらの編集中の `pinballviz.py` と
> この AGENT-NOTES.md が巻き込まれて 1 コミットに入った。

---

## セッション A — ピンボール / VLA 本体

観測できた範囲。正確なところは当該セッションが上書きしてほしい。

- `tetris_vla/pinball_agents.py` の VLM 画像縮尺 `px_per_unit` (8GB 機のメモリ対策)
- `results/pinball_vlm.*` を生成。実機 qwen で 374 秒/回、スコア 950
- 11:13 の `a953537` を最後に停止 (上記の push 認証切れが原因と推測)
