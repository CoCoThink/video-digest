#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕排版规范化（由 build_srt_layout.py 逐行等价迁移，详细设计 §4.3 步骤 3）。

输入 items.json（带标点全文 + 逐 token 时间戳），输出规范化 cue 列表。在
标点感知断句之上强制专业字幕可读性规则：最短时长、阅读速度上限(CPS)、两行
折行、合并碎片、绝不劈开拉丁/公式 token（w_k / Q1 / RNN）与 jieba 中文词、
消重叠。这些规则不针对任何特定视频，仅随语言约定切换（R5）。
"""
import unicodedata

try:
    import jieba
    jieba.setLogLevel(60)  # 静音一次性词典加载横幅
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

STRONG = set("。！？!?…")
SOFT = set("，,、；;：:")
PUNCT = STRONG | SOFT | set("\"'“”‘’()（）《》—-·… ")


def is_latin(ch: str) -> bool:
    """属于不可劈开的公式/词 token 的字符：ASCII 字母、数字、下划线。"""
    return ch.isascii() and (ch.isalnum() or ch == "_")


def visible_len(text: str) -> int:
    return sum(1 for ch in text if ch.strip() and ch not in PUNCT)


def _is_aligner_char(ch: str) -> bool:
    """Match Qwen3ForceAlignProcessor.clean_token's character filter."""
    return ch == "'" or unicodedata.category(ch)[:1] in {"L", "N"}


def word_starts(text: str):
    """jieba 词起始字符下标集合（安全断点）。无 jieba 时返回 None（无约束）。"""
    if not _HAS_JIEBA or not text:
        return None
    starts = set()
    idx = 0
    for tok in jieba.cut(text, HMM=True):
        starts.add(idx)
        idx += len(tok)
    starts.add(len(text))
    return starts


# ---------------------------------------------------------------------------
# Step 0：把带标点全文的每个字符对齐到无标点的定时 token 上。
# ---------------------------------------------------------------------------
def align_chars(data: dict):
    text = data["text"]
    items = data["items"]
    aligned = [
        (ch, item["s"], item["e"])
        for item in items
        for ch in item["t"]
    ]
    source = "".join(ch for ch in text if _is_aligner_char(ch))
    target = "".join(ch for ch, _, _ in aligned)
    if source != target:
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(source, target)) if left != right),
            min(len(source), len(target)),
        )
        raise ValueError(
            "ASR text and forced-alignment tokens differ at normalized character "
            f"{mismatch} (text={len(source)}, aligned={len(target)})"
        )

    timed = []  # (char, start|None, end|None)
    aligned_index = 0
    for ch in text:
        if not _is_aligner_char(ch):
            timed.append((ch, None, None))
            continue
        _, start, end = aligned[aligned_index]
        timed.append((ch, start, end))
        aligned_index += 1
    return timed


# ---------------------------------------------------------------------------
# Step 1：在标点处断句，遵守间隔/长度限制，但绝不在拉丁/数字 run 内断（规则 5）。
# ---------------------------------------------------------------------------
def segment(timed, wstarts=None, soft_min=12, hard_max=34, max_dur=7.0, max_gap=0.7):
    cues = []
    buf = []

    def at_word_boundary(pos):
        return wstarts is None or pos in wstarts

    def bounds():
        ss = [s for _, s, e in buf if s is not None]
        ee = [e for _, s, e in buf if e is not None]
        return (min(ss) if ss else None, max(ee) if ee else None)

    def flush():
        nonlocal buf
        s, e = bounds()
        txt = "".join(ch for ch, _, _ in buf).strip()
        txt = txt.strip("".join(SOFT) + " ")
        if txt and s is not None and e is not None:
            cues.append([s, e, txt])
        buf = []

    prev_e = None
    n = len(timed)
    for idx in range(n):
        ch, s, e = timed[idx]
        nxt = timed[idx + 1][0] if idx + 1 < n else ""
        prev_ch = timed[idx - 1][0] if idx > 0 else ""
        in_run = is_latin(ch) and is_latin(nxt)          # 断在 ch 后会劈开 token
        split_here = is_latin(prev_ch) and is_latin(ch)  # 断在 ch 前会劈开 token
        next_is_punct = nxt in STRONG or nxt in SOFT
        # 间隔断点落在 ch 前（位置 idx），仅允许在词边界
        if (s is not None and prev_e is not None and (s - prev_e) > max_gap
                and visible_len("".join(c for c, _, _ in buf)) > 0
                and not split_here and at_word_boundary(idx)):
            flush()
        buf.append((ch, s, e))
        if e is not None:
            prev_e = e
        bs, be = bounds()
        dur = (be - bs) if (bs is not None and be is not None) else 0
        vlen = visible_len("".join(c for c, _, _ in buf))
        if ch in STRONG:
            flush()
        elif ch in SOFT and vlen >= soft_min:
            flush()
        elif ((vlen >= hard_max or dur >= max_dur) and not next_is_punct
              and not in_run and at_word_boundary(idx + 1)):
            # 长度/时长断点落在 ch 后（位置 idx+1）
            flush()
    flush()
    return cues


# ---------------------------------------------------------------------------
# Step 2：把碎片合并进前一条 cue（规则 4）。
# ---------------------------------------------------------------------------
def merge_short(cues, merge_below=4, cap=22, max_gap=0.5):
    out = []
    for c in cues:
        vis = visible_len(c[2])
        if out and vis <= merge_below:
            prev = out[-1]
            if visible_len(prev[2]) + vis <= cap and (c[0] - prev[1]) <= max_gap:
                prev[1] = c[1]
                prev[2] = prev[2] + c[2]
                continue
        out.append(c[:])
    return out


# ---------------------------------------------------------------------------
# Step 3：为最短时长（规则 1）与阅读速度（规则 2）延长时长，仅延进下一条之前
# 的空档；修正重叠（规则 6）。
# ---------------------------------------------------------------------------
def retime(cues, min_dur=1.0, cps=9.0, max_dur=7.0, min_gap=0.04):
    # Two adjacent phrases can share a point timestamp when the aligner has no
    # duration evidence for the first one. They cannot be displayed as separate,
    # non-overlapping cues, so fold the earlier text into the later span.
    merged = []
    for cue in cues:
        cue = cue[:]
        if merged and cue[0] - merged[-1][0] <= min_gap:
            previous = merged.pop()
            cue[0] = min(previous[0], cue[0])
            cue[1] = max(previous[1], cue[1])
            cue[2] = previous[2] + cue[2]
        merged.append(cue)
    cues[:] = merged

    too_fast = []
    clipped = 0
    for i, c in enumerate(cues):
        chars = visible_len(c[2])
        need = max(min_dur, chars / cps)
        desired = c[0] + need
        if i + 1 < len(cues):
            limit = cues[i + 1][0] - min_gap
            c[1] = max(c[1], min(desired, limit))
            if c[1] > cues[i + 1][0] - min_gap:      # 消除任何重叠
                c[1] = cues[i + 1][0] - min_gap
            if c[1] <= c[0]:
                c[1] = min(c[0] + 0.2, cues[i + 1][0] - 0.01)
        else:
            c[1] = max(c[1], desired)
        # 收束滞留 cue（通常是对齐空洞留下的超大内部间隔）。
        if c[1] - c[0] > max_dur:
            c[1] = c[0] + max_dur
            clipped += 1
        actual_cps = chars / (c[1] - c[0]) if c[1] > c[0] else 999
        if actual_cps > cps + 0.5:
            too_fast.append((round(actual_cps, 1), c[2]))
    return cues, too_fast, clipped


# ---------------------------------------------------------------------------
# Step 4：把长 cue 折成至多两行（规则 3），在最靠近中点的软标点/空格/拉丁↔CJK
# 边界处断。
# ---------------------------------------------------------------------------
def wrap(text: str, line_width=18) -> str:
    if len(text) <= line_width:
        return text
    mid = len(text) / 2
    wstarts = word_starts(text)
    best_i, best_cost = None, None
    for i in range(1, len(text)):
        prev, cur = text[i - 1], text[i]
        if is_latin(prev) and is_latin(cur):
            continue                                    # 绝不在拉丁 token 内（规则 5）
        if wstarts is not None and i not in wstarts:
            continue                                    # 绝不在中文词内（jieba）
        if prev in STRONG or prev in SOFT:
            penalty = 0                                 # 第 1 档：标点后
        elif cur == " " or prev == " " or (is_latin(prev) != is_latin(cur)):
            penalty = 3                                 # 第 2 档：空格/脚本边界
        else:
            penalty = 8                                 # 第 3 档：任意词边界
        cost = abs(i - mid) + penalty
        if best_cost is None or cost < best_cost:
            best_cost, best_i = cost, i
    if best_i is None:
        best_i = int(round(mid))
    a = text[:best_i].rstrip()
    b = text[best_i:].strip()
    return a + "\n" + b if b else a


def build_cues(data: dict, cfg: dict):
    """items.json → 规范化 cue 列表 [[start, end, text], ...]（text 未折行）。

    参数默认取 build_srt_layout.py 的 CLI 默认值，保证与实测参照物一致。
    """
    timed = align_chars(data)
    wstarts = word_starts(data["text"])
    cues = segment(
        timed, wstarts=wstarts,
        soft_min=cfg.get("layout_soft_min", 12),
        hard_max=cfg.get("layout_hard_max", 34),
        max_dur=cfg.get("layout_max_dur", 7.0),
    )
    cues = merge_short(cues, merge_below=cfg.get("layout_merge_below", 4))
    cues, too_fast, clipped = retime(
        cues,
        min_dur=cfg.get("layout_min_dur", 1.0),
        cps=cfg.get("layout_cps", 9.0),
        max_dur=cfg.get("layout_max_dur", 7.0),
    )
    return cues, {"too_fast": len(too_fast), "clipped": clipped}
