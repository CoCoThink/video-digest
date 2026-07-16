#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标点感知断句（由 build_srt_punct.py 去 CLI 化重构）。

注意：主转写路径使用 layout.py（其 segment() 已自带断句 + 排版一体化，且是
实测参照物 *.layout.srt 的生成器）。本模块保留为更简单的独立断句实现，供
P1 分块级进度或备用路径调用，不在默认后处理链上。
"""
STRONG = set("。！？!?…")
SOFT = set("，,、；;：:")
PUNCT = STRONG | SOFT | set("\"'“”‘’()（）《》—-· ")


def build(data: dict, max_chars=28, max_dur=6.0, min_dur=0.8, max_gap=0.7):
    text = data["text"]
    items = data["items"]
    timed = []  # (char, start, end)；标点 char 带 None 时间
    ti = 0
    ci_in_tok = 0
    for ch in text:
        if ch.strip() == "" or ch in PUNCT:
            timed.append((ch, None, None))
            continue
        matched = False
        scan = 0
        while ti < len(items) and scan < 3:
            tok = items[ti]["t"]
            if ci_in_tok >= len(tok):
                ti += 1
                ci_in_tok = 0
                continue
            if tok[ci_in_tok] == ch:
                timed.append((ch, items[ti]["s"], items[ti]["e"]))
                ci_in_tok += 1
                matched = True
                break
            else:
                ci_in_tok += 1
                scan += 1
        if not matched:
            timed.append((ch, None, None))

    cues = []
    buf = []

    def cur_bounds():
        ss = [s for _, s, e in buf if s is not None]
        ee = [e for _, s, e in buf if e is not None]
        return (min(ss) if ss else None, max(ee) if ee else None)

    def vlen(b):
        return sum(1 for ch, _, _ in b if ch.strip() and ch not in PUNCT)

    def flush():
        nonlocal buf
        s, e = cur_bounds()
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
        if s is not None and prev_e is not None and (s - prev_e) > max_gap and vlen(buf) > 0:
            flush()
        buf.append((ch, s, e))
        if e is not None:
            prev_e = e
        bs, be = cur_bounds()
        dur = (be - bs) if (bs is not None and be is not None) else 0
        v = vlen(buf)
        next_is_punct = nxt in STRONG or nxt in SOFT
        if ch in STRONG:
            flush()
        elif ch in SOFT and v >= max_chars * 0.6:
            flush()
        elif (v >= max_chars or dur >= max_dur) and not next_is_punct:
            flush()
    flush()

    for i, c in enumerate(cues):
        if c[1] - c[0] < min_dur:
            c[1] = c[0] + min_dur
        if i + 1 < len(cues) and c[1] > cues[i + 1][0]:
            c[1] = max(c[0] + min_dur * 0.5, cues[i + 1][0] - 0.02)
    return cues
