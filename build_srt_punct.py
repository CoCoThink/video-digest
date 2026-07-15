#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a natural, punctuation-aware SRT from Qwen3-ASR output.

The forced aligner yields per-token timings WITHOUT punctuation, while the ASR
full text HAS punctuation. We align the punctuated text onto the timed tokens
(the tokens are a punctuation-free subsequence of the text), then break subtitle
cues at sentence/clause punctuation, honoring max chars and duration.
"""
import argparse
import json
import re

STRONG = set("。！？!?…")
SOFT = set("，,、；;：:")
PUNCT = STRONG | SOFT | set("\"'“”‘’()（）《》—-· ")


def fmt(sec):
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build(data, max_chars=28, max_dur=6.0, min_dur=0.8, max_gap=0.7):
    text = data["text"]
    items = data["items"]
    # Attach a timing to each character of the full text by consuming timed tokens
    # in order. Token text may be multi-char; match char-by-char, skipping punctuation.
    timed = []  # (char, start, end) for non-punct chars; punct carries None timing
    ti = 0            # item index
    ci_in_tok = 0     # char index inside current token's text
    for ch in text:
        if ch.strip() == "" or ch in PUNCT:
            timed.append((ch, None, None))
            continue
        # advance to a token whose current char matches ch
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
                # token/text mismatch (rare); skip this token char
                ci_in_tok += 1
                scan += 1
        if not matched:
            timed.append((ch, None, None))

    # Now group into cues, breaking at punctuation / gaps / limits.
    cues = []
    buf = []           # list of (char, s, e)
    def cur_bounds():
        ss = [s for _, s, e in buf if s is not None]
        ee = [e for _, s, e in buf if e is not None]
        return (min(ss) if ss else None, max(ee) if ee else None)

    def visible_len(b):
        return sum(1 for ch, _, _ in b if ch.strip() and ch not in PUNCT)

    def flush():
        nonlocal buf
        s, e = cur_bounds()
        txt = "".join(ch for ch, _, _ in buf).strip()
        # trim leading/trailing punctuation-only artifacts but keep inner punctuation
        txt = txt.strip("".join(SOFT) + " ")
        if txt and s is not None and e is not None:
            cues.append([s, e, txt])
        buf = []

    prev_e = None
    n = len(timed)
    for idx in range(n):
        ch, s, e = timed[idx]
        nxt = timed[idx + 1][0] if idx + 1 < n else ""
        # gap-based break before adding
        if s is not None and prev_e is not None and (s - prev_e) > max_gap and visible_len(buf) > 0:
            flush()
        buf.append((ch, s, e))
        if e is not None:
            prev_e = e
        bs, be = cur_bounds()
        dur = (be - bs) if (bs is not None and be is not None) else 0
        vlen = visible_len(buf)
        # Don't break if the very next char is punctuation — keep it attached.
        next_is_punct = nxt in STRONG or nxt in SOFT
        if ch in STRONG:
            flush()
        elif ch in SOFT and vlen >= max_chars * 0.6:
            flush()
        elif (vlen >= max_chars or dur >= max_dur) and not next_is_punct:
            flush()
    flush()

    # enforce a minimum display duration and non-overlap
    for i, c in enumerate(cues):
        if c[1] - c[0] < min_dur:
            c[1] = c[0] + min_dur
        if i + 1 < len(cues) and c[1] > cues[i + 1][0]:
            c[1] = max(c[0] + min_dur * 0.5, cues[i + 1][0] - 0.02)
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=28)
    args = ap.parse_args()
    data = json.load(open(args.items, encoding="utf-8"))
    cues = build(data, max_chars=args.max_chars)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")
    print(f"wrote {len(cues)} cues -> {args.out}")


if __name__ == "__main__":
    main()
