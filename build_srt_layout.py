#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General, parameterized subtitle-layout builder for Qwen3-ASR output.

Same input as build_srt_punct.py (items.<tag>.json = punctuated full text +
per-token timings). On top of punctuation-aware segmentation it enforces the
standard readability rules used by professional subtitles — none of which are
tuned to any particular video; the only knobs are language conventions:

  (1) min display duration        --min-dur
  (2) reading-speed cap (CPS)      --cps       (chars per second)
  (3) line width + 2-line wrap     --line-width
  (4) merge tiny fragments         --merge-below
  (5) never break inside a         (always on) latin/digit/underscore run,
      formula/word token                       e.g. w_k, Q1, K2, RNN
  (6) no overlap / gap sanity      (always on)

Durations are extended only into the available gap before the next cue, so we
never shift timing off the audio; a cue that is still too fast afterwards is
reported (its underlying alignment span is genuinely that short).
"""
import argparse
import json

try:
    import jieba
    jieba.setLogLevel(60)  # silence the one-time dict-load banner
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

STRONG = set("。！？!?…")
SOFT = set("，,、；;：:")
PUNCT = STRONG | SOFT | set("\"'“”‘’()（）《》—-·… ")


def fmt(sec):
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def is_latin(ch):
    """Char that belongs to a formula/word token we must not split: ASCII
    letters, digits, and underscore (covers w_k, Q1, K2, RNN, QKV, Softmax)."""
    return ch.isascii() and (ch.isalnum() or ch == "_")


def visible_len(text):
    return sum(1 for ch in text if ch.strip() and ch not in PUNCT)


def word_starts(text):
    """Set of char indices at which a jieba word begins (i.e. safe break points).
    A break between text[i-1] and text[i] splits a word iff i is NOT in this set.
    Without jieba, every position is a boundary (falls back to old behavior)."""
    if not _HAS_JIEBA or not text:
        return None  # None => "no word constraint"
    starts = set()
    idx = 0
    for tok in jieba.cut(text, HMM=True):
        starts.add(idx)
        idx += len(tok)
    starts.add(len(text))
    return starts


# ---------------------------------------------------------------------------
# Step 0: attach a timing to every char of the punctuated full text by
# consuming the timed (punctuation-free) tokens in order.
# ---------------------------------------------------------------------------
def align_chars(data):
    text = data["text"]
    items = data["items"]
    timed = []          # (char, start|None, end|None)
    ti = 0
    ci = 0
    for ch in text:
        if ch.strip() == "" or ch in PUNCT:
            timed.append((ch, None, None))
            continue
        matched = False
        scan = 0
        while ti < len(items) and scan < 3:
            tok = items[ti]["t"]
            if ci >= len(tok):
                ti += 1
                ci = 0
                continue
            if tok[ci] == ch:
                timed.append((ch, items[ti]["s"], items[ti]["e"]))
                ci += 1
                matched = True
                break
            ci += 1
            scan += 1
        if not matched:
            timed.append((ch, None, None))
    return timed


# ---------------------------------------------------------------------------
# Step 1: segment into cues at punctuation, honoring gaps / length limits, but
# NEVER breaking inside a latin/digit run (rule 5).
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
        in_run = is_latin(ch) and is_latin(nxt)          # break after ch would split a token
        split_here = is_latin(prev_ch) and is_latin(ch)  # break before ch would split a token
        next_is_punct = nxt in STRONG or nxt in SOFT
        # gap break falls *before* ch (position idx); allow only at a word boundary
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
            # length/duration break falls *after* ch (position idx+1)
            flush()
    flush()
    return cues


# ---------------------------------------------------------------------------
# Step 2: merge tiny fragments into the previous cue (rule 4).
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
# Step 3: extend durations for min-dur (rule 1) and reading speed (rule 2),
# only into the free gap before the next cue; fix overlaps (rule 6).
# ---------------------------------------------------------------------------
def retime(cues, min_dur=1.0, cps=9.0, max_dur=7.0, min_gap=0.04):
    too_fast = []
    clipped = 0
    for i, c in enumerate(cues):
        chars = visible_len(c[2])
        need = max(min_dur, chars / cps)
        desired = c[0] + need
        if i + 1 < len(cues):
            limit = cues[i + 1][0] - min_gap
            c[1] = max(c[1], min(desired, limit))
            if c[1] > cues[i + 1][0] - min_gap:      # kill any overlap
                c[1] = cues[i + 1][0] - min_gap
            if c[1] <= c[0]:
                c[1] = min(c[0] + 0.2, cues[i + 1][0] - 0.01)
        else:
            c[1] = max(c[1], desired)
        # Cap lingering cues (usually an alignment hole left a huge internal gap).
        if c[1] - c[0] > max_dur:
            c[1] = c[0] + max_dur
            clipped += 1
        actual_cps = chars / (c[1] - c[0]) if c[1] > c[0] else 999
        if actual_cps > cps + 0.5:
            too_fast.append((round(actual_cps, 1), c[2]))
    return cues, too_fast, clipped


# ---------------------------------------------------------------------------
# Step 4: wrap a long cue into at most two lines (rule 3), breaking at a soft
# punctuation / space / latin<->CJK boundary nearest the middle.
# ---------------------------------------------------------------------------
def wrap(text, line_width=18):
    """Break into at most two balanced lines. Prefer a punctuation boundary,
    then a latin<->CJK boundary or space, then anywhere — each weighted by how
    far it sits from the middle so we never split an English phrase or a
    formula when a nearby clause break exists."""
    if len(text) <= line_width:
        return text
    mid = len(text) / 2
    wstarts = word_starts(text)
    best_i, best_cost = None, None
    for i in range(1, len(text)):
        prev, cur = text[i - 1], text[i]
        if is_latin(prev) and is_latin(cur):
            continue                                    # never inside a latin token (rule 5)
        if wstarts is not None and i not in wstarts:
            continue                                    # never inside a Chinese word (jieba)
        if prev in STRONG or prev in SOFT:
            penalty = 0                                 # tier 1: after punctuation
        elif cur == " " or prev == " " or (is_latin(prev) != is_latin(cur)):
            penalty = 3                                 # tier 2: space / script boundary
        else:
            penalty = 8                                 # tier 3: any word boundary
        cost = abs(i - mid) + penalty
        if best_cost is None or cost < best_cost:
            best_cost, best_i = cost, i
    if best_i is None:                                  # no boundary found; fall back to middle
        best_i = int(round(mid))
    a = text[:best_i].rstrip()
    b = text[best_i:].strip()
    return a + "\n" + b if b else a


def stats(cues):
    durs = [e - s for s, e, _ in cues]
    lens = [visible_len(t) for _, _, t in cues]
    cps = [(l / d if d > 0 else 999) for l, d in zip(lens, durs)]
    return {
        "n": len(cues),
        "dur_min": round(min(durs), 2), "dur_mean": round(sum(durs) / len(durs), 2),
        "len_max": max(lens), "len_mean": round(sum(lens) / len(lens), 1),
        "under_1s": sum(1 for d in durs if d < 1.0),
        "over_cps9": sum(1 for c in cps if c > 9.0),
        "cps_max": round(max(cps), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--cps", type=float, default=9.0, help="max chars/sec (Chinese ~9)")
    ap.add_argument("--max-dur", type=float, default=7.0, help="clip cues lingering longer than this")
    ap.add_argument("--line-width", type=int, default=18)
    ap.add_argument("--merge-below", type=int, default=4, help="merge cues with <= this many visible chars")
    ap.add_argument("--soft-min", type=int, default=12)
    ap.add_argument("--hard-max", type=int, default=34)
    args = ap.parse_args()

    data = json.load(open(args.items, encoding="utf-8"))
    timed = align_chars(data)
    wstarts = word_starts(data["text"])
    cues = segment(timed, wstarts=wstarts, soft_min=args.soft_min, hard_max=args.hard_max)
    before = stats(cues)
    cues = merge_short(cues, merge_below=args.merge_below)
    cues, too_fast, clipped = retime(cues, min_dur=args.min_dur, cps=args.cps, max_dur=args.max_dur)
    after = stats(cues)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{wrap(t, args.line_width)}\n\n")

    print(f"wrote {len(cues)} cues -> {args.out}")
    print("            n  dur_min dur_mean len_max len_mean <1s  >9cps cps_max")
    for name, st in (("before", before), ("after ", after)):
        print(f"  {name}  {st['n']:>3} {st['dur_min']:>7} {st['dur_mean']:>8} "
              f"{st['len_max']:>7} {st['len_mean']:>8} {st['under_1s']:>4} "
              f"{st['over_cps9']:>5} {st['cps_max']:>7}")
    if clipped:
        print(f"\n  clipped {clipped} lingering cue(s) to --max-dur {args.max_dur}s "
              f"(usually an alignment hole in a formula-dense span)")
    if too_fast:
        print(f"\n  still faster than {args.cps} cps after extending into gaps "
              f"({len(too_fast)} cues, alignment span genuinely short):")
        for c, t in sorted(too_fast, reverse=True)[:6]:
            print(f"    {c:>5} cps  {t}")


if __name__ == "__main__":
    main()
