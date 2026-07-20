"""FireRedPunc adapter with a fail-safe projection onto original ASR text."""
from __future__ import annotations

from pathlib import Path
import unicodedata


class PunctuationProjectionError(ValueError):
    """Punctuation output changed semantic ASR characters."""


def _semantic_char(ch: str) -> bool:
    # Apostrophes are meaningful inside English words and must not be inserted or removed.
    return ch == "'" or unicodedata.category(ch)[:1] in {"L", "N"}


def _semantic_key(ch: str) -> str:
    return unicodedata.normalize("NFKC", ch).casefold()


def project_punctuation(raw: str, punctuated: str) -> str:
    """Keep raw letters/digits/case and take only separators from Punc output.

    A per-character normalized comparison is intentionally stricter than an edit-distance
    alignment. If the model changes, removes or adds a semantic character, callers must
    fail open to ``raw`` instead of risking text/timestamp desynchronization.
    """
    raw_chars = [ch for ch in raw if _semantic_char(ch)]
    punc_chars = [ch for ch in punctuated if _semantic_char(ch)]
    if (len(raw_chars) != len(punc_chars)
            or [_semantic_key(ch) for ch in raw_chars]
            != [_semantic_key(ch) for ch in punc_chars]):
        raise PunctuationProjectionError(
            f"punctuation changed semantic text (raw={len(raw_chars)}, "
            f"punctuated={len(punc_chars)})"
        )

    projected = []
    raw_index = 0
    for ch in punctuated:
        if _semantic_char(ch):
            projected.append(raw_chars[raw_index])
            raw_index += 1
        elif unicodedata.category(ch)[:1] in {"P", "S", "Z"} or ch.isspace():
            projected.append(ch)
        else:
            raise PunctuationProjectionError(
                f"punctuation output contains unsupported character U+{ord(ch):04X}"
            )
    return "".join(projected)


def partition_text_by_time(text: str, items: list[dict], chunk_sec: float) -> list[tuple[float, str]]:
    """Recover display-text chunks from timed items without losing spaces/punctuation."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    grouped: dict[int, list[str]] = {}
    for item in items:
        index = max(0, int(float(item["s"]) // chunk_sec))
        grouped.setdefault(index, []).append(str(item["t"]))

    parts: list[list] = []
    cursor = 0
    for index, tokens in grouped.items():
        expected = [ch for token in tokens for ch in token if _semantic_char(ch)]
        actual = []
        start = cursor
        while cursor < len(text) and len(actual) < len(expected):
            ch = text[cursor]
            cursor += 1
            if _semantic_char(ch):
                actual.append(ch)
        if (len(actual) != len(expected)
                or [_semantic_key(ch) for ch in actual]
                != [_semantic_key(ch) for ch in expected]):
            raise PunctuationProjectionError(
                f"text/items differ while partitioning chunk at {index * chunk_sec:g}s"
            )
        parts.append([index * chunk_sec, text[start:cursor]])

    tail = text[cursor:]
    if any(_semantic_char(ch) for ch in tail):
        raise PunctuationProjectionError("unmatched semantic text remains after final chunk")
    if parts:
        parts[-1][1] += tail
    elif text:
        raise PunctuationProjectionError("text exists without timed items")
    return [(offset, value) for offset, value in parts]


class FireRedPunctuation:
    model_id = "FireRedPunc"

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        model_path = Path(model_path).expanduser()
        required = (
            model_path / "model.pth.tar",
            model_path / "chinese-bert-wwm-ext_vocab.txt",
            model_path / "out_dict",
            model_path / "chinese-lert-base",
        )
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise ValueError(f"FireRedPunc model directory is incomplete: {', '.join(missing)}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("VD_FIRERED_PUNC_DEVICE must be cpu or cuda")

        try:
            from fireredpunc.punc import FireRedPunc, FireRedPuncConfig
        except ImportError:
            try:
                from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig
            except ImportError as exc:
                raise RuntimeError("FireRedPunc source is unavailable") from exc

        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for FireRedPunc")
        config = FireRedPuncConfig(use_gpu=device == "cuda")
        self._model = FireRedPunc.from_pretrained(str(model_path), config)

    def process(self, raw: str) -> str:
        if not raw:
            return raw
        result = self._model.process([raw])[0]
        if result.get("origin_text") != raw:
            raise PunctuationProjectionError("FireRedPunc origin_text differs from input")
        return project_punctuation(raw, result.get("punc_text", ""))
