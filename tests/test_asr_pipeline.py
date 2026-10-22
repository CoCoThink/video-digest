import threading
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import qwen_asr.inference.qwen3_asr as qa

from server.pipeline.asr import _Progress, _transcribe_pipelined


@dataclass(frozen=True)
class Token:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class AlignResult:
    items: list


class FakeAligner:
    def __init__(self, started=None, release=None, fail=False):
        self.started = started
        self.release = release
        self.fail = fail
        self.calls = []

    def align(self, audio, text, language):
        chunk = audio[0][0]
        self.calls.append((chunk, text[0], language[0]))
        if self.started is not None and chunk == "chunk-0":
            self.started.set()
            if not self.release.wait(timeout=2):
                raise AssertionError("ASR did not overlap the pending alignment")
        if self.fail:
            raise RuntimeError("alignment failed")
        return [AlignResult([Token(text[0], 0.1, 0.5)])]


class FakeModel:
    backend = "transformers"

    def __init__(self, outputs, aligner, started=None, release=None):
        self.outputs = dict(outputs)
        self.forced_aligner = aligner
        self.started = started
        self.release = release
        self.asr_calls = []

    def _infer_asr(self, contexts, wavs, languages):
        chunk = wavs[0]
        self.asr_calls.append((contexts[0], chunk, languages[0]))
        if chunk == "chunk-1" and self.started is not None:
            if not self.started.wait(timeout=2):
                raise AssertionError("alignment did not start before the next ASR chunk")
            self.release.set()
        return [self.outputs[chunk]]

    @staticmethod
    def _offset_align_result(result, offset):
        return AlignResult([
            Token(item.text, item.start_time + offset, item.end_time + offset)
            for item in result.items
        ])

    @staticmethod
    def _merge_align_results(results):
        if not results:
            return None
        return AlignResult([item for result in results for item in result.items])


class PipelineTest(unittest.TestCase):
    def run_pipeline(
        self, outputs, offsets, model, language=None, context="", progress=None,
    ):
        chunks = [(f"chunk-{i}", offset) for i, offset in enumerate(offsets)]

        def parse(raw, user_language=None):
            return "Chinese", outputs[raw]

        with (
            patch.object(qa, "normalize_audios", return_value=["whole-audio"]),
            patch.object(qa, "split_audio_into_chunks", return_value=chunks),
            patch.object(qa, "parse_asr_output", side_effect=parse),
            patch.object(qa, "merge_languages", return_value="Chinese"),
        ):
            return _transcribe_pipelined(
                model, "audio.wav", language, context, progress,
            )

    def test_overlaps_alignment_and_preserves_chunk_order_and_offsets(self):
        started = threading.Event()
        release = threading.Event()
        raw_to_text = {"raw-0": "甲", "raw-1": "乙", "raw-2": "丙"}
        model = FakeModel(
            {"chunk-0": "raw-0", "chunk-1": "raw-1", "chunk-2": "raw-2"},
            FakeAligner(started, release), started, release,
        )
        percentages = []

        result = self.run_pipeline(
            raw_to_text, [0.0, 30.0, 60.0], model,
            progress=_Progress(percentages.append),
        )

        self.assertEqual(result.text, "甲乙丙")
        self.assertEqual(result.language, "Chinese")
        self.assertEqual(
            [(item.text, item.start_time, item.end_time) for item in result.time_stamps.items],
            [("甲", 0.1, 0.5), ("乙", 30.1, 30.5), ("丙", 60.1, 60.5)],
        )
        self.assertEqual([call[0] for call in model.forced_aligner.calls],
                         ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual([call[0] for call in model.asr_calls], ["", "", ""])
        self.assertEqual(percentages, [16, 33, 50, 66, 83, 99])

    def test_forwards_context_to_every_qwen_chunk(self):
        raw_to_text = {"raw-0": "甲", "raw-1": "乙"}
        model = FakeModel(
            {"chunk-0": "raw-0", "chunk-1": "raw-1"}, FakeAligner(),
        )

        self.run_pipeline(
            raw_to_text, [0.0, 30.0], model, context="设计域, SDK",
        )

        self.assertEqual(
            [call[0] for call in model.asr_calls],
            ["设计域, SDK", "设计域, SDK"],
        )

    def test_skips_silent_chunk_without_inventing_alignment_progress(self):
        raw_to_text = {"raw-0": "甲", "raw-1": ""}
        aligner = FakeAligner()
        model = FakeModel(
            {"chunk-0": "raw-0", "chunk-1": "raw-1"}, aligner
        )
        percentages = []

        result = self.run_pipeline(
            raw_to_text, [0.0, 30.0], model,
            progress=_Progress(percentages.append),
        )

        self.assertEqual(result.text, "甲")
        self.assertEqual(len(aligner.calls), 1)
        self.assertEqual(percentages, [25, 50, 75])

    def test_propagates_alignment_failure(self):
        raw_to_text = {"raw-0": "甲", "raw-1": "乙"}
        model = FakeModel(
            {"chunk-0": "raw-0", "chunk-1": "raw-1"},
            FakeAligner(fail=True),
        )

        with self.assertRaisesRegex(RuntimeError, "alignment failed"):
            self.run_pipeline(raw_to_text, [0.0, 30.0], model)


if __name__ == "__main__":
    unittest.main()
