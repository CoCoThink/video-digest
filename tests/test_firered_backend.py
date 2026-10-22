import tempfile
from array import array
from pathlib import Path
import unittest
import wave

from server.pipeline import asr
from server.pipeline.backends.base import AsrOutputError, AsrUnstableError
from server.pipeline.backends.firered import (
    FireRedBackend,
    _iter_wav_chunks,
    _suspicious_result,
    _transcribe_with_retry,
    _validate_contract,
)
from server.pipeline.punctuation import (
    PunctuationProjectionError,
    partition_text_by_time,
    project_punctuation,
)


def _make_wav(path: Path, frames: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\0\0" * frames)


def _make_amplitude_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(array("h", samples).tobytes())


class FireRedBackendTest(unittest.TestCase):
    def test_punctuation_projection_preserves_original_case_and_words(self):
        self.assertEqual(
            project_punctuation("你好AI Model", "你好，ai model。"),
            "你好，AI Model。",
        )

    def test_punctuation_projection_rejects_word_changes(self):
        for value in ("您好，AI。", "你好，API。", "你好，AI's。"):
            with self.subTest(value=value), self.assertRaises(PunctuationProjectionError):
                project_punctuation("你好AI", value)

    def test_punctuation_projection_accepts_word_internal_raw_apostrophe(self):
        self.assertEqual(project_punctuation("I'm ok", "i'm ok!"), "I'm ok!")

    def test_partition_text_uses_timestamps_and_preserves_boundary_space(self):
        items = [
            {"t": "你", "s": 0.1, "e": 0.2},
            {"t": "好", "s": 0.2, "e": 0.3},
            {"t": "AI", "s": 30.1, "e": 30.5},
        ]
        self.assertEqual(
            partition_text_by_time("你好 AI", items, 30),
            [(0, "你好"), (30, " AI")],
        )

    def test_factory_rejects_unknown_backend_without_loading_models(self):
        with self.assertRaisesRegex(ValueError, "unsupported ASR backend"):
            asr.create_backend({"asr_backend": "other"})

    def test_chunk_iterator_has_exact_offsets_and_removes_consumed_files(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            source = directory / "source.wav"
            _make_wav(source, 32_321)
            seen = []
            iterator = _iter_wav_chunks(source, directory, 1.0)
            for path, offset, duration in iterator:
                self.assertTrue(path.exists())
                seen.append((path, offset, duration))
            self.assertEqual([x[1] for x in seen], [0.0, 1.0, 2.0])
            self.assertAlmostEqual(sum(x[2] for x in seen), 32_321 / 16_000)
            self.assertTrue(all(not x[0].exists() for x in seen))

    def test_chunk_iterator_prefers_quiet_boundary_without_exceeding_limit(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            source = directory / "source.wav"
            samples = [1000] * (16_000 * 2 + 8_000)
            samples[11_200:12_800] = [0] * 1_600  # 0.7s-0.8s quiet region
            _make_amplitude_wav(source, samples)

            seen = list(_iter_wav_chunks(
                source, directory, 1.0, boundary_search_sec=0.4,
            ))

            self.assertGreaterEqual(seen[0][2], 0.7)
            self.assertLessEqual(seen[0][2], 0.81)
            self.assertTrue(all(duration <= 1.0 for _, _, duration in seen))
            self.assertAlmostEqual(sum(duration for _, _, duration in seen), 2.5)
            for previous, current in zip(seen, seen[1:]):
                self.assertAlmostEqual(previous[1] + previous[2], current[1])

    def test_invalid_timestamps_trigger_retry_guard(self):
        result = {"text": "测试", "timestamp": [["测", 0.1, 0.2], ["试", 0.3, 31.0]]}
        self.assertEqual(_suspicious_result(result, 30), "timestamp_out_of_range")

    def test_minimum_chunk_raises_typed_unstable_error(self):
        class FakeModel:
            def transcribe(self, _, __):
                return [{"text": "异常", "timestamp": []}]

        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            source = directory / "source.wav"
            _make_wav(source, 16_000)
            with self.assertRaises(AsrUnstableError):
                _transcribe_with_retry(
                    FakeModel(), source, 0, 1, directory / "retry", 1,
                )

    def test_contract_accepts_punctuation_but_rejects_changed_words(self):
        valid = {
            "text": "你好，AI!",
            "items": [
                {"t": "你", "s": 0.1, "e": 0.2},
                {"t": "好", "s": 0.2, "e": 0.3},
                {"t": "AI", "s": 0.4, "e": 0.7},
            ],
        }
        _validate_contract(valid, 1.0)
        invalid = {**valid, "text": "您好，AI!"}
        with self.assertRaises(AsrOutputError):
            _validate_contract(invalid, 1.0)

    def test_contract_rejects_non_finite_timestamp(self):
        with self.assertRaisesRegex(AsrOutputError, "invalid absolute timestamps"):
            _validate_contract({
                "text": "好",
                "items": [{"t": "好", "s": float("nan"), "e": 0.2}],
            }, 1.0)

    def test_forced_language_is_rejected_instead_of_being_faked(self):
        backend = FireRedBackend.__new__(FireRedBackend)
        with self.assertRaisesRegex(ValueError, "does not support forced language"):
            backend.transcribe("unused.wav", language="English")


if __name__ == "__main__":
    unittest.main()
