import tempfile
from pathlib import Path
import unittest
import wave

from fireredasr2_to_srt import (
    _join_text,
    _split_wav,
    _suspicious_result,
    _transcribe_with_retry,
)


class FireRedAsr2HelpersTest(unittest.TestCase):
    def test_join_text_only_inserts_space_between_latin_chunks(self):
        self.assertEqual(_join_text(["hello", "world", "中文", "测试"]), "hello world中文测试")

    def test_split_wav_has_exact_offsets_and_no_lost_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "source.wav"
            frame_count = 16_000 * 2 + 321
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(b"\0\0" * frame_count)

            chunks = _split_wav(source, directory, 1.0)

            self.assertEqual(len(chunks), 3)
            self.assertEqual([offset for _, offset, _ in chunks], [0.0, 1.0, 2.0])
            self.assertAlmostEqual(sum(duration for _, _, duration in chunks), frame_count / 16_000)
            with wave.open(str(chunks[-1][0]), "rb") as reader:
                self.assertEqual(reader.getnframes(), 321)

    def test_suspicious_result_detects_runaway_decoder(self):
        timestamps = [["嗯", i * 0.04, (i + 1) * 0.04] for i in range(750)]
        result = {"text": "嗯" * 750, "timestamp": timestamps}
        self.assertEqual(_suspicious_result(result, 30), "implausible_token_rate")

    def test_suspicious_result_accepts_normal_alignment(self):
        timestamps = [["字", i * 0.2, i * 0.2 + 0.12 + (i % 3) * 0.04]
                      for i in range(120)]
        result = {"text": "字" * 120, "timestamp": timestamps}
        self.assertIsNone(_suspicious_result(result, 30))

    def test_suspicious_result_detects_uniform_fallback(self):
        timestamps = [["字", i * 0.1, (i + 1) * 0.1] for i in range(100)]
        result = {"text": "字" * 100, "timestamp": timestamps}
        self.assertEqual(
            _suspicious_result(result, 30), "uniform_fallback_timestamps")

    def test_recursive_retry_uses_distinct_wav_directories(self):
        class FakeModel:
            def transcribe(self, _, paths):
                with wave.open(paths[0], "rb") as reader:
                    duration = reader.getnframes() / reader.getframerate()
                if duration > 0.75:
                    timestamps = [["嗯", i * duration / 101, (i + 1) * duration / 101]
                                  for i in range(101)]
                    return [{"text": "嗯" * 101, "timestamp": timestamps}]
                return [{"text": "好", "timestamp": [["好", 0.1, duration - 0.1]]}]

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16_000)
                writer.writeframes(b"\0\0" * 32_000)

            leaves, events = _transcribe_with_retry(
                FakeModel(), source, 10.0, 2.0, directory / "retry",
                min_chunk_sec=0.5,
            )

            self.assertEqual(len(leaves), 4)
            self.assertEqual([round(x[0], 1) for x in leaves], [10.0, 10.5, 11.0, 11.5])
            self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
