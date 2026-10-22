import unittest

from server.pipeline.backends.base import AsrOutputError, context_terms
from server.pipeline.backends.funasr import _adapt_result


class FunAsrBackendTest(unittest.TestCase):
    def test_context_terms_are_deduplicated_and_keep_order(self):
        self.assertEqual(
            context_terms("设计域, SDK；设计域\nSocket"),
            ["设计域", "SDK", "Socket"],
        )

    def test_adapter_filters_punctuation_timestamps_and_adds_offset(self):
        text, items = _adapt_result({
            "text": "你好，SDK!",
            "timestamps": [
                {"token": "你", "start_time": 0.1, "end_time": 0.2},
                {"token": "好", "start_time": 0.2, "end_time": 0.3},
                {"token": "，", "start_time": 0.3, "end_time": 0.4},
                {"token": "SDK", "start_time": 0.4, "end_time": 0.8},
                {"token": "!", "start_time": 0.8, "end_time": 0.9},
            ],
        }, 30.0, 1.0)

        self.assertEqual(text, "你好，SDK!")
        self.assertEqual([item["t"] for item in items], ["你", "好", "SDK"])
        self.assertEqual(items[-1], {"t": "SDK", "s": 30.4, "e": 30.8})

    def test_adapter_requires_timestamps_for_text(self):
        with self.assertRaisesRegex(AsrOutputError, "without timestamps"):
            _adapt_result({"text": "测试"}, 0.0, 30.0)


if __name__ == "__main__":
    unittest.main()
