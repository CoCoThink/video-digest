import unittest

from server.pipeline.layout import align_chars, build_cues, retime


class LayoutAlignmentTest(unittest.TestCase):
    def test_qwen_filtered_punctuation_does_not_desynchronize_timestamps(self):
        data = {
            "text": "这个 A.I. / Agent，继续。",
            "items": [
                {"t": "这", "s": 0.0, "e": 0.1},
                {"t": "个", "s": 0.1, "e": 0.2},
                {"t": "AI", "s": 0.3, "e": 0.5},
                {"t": "Agent", "s": 0.6, "e": 1.0},
                {"t": "继", "s": 1.1, "e": 1.2},
                {"t": "续", "s": 1.2, "e": 1.3},
            ],
        }

        timed = align_chars(data)

        self.assertEqual("".join(ch for ch, start, _ in timed if start is not None),
                         "这个AIAgent继续")
        self.assertEqual(next(start for ch, start, _ in timed if ch == "继"), 1.1)
        cues, _ = build_cues(data, {})
        cue_text = "".join(cue[2] for cue in cues)
        self.assertIn("Agent", cue_text)
        self.assertTrue(cue_text.endswith("继续。"))
        self.assertGreaterEqual(cues[-1][1], 1.3)

    def test_apostrophe_kept_by_qwen_aligner_receives_timestamp(self):
        data = {
            "text": "don't stop.",
            "items": [{"t": "don't", "s": 1.0, "e": 1.5},
                      {"t": "stop", "s": 1.6, "e": 2.0}],
        }

        timed = align_chars(data)

        apostrophe = timed[data["text"].index("'")]
        self.assertEqual(apostrophe, ("'", 1.0, 1.5))
        self.assertEqual(timed[data["text"].index("s")][1], 1.6)

    def test_text_token_contract_mismatch_fails_instead_of_dropping_subtitles(self):
        data = {
            "text": "甲乙丙",
            "items": [{"t": "甲丁丙", "s": 0.0, "e": 1.0}],
        }

        with self.assertRaisesRegex(ValueError, "normalized character 1"):
            align_chars(data)

    def test_retime_merges_cues_that_share_the_same_start(self):
        cues = [[10.0, 10.0, "它？"], [10.0, 12.0, "后一句"]]

        retimed, too_fast, clipped = retime(cues)

        self.assertEqual(retimed, [[10.0, 12.0, "它？后一句"]])
        self.assertEqual(too_fast, [])
        self.assertEqual(clipped, 0)


if __name__ == "__main__":
    unittest.main()
