"""Unit tests for the text pipeline and the classifier."""

import unittest

from spam_detector.classifier import classify_text, transform_text


class TransformTextTests(unittest.TestCase):
    def test_removes_stopwords_and_stems(self):
        cleaned = transform_text("The prizes are being claimed").split()
        self.assertIn("prize", cleaned)
        self.assertNotIn("the", cleaned)

    def test_drops_punctuation(self):
        cleaned = transform_text("Free!!! entry??? txt WIN now...")
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)


class ClassifyTextTests(unittest.TestCase):
    def test_lottery_message_is_spam(self):
        result = classify_text(
            "URGENT! You have won a 100,000 Prize Jackpot. Txt CLAIM to 81010 now!"
        )
        self.assertTrue(result["is_spam"])
        self.assertEqual(result["label"], "Spam")
        self.assertGreater(result["spam_probability"], 50)

    def test_normal_message_is_ham(self):
        result = classify_text("Hey Sarah, are we still meeting for dinner at 7:30?")
        self.assertFalse(result["is_spam"])
        self.assertEqual(result["label"], "Not Spam")

    def test_empty_input(self):
        result = classify_text("   ")
        self.assertFalse(result["is_spam"])
        self.assertEqual(result["spam_probability"], 0.0)
        self.assertEqual(result["confidence"], 100.0)

    def test_probabilities_add_up(self):
        result = classify_text("Free ringtone, reply YES to 44222")
        total = result["spam_probability"] + result["ham_probability"]
        self.assertAlmostEqual(total, 100.0, delta=0.05)

    def test_spam_keywords_are_reported(self):
        result = classify_text("Congratulations, you won a free prize. Claim now!")
        keywords = [word for word, _ in result["flagged_keywords"]]
        self.assertTrue(keywords)


if __name__ == "__main__":
    unittest.main()
