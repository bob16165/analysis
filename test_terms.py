import unittest

import pandas as pd

import terms


class TopicFrequentTermsTests(unittest.TestCase):
    def test_counts_questions_in_each_topic_and_orders_ties(self):
        questions = pd.DataFrame([
            {"中項目": "A", "問題ID": "q1", "問題": "骨折 骨折", "選択肢1": "神経",
             "選択肢2": "関節", "選択肢3": "", "選択肢4": "", "模範解答": "1"},
            {"中項目": "A", "問題ID": "q2", "問題": "骨折", "選択肢1": "神経",
             "選択肢2": "関節", "選択肢3": "", "選択肢4": "", "模範解答": "1"},
            {"中項目": "A", "問題ID": "q3", "問題": "骨折", "選択肢1": "筋肉",
             "選択肢2": "関節", "選択肢3": "", "選択肢4": "", "模範解答": "1"},
            {"中項目": "B", "問題ID": "q4", "問題": "神経", "選択肢1": "神経",
             "選択肢2": "", "選択肢3": "", "選択肢4": "", "模範解答": "1"},
        ])
        result = terms.topic_frequent_terms(
            questions, ["A", "B", "C"], ["骨折", "神経", "関節"],
            set(), set(),
        )
        a = result[result["中項目"] == "A"]
        self.assertEqual(a["語"].tolist(), ["関節", "骨折", "神経"])
        self.assertEqual(a["出現問題数"].tolist(), [3, 3, 2])
        self.assertEqual(a["対象問題数"].tolist(), [3, 3, 3])
        self.assertEqual(a.iloc[2]["問題ID"], "q1、q2")
        self.assertNotIn("B", result["中項目"].tolist())
        self.assertNotIn("C", result["中項目"].tolist())

    def test_stop_words_and_empty_questions(self):
        questions = pd.DataFrame([
            {"中項目": "A", "問題ID": f"q{i}", "問題": "骨折",
             "選択肢1": "", "選択肢2": "", "選択肢3": "", "選択肢4": "", "模範解答": "1"}
            for i in range(2)
        ])
        self.assertTrue(terms.topic_frequent_terms(
            questions, ["A"], ["骨折"], {"骨折"}, set()
        ).empty)
        self.assertEqual(
            terms.topic_frequent_terms(questions.iloc[:0], ["A"], [], set(), set()).columns.tolist(),
            ["中項目", "順位", "語", "出現問題数", "対象問題数", "問題ID"],
        )


if __name__ == "__main__":
    unittest.main()
