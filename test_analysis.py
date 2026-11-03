import unittest

import pandas as pd

import analysis


class CrossSubjectAnalysisTests(unittest.TestCase):
    def test_summarizes_term_matches_by_subject_and_period(self):
        question_sheets = {
            "解剖学": pd.DataFrame(),
            "リハ医学": pd.DataFrame([
                {"回": 1, "問題ID": "第1回No.1", "問題": "橈骨神経について", "選択肢1": "正しい"},
                {"回": 28, "問題ID": "第28回No.1", "問題": "神経の働き", "選択肢1": "別の内容"},
            ]),
            "一般臨床": pd.DataFrame([
                {"回": 28, "問題ID": "第28回No.2", "問題": "患者の症状", "選択肢1": "橈骨神経"},
            ]),
        }
        extracted_terms = pd.DataFrame({"語": ["橈骨神経"]})

        summary, yearly, periods, matches = analysis.cross_subject_analysis(
            question_sheets, "解剖学", extracted_terms,
            subjects=["リハ医学", "一般臨床", "解剖学"],
        )

        rehab = summary.set_index("科目").loc["リハ医学"]
        self.assertEqual(rehab["対象問題数(全回)"], 2)
        self.assertEqual(rehab["一致問題数(全回)"], 1)
        self.assertEqual(rehab["一致率(%)"], 50.0)
        self.assertNotIn("解剖学", summary["科目"].tolist())
        self.assertEqual(matches["問題ID"].tolist(), ["第1回No.1", "第28回No.2"])
        self.assertEqual(
            periods[periods["科目"] == "リハ医学"]["一致問題数"].tolist(),
            [1, 0],
        )
        self.assertEqual(
            yearly[(yearly["科目"] == "一般臨床") & (yearly["回"] == 28)]["一致率(%)"].iloc[0],
            100.0,
        )

    def test_empty_extracted_terms_returns_zero_match_rates(self):
        question_sheets = {
            "運動学": pd.DataFrame([
                {"回": 1, "問題ID": "第1回No.1", "問題": "問題文", "選択肢1": "選択肢"},
            ])
        }

        summary, _, _, matches = analysis.cross_subject_analysis(
            question_sheets, "解剖学", pd.DataFrame(columns=["語"]),
            subjects=["運動学"],
        )

        self.assertEqual(summary["一致問題数(全回)"].tolist(), [0])
        self.assertEqual(summary["一致率(%)"].tolist(), [0.0])
        self.assertTrue(matches.empty)


if __name__ == "__main__":
    unittest.main()
