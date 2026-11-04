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


class ArticleTableTests(unittest.TestCase):
    def test_empty_question_selection_keeps_tag_columns(self):
        questions = pd.DataFrame(columns=["問題", "選択肢1", "選択肢2", "選択肢3", "選択肢4", "正答"])

        tagged = analysis.tag_questions(questions, {"疾患・病態の軸": {}, "重要語(疾患・所見)": {}})

        self.assertIn("関連軸", tagged.columns)
        self.assertIn("正答に含まれる重要語", tagged.columns)
        self.assertTrue(tagged.empty)

    def test_missing_major_questions_leave_shares_unavailable(self):
        questions = pd.DataFrame(columns=["大項目", "中項目", "回"])

        summary, topics, trend, examples = analysis.article_tables(
            questions, pd.DataFrame(), "整形外科", "骨折", ["A"], 25, 26,
        )

        self.assertTrue(pd.isna(summary.loc[0, "大項目内割合(%)"]))
        self.assertTrue(pd.isna(topics.loc[0, "大項目内割合(%)"]))
        self.assertTrue(trend["大項目内割合(%)"].isna().all())
        self.assertTrue(examples.empty)

    def test_summarizes_denominators_recurrence_trend_and_latest_examples(self):
        questions = pd.DataFrame([
            {"大項目": "骨折", "中項目": "A", "回": 25},
            {"大項目": "骨折", "中項目": "他", "回": 25},
            {"大項目": "骨折", "中項目": "A", "回": 26},
            {"大項目": "骨折", "中項目": "他", "回": 27},
        ])
        tagged = pd.DataFrame([
            {
                "中項目": "A", "問題番号": 1, "問題ID": "第25回No.1", "回": 25,
                "問題": "古い問題", "選択肢1": "a", "選択肢2": "b",
                "選択肢3": "c", "選択肢4": "d", "模範解答": "1", "正答": "a",
                "関連軸": "", "重要語": "", "正答に含まれる重要語": "", "関連疾患": "",
            },
            {
                "中項目": "A", "問題番号": 2, "問題ID": "第26回No.2", "回": 26,
                "問題": "新しい問題", "選択肢1": "e", "選択肢2": "f",
                "選択肢3": "g", "選択肢4": "h", "模範解答": "1", "正答": "e",
                "関連軸": "", "重要語": "", "正答に含まれる重要語": "", "関連疾患": "",
            },
        ])

        summary, topics, trend, examples = analysis.article_tables(
            questions, tagged, "整形外科", "骨折", ["A", "B"], 25, 27,
        )

        self.assertEqual(summary.loc[0, "対象問題数"], 2)
        self.assertEqual(summary.loc[0, "大項目の問題数"], 4)
        self.assertEqual(summary.loc[0, "大項目内割合(%)"], 50.0)
        self.assertEqual(summary.loc[0, "対象が出題された回数"], 2)
        self.assertEqual(topics["対象期間出題数"].tolist(), [2, 0])
        self.assertEqual(topics["出題回数"].tolist(), [2, 0])
        self.assertEqual(trend["大項目内割合(%)"].tolist(), [50.0, 100.0, 0.0])
        self.assertEqual(examples["問題"].tolist(), ["新しい問題"])


if __name__ == "__main__":
    unittest.main()
