import unittest
from unittest.mock import patch

import pandas as pd

import article


class ArticleGenerationTests(unittest.TestCase):
    def setUp(self):
        years = [f"第{year}回" for year in range(1, 35)]
        rows = []
        for topic, count in (("A", 5), ("B", 4), ("C", 3), ("D", 2), ("記載なし", 99)):
            row = {"大項目": "骨折", "中項目": topic, **{year: 0 for year in years}}
            row["第1回"] = count
            rows.append(row)
        self.frequency = pd.DataFrame(rows)
        self.years = years
        questions = []
        for year in range(1, 35):
            questions.append(self.question(year, 1, "別", "他", "別の問題"))
        questions += [
            self.question(1, 2, "骨折", "A", "橈骨骨折を問う"),
            self.question(10, 2, "骨折", "A", "橈骨骨折を問う"),
            self.question(19, 2, "骨折", "A", "橈骨骨折を問う"),
            self.question(19, 3, "骨折", "B", "選択する"),
            self.question(28, 2, "骨折", "B", "選択する"),
            self.question(28, 3, "骨折", "C", "選択する"),
            self.question(28, 4, "骨折", "記載なし", "未対応"),
        ]
        self.questions = pd.DataFrame(questions)
        self.details = pd.DataFrame([
            {"問題ID": "第1回/整形外科/No.2", "Taxonomy": "Ⅰ型", "Taxonomy状態": "人間確認済み"},
            {"問題ID": "第10回/整形外科/No.2", "Taxonomy": "判定保留", "Taxonomy状態": "人間確認済み"},
            {"問題ID": "第19回/整形外科/No.2", "Taxonomy": "Ⅱ型", "Taxonomy状態": "ルール候補"},
        ])
        self.relations = pd.DataFrame([
            {"question_id_a": "第1回/整形外科/No.2", "question_id_b": "第10回/整形外科/No.2",
             "active_a": 1, "active_b": 1, "subject_a": "整形外科", "subject_b": "整形外科",
             "round_a": 1, "round_b": 10, "review_status": "確認済み", "reviewed_type": "A", "auto_type": "A"},
            {"question_id_a": "第10回/整形外科/No.2", "question_id_b": "第19回/整形外科/No.2",
             "active_a": 1, "active_b": 1, "subject_a": "整形外科", "subject_b": "整形外科",
             "round_a": 10, "round_b": 19, "review_status": "自動候補", "reviewed_type": None, "auto_type": "A"},
            {"question_id_a": "第1回/整形外科/No.2", "question_id_b": "第19回/整形外科/No.2",
             "active_a": 1, "active_b": 1, "subject_a": "整形外科", "subject_b": "整形外科",
             "round_a": 1, "round_b": 19, "review_status": "対象外", "reviewed_type": "対象外", "auto_type": "A"},
            {"question_id_a": "第1回/整形外科/No.2", "question_id_b": "第1回/整形外科/No.3",
             "active_a": 1, "active_b": 1, "subject_a": "整形外科", "subject_b": "整形外科",
             "round_a": 1, "round_b": 1, "review_status": "確認済み", "reviewed_type": "A", "auto_type": "A"},
        ])

    @staticmethod
    def question(year, number, major, topic, stem):
        return {
            "回": year, "問題番号": number, "問題ID": f"第{year}回No.{number}",
            "大項目": major, "中項目": topic, "問題": stem,
            "選択肢1": "正しい", "選択肢2": "誤り", "選択肢3": "別", "選択肢4": "他",
            "模範解答": "1", "正答": "正しい",
        }

    def build(self, details=None, relations=None):
        with patch("article.analysis.load_freq", return_value=(self.frequency, self.years)), \
                patch("article.analysis.load_questions", return_value=self.questions), \
                patch("article.analysis.load_dictionary", return_value={
                    "除外語": [], "重要語(疾患・所見)": {"橈骨骨折": ""}
                }), \
                patch("article.analysis.load_medical_terms", return_value=[]):
            return article.build_article_data(
                "freq", "past", "整形外科", "骨折", details, relations
            )

    def test_scoped_periods_rankings_terms_and_classification(self):
        data = self.build(self.details, self.relations)
        self.assertEqual(data["順位"]["中項目"].tolist(), ["A", "B", "C"])
        periods = data["期間別"]
        a_periods = periods[periods["中項目"].eq("A")]
        self.assertEqual(a_periods["期間"].tolist(),
                         ["第1〜9回", "第10〜18回", "第19〜27回", "第28〜34回"])
        self.assertEqual(a_periods["問題数"].tolist(), [1, 1, 1, 0])
        row = periods[(periods["期間"] == "第28〜34回") & (periods["中項目"] == "B")].iloc[0]
        self.assertEqual((row["問題数"], row["分母"], row["割合(%)"], row["中項目未対応"]),
                         (1, 2, 50.0, 1))
        taxonomy = data["Taxonomy"]
        a = taxonomy[taxonomy["中項目"] == "A"].set_index("期間")
        self.assertEqual(a.loc["第1〜9回", "Ⅰ型"], 1)
        self.assertEqual(a.loc["第10〜18回", "判定保留"], 1)
        self.assertEqual(a.loc["第19〜27回", "未確認"], 1)
        text = article.generate_article(data, selected_terms={"A": ["橈骨骨折"]})
        self.assertIn("「橈骨骨折」：3問", text)
        self.assertIn("問題文：第1回No.2", text)
        self.assertIn("第1回No.2", text)
        self.assertIn("第19回No.2", text)
        self.assertIn("人間確認済み：第1回/整形外科/No.2", text)
        self.assertIn("未確認候補：第10回/整形外科/No.2", text)
        self.assertNotIn("第1回/整形外科/No.3", text)
        self.assertNotIn("対象外：", text)
        self.assertIn("B：前期から現行期へ+0.0ポイント", text)

    def test_missing_rounds_and_unavailable_reviews_are_explicit(self):
        self.questions = self.questions[self.questions["回"] != 2]
        data = self.build()
        row = data["期間別"].iloc[0]
        self.assertEqual(row["未収録回"], [2])
        self.assertTrue(pd.isna(row["割合(%)"]))
        text = article.generate_article(data)
        self.assertIn("未収録：第2回", text)
        self.assertIn("Taxonomy：現在の過去問に同期済みの分類を確認できません", text)
        self.assertIn("同期済みの再出題レビューを確認できません", text)

    def test_incomplete_frequency_table_is_not_treated_as_full_ranking(self):
        self.years.remove("第34回")
        with self.assertRaisesRegex(ValueError, "第34回"):
            self.build()

    def test_selected_topics_without_questions_remain_zero(self):
        self.questions = self.questions[
            ~self.questions["中項目"].isin(["A", "B", "C"])
        ]
        data = self.build()
        self.assertTrue(data["期間別"]["問題数"].eq(0).all())
        self.assertNotIn("頻出語候補", article.generate_article(data))

    def test_only_selected_candidates_are_included_and_invalid_selections_fail(self):
        data = self.build()
        original = data["頻出語"].iloc[0].copy()
        candidates = []
        for rank, word in enumerate(["骨折", "血管", "関節", "神経", "筋肉"], 1):
            row = original.copy()
            row["順位"], row["語"] = rank, word
            candidates.append(row)
        data["頻出語"] = pd.DataFrame(candidates)
        text = article.generate_article(data, selected_terms={"A": ["筋肉", "血管"]})
        self.assertIn("「筋肉」", text)
        self.assertIn("「血管」", text)
        self.assertNotIn("「骨折」：", text)
        self.assertNotIn("「神経」", text)
        self.assertNotIn("頻出語候補", article.generate_article(data))
        for selection in ({"B": ["筋肉"]}, {"対象外": ["筋肉"]},
                          {"A": ["存在しない語"]}, {"A": ["筋肉", "筋肉"]},
                          {"A": ["骨折", "血管", "関節", "神経"]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                article.generate_article(data, selected_terms=selection)

    def test_only_unassigned_questions_do_not_create_trend_results(self):
        self.questions = self.questions[
            (self.questions["大項目"] == "骨折") & (self.questions["中項目"] == "記載なし")
        ]
        data = self.build()
        self.assertIsNone(data["トレンド"])
        self.assertIn("分類済みの分析対象データがないため、判定できません", article.generate_article(data))


if __name__ == "__main__":
    unittest.main()
