import unittest

import pandas as pd

import analysis
import article


class ArticleGenerationTests(unittest.TestCase):
    def setUp(self):
        questions = pd.DataFrame([
            {"大項目": "骨折", "中項目": "A", "回": 25},
            {"大項目": "骨折", "中項目": "B", "回": 25},
            {"大項目": "骨折", "中項目": "A", "回": 26},
            {"大項目": "骨折", "中項目": "他", "回": 28},
        ])
        tagged = pd.DataFrame([
            {
                "中項目": "A", "問題番号": 1, "問題ID": "第25回No.1", "回": 25,
                "問題": "原文は転載しない", "選択肢1": "選択肢は転載しない",
                "選択肢2": "b", "選択肢3": "c", "選択肢4": "d",
                "模範解答": "1", "正答": "選択肢は転載しない",
                "関連軸": "", "重要語": "橈骨", "正答に含まれる重要語": "", "関連疾患": "",
            },
        ])
        summary, topics, trend, examples = analysis.article_tables(
            questions, tagged, "整形外科", "骨折", ["A", "B"], 25, 28
        )
        self.result = {
            "記事向け要約": summary,
            "中項目別の出題状況": topics,
            "対象中項目の年度推移": trend,
            "記事向け問題例": examples,
            "中項目別の年度推移": analysis.article_topic_trend(
                questions, "骨折", ["A", "B"], 25, 28
            ),
            "中項目別の頻出語": pd.DataFrame([
                {"中項目": "A", "順位": 1, "語": "橈骨", "出現問題数": 2, "対象問題数": 2, "問題ID": "第25回No.1"},
            ]),
        }

    def test_article_uses_data_and_user_inputs_without_reproducing_question(self):
        text = article.generate_article(
            self.result, mea="教育分析を支援します。", next_theme="脱臼",
            mock_results="模試Aは20問中10問正答。",
        )
        self.assertIn("A：2問（大項目全体の50%", text)
        self.assertIn("出題 第25回、第26回／出題なし 第28回", text)
        self.assertIn("| 第27回 | 0 | 集計不可 | 集計不可 |", text)
        self.assertIn("2/2問（100.0%）、1/2問（50.0%）", text)
        self.assertIn("第25回No.1／模範解答番号：1", text)
        self.assertIn("- A：「橈骨」2問", text)
        self.assertIn("- B：条件を満たす語なし", text)
        self.assertNotIn("原文は転載しない", text)
        self.assertNotIn("選択肢は転載しない", text)
        self.assertIn("教育分析を支援します。", text)
        self.assertIn("模試Aは20問中10問正答。", text)
        self.assertIn("脱臼", text)
        self.assertTrue(3000 <= len(text) <= 4000, len(text))

    def test_empty_major_raises_instead_of_inventing_statistics(self):
        self.result["記事向け要約"].loc[0, "大項目の問題数"] = 0
        with self.assertRaisesRegex(ValueError, "問題がない"):
            article.generate_article(self.result)

    def test_missing_inputs_are_visible_and_one_year_is_not_compared(self):
        self.result["対象中項目の年度推移"] = self.result["対象中項目の年度推移"].iloc[:1]
        text = article.generate_article(self.result)
        self.assertIn("期間を前後に分けた比較は控えます", text)
        self.assertIn("［MEAの機能・提供内容を入力してください］", text)
        self.assertIn("［次回の分析テーマを入力してください］", text)


if __name__ == "__main__":
    unittest.main()
