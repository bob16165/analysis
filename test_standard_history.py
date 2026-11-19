import unittest

import pandas as pd

import standard_history


def question(year, number, major, middle):
    return {
        "回": year,
        "問題番号": number,
        "大項目": major,
        "中項目": middle,
        "問題": "設問",
    }


class StandardHistoryTests(unittest.TestCase):
    def setUp(self):
        self.frequency_items = pd.DataFrame([
            {"大項目": "骨折", "中項目": "骨幹部"},
            {"大項目": "骨折", "中項目": "未出題項目"},
            {"大項目": "旧分類", "中項目": "記載なし"},
        ])
        self.questions = pd.DataFrame([
            question(1, 1, "骨折", "骨幹部"),
            question(1, 2, "旧分類", "記載なし"),
            question(2, 1, "骨折", "骨幹部"),
            question(2, 2, "未分類", None),
        ])

    def analyze(self, expected_rounds=(1, 2), boundaries=(1, 3, 5)):
        return standard_history.analyze_standard_history(
            self.questions,
            self.frequency_items,
            "整形外科",
            list(expected_rounds),
            1,
            max(expected_rounds),
            recent_count=2,
            frequency_bin_boundaries=boundaries,
        )

    def test_current_and_former_categories_are_reported_with_observed_denominators(self):
        result = self.analyze()
        summary = result["出題基準区分別集計"].set_index("出題基準区分")
        items = result["項目別出題実績"].set_index(["大項目", "中項目"])

        self.assertEqual(summary.loc["現行出題基準", "確認できた項目数"], 2)
        self.assertEqual(summary.loc["現行出題基準", "対象期間出題実績のある項目数"], 1)
        self.assertEqual(summary.loc["現行出題基準", "対象期間の項目出題実績率(%)"], 50.0)
        self.assertEqual(summary.loc["旧出題基準由来", "対象期間の問題数"], 1)
        self.assertEqual(items.loc[("旧分類", "記載なし"), "出題基準区分"], "旧出題基準由来")
        self.assertEqual(items.loc[("骨折", "骨幹部"), "対象期間出題回数"], 2)
        self.assertEqual(len(result["分類未確定問題"]), 1)
        self.assertIn(
            "公式出題基準全体の網羅率は算出していません",
            result["分析条件"].iloc[4]["内容"],
        )

    def test_frequency_bins_are_adjustable(self):
        result = self.analyze(boundaries=(2, 4, 8))
        items = result["項目別出題実績"].set_index(["大項目", "中項目"])

        self.assertEqual(
            items.loc[("骨折", "骨幹部"), "対象期間の出題頻度区分"], "1〜2回"
        )
        self.assertIn(
            "3〜4回",
            result["出題頻度区分別集計"]["対象期間の出題頻度区分"].tolist(),
        )

    def test_recent_rate_is_held_when_an_expected_round_is_missing(self):
        result = self.analyze(expected_rounds=(1, 2, 3))
        summary = result["出題基準区分別集計"].set_index("出題基準区分")

        self.assertEqual(summary.loc["現行出題基準", "直近期間の集計状態"], "未収録回あり")
        self.assertTrue(
            pd.isna(summary.loc["現行出題基準", "直近2回の項目出題実績率(%)"])
        )
        self.assertEqual(
            result["年度別の区分別出題"]
            .query("回 == 3")["収録状態"]
            .unique()
            .tolist(),
            ["未収録"],
        )

    def test_items_only_in_questions_are_included_in_the_observed_inventory(self):
        questions = pd.concat([
            self.questions,
            pd.DataFrame([question(2, 3, "追加領域", "追加項目")]),
        ], ignore_index=True)
        result = standard_history.analyze_standard_history(
            questions, self.frequency_items, "整形外科", [1, 2], 1, 2
        )
        row = result["項目別出題実績"].query("中項目 == '追加項目'").iloc[0]
        self.assertEqual(row["項目一覧の出典"], "過去問のみ")

    def test_frequency_bins_count_questions_not_only_exam_rounds(self):
        questions = pd.concat([
            self.questions,
            pd.DataFrame([question(1, 3, "骨折", "骨幹部")]),
        ], ignore_index=True)
        result = standard_history.analyze_standard_history(
            questions, self.frequency_items, "整形外科", [1, 2], 1, 2
        )
        item = result["項目別出題実績"].query("中項目 == '骨幹部'").iloc[0]

        self.assertEqual(item["対象期間出題数"], 3)
        self.assertEqual(item["対象期間出題回数"], 2)
        self.assertEqual(item["対象期間の出題頻度区分"], "2〜3回")


if __name__ == "__main__":
    unittest.main()
