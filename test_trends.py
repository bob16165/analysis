import unittest

import pandas as pd

import trends


def questions_for_rates(rates, missing_rounds=(), empty_major_rounds=()):
    rows = []
    for year, target_count in enumerate(rates, 1):
        if year in missing_rounds:
            continue
        question_number = 1
        if year not in empty_major_rounds:
            for _ in range(target_count):
                rows.append({
                    "回": year,
                    "問題番号": question_number,
                    "大項目": "骨折",
                    "中項目": "骨幹部骨折",
                })
                question_number += 1
            for _ in range(10 - target_count):
                rows.append({
                    "回": year,
                    "問題番号": question_number,
                    "大項目": "骨折",
                    "中項目": "その他",
                })
                question_number += 1
        else:
            for _ in range(10):
                rows.append({
                    "回": year,
                    "問題番号": question_number,
                    "大項目": "脱臼",
                    "中項目": "その他",
                })
                question_number += 1
        for _ in range(10):
            rows.append({
                "回": year,
                "問題番号": question_number,
                "大項目": "脱臼",
                "中項目": "その他",
            })
            question_number += 1
    return pd.DataFrame(rows)


class TrendAnalysisTests(unittest.TestCase):
    def analyze(self, questions, start=1, end=10, basis="大項目"):
        return trends.analyze_trends(
            questions,
            "整形外科",
            "骨折",
            "骨幹部骨折",
            basis,
            start,
            end,
            ["骨幹部骨折", "その他"],
        )

    def test_weighted_period_shares_and_regression_are_separate(self):
        result = self.analyze(questions_for_rates(list(range(1, 11))))
        summary = result["選択領域指標"].iloc[0]
        yearly = result["年度別推移"]

        self.assertEqual(summary["全期間出題割合(%)"], 55.0)
        self.assertEqual(summary["直近5回出題割合(%)"], 80.0)
        self.assertEqual(summary["前期後期差(pp)"], 50.0)
        self.assertEqual(summary["回帰判定"], "増加")
        self.assertEqual(summary["回帰推定差(pp)"], 90.0)
        self.assertEqual(yearly.loc[0, "出題割合(%)"], 10.0)
        self.assertEqual(
            yearly.loc[0, "問題ID"], "第1回/整形外科/No.1"
        )

    def test_zero_major_round_is_excluded_but_not_treated_as_missing(self):
        questions = questions_for_rates(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            empty_major_rounds=(3,),
        )
        result = self.analyze(questions)
        yearly = result["年度別推移"].set_index("回")
        periods = result["期間別出題割合"].set_index("期間")

        self.assertTrue(pd.isna(yearly.loc[3, "出題割合(%)"]))
        self.assertEqual(yearly.loc[3, "収録状態"], "収録あり")
        self.assertEqual(periods.loc["全期間", "集計状態"], "算出済み")
        self.assertEqual(periods.loc["全期間", "分母問題数(算出対象回)"], 90)

    def test_missing_exam_round_prevents_period_rate_and_trend_classification(self):
        result = self.analyze(
            questions_for_rates(list(range(1, 11)), missing_rounds=(5,))
        )
        summary = result["選択領域指標"].iloc[0]
        periods = result["期間別出題割合"].set_index("期間")

        self.assertEqual(summary["回帰判定"], "判定保留")
        self.assertIn("未収録回", summary["回帰判定理由"])
        self.assertTrue(pd.isna(periods.loc["選択期間", "出題割合(%)"]))
        self.assertEqual(periods.loc["選択期間", "未収録回"], "5")

    def test_large_slope_with_low_r_squared_is_not_classified_as_a_trend(self):
        result = self.analyze(questions_for_rates([1, 9, 2, 8, 3, 7, 4, 6, 5, 10]))
        summary = result["選択領域指標"].iloc[0]

        self.assertGreater(abs(summary["回帰推定差(pp)"]), 5)
        self.assertLess(summary["回帰R²"], 0.30)
        self.assertEqual(summary["回帰判定"], "判定保留")

    def test_short_series_and_incomplete_recent_window_are_not_overstated(self):
        result = self.analyze(questions_for_rates([1, 2, 3, 4, 5, 6]), 1, 6)
        summary = result["選択領域指標"].iloc[0]
        periods = result["期間別出題割合"].set_index("期間")

        self.assertEqual(summary["回帰判定"], "判定保留")
        self.assertIn("8回未満", summary["回帰判定理由"])
        self.assertEqual(periods.loc["直近5回", "出題割合(%)"], 40.0)
        self.assertTrue(pd.isna(periods.loc["直近10回", "出題割合(%)"]))
        self.assertEqual(periods.loc["直近10回", "集計状態"], "未収録回あり")

    def test_subject_denominator_is_explicit_and_changes_the_rate(self):
        result = self.analyze(
            questions_for_rates(list(range(1, 11))),
            basis="科目全体",
        )
        yearly = result["年度別推移"]
        summary = result["選択領域指標"].iloc[0]

        self.assertEqual(yearly.loc[0, "分母問題数"], 20)
        self.assertEqual(yearly.loc[0, "出題割合(%)"], 5.0)
        self.assertEqual(summary["全期間出題割合(%)"], 27.5)


if __name__ == "__main__":
    unittest.main()
