import tempfile
import unittest
from pathlib import Path

import pandas as pd

import question_review


def question_row(year, stem, number=1, major="外傷", middle="骨折"):
    return {
        "回": year,
        "問題番号": number,
        "大項目": major,
        "中項目": middle,
        "問題": stem,
        "選択肢1": "橈骨骨折",
        "選択肢2": "別の選択肢",
        "選択肢3": "別の選択肢",
        "選択肢4": "別の選択肢",
        "模範解答": "1",
        "正答": "橈骨骨折",
        "図": "",
        "症例問題": "",
        "解説": "",
    }


class ThemeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "review.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def annotate_theme(self, problem_id, themes, rationale="本文を確認した。"):
        question_review.save_annotation(
            self.database_path,
            problem_id,
            {
                "format_status": "未分類",
                "knowledge_status": "未分類",
                "themes": [
                    {"category": "疾患", "label": theme} for theme in themes
                ],
                "theme_rationale": rationale,
                "theme_status": "人間確認済み",
                "cross_subject_status": "未分類",
            },
        )

    def trend_sheet(self, incomplete_year=None):
        themes_by_year = {
            1: ["減少テーマ", "継続テーマ"],
            2: ["減少テーマ", "継続テーマ"],
            3: ["減少テーマ"],
            4: ["減少テーマ", "継続テーマ"],
            5: ["減少テーマ", "継続テーマ", "新出テーマ"],
            6: ["継続テーマ", "新出テーマ"],
            7: ["継続テーマ", "新出テーマ"],
            8: ["継続テーマ"],
            9: ["継続テーマ"],
            10: ["継続テーマ"],
            11: [],
            12: [],
            13: [],
        }
        questions = [
            question_row(year, "橈骨骨折の特徴を問う問題。")
            for year in range(1, 14)
        ]
        question_review.sync_question_annotations(
            self.database_path,
            {"整形外科": pd.DataFrame(questions)},
        )
        for year, themes in themes_by_year.items():
            if year == incomplete_year:
                continue
            self.annotate_theme(f"第{year}回/整形外科/No.1", themes)
        return themes_by_year

    def test_theme_trends_require_full_reviewed_rounds_and_compare_recent_10(self):
        self.trend_sheet()
        result = question_review.build_theme_change_analysis(
            self.database_path, 1, 13
        )
        trends = result["中項目別テーマ頻度比較"].set_index("テーマ")

        self.assertEqual(trends.loc["減少テーマ", "過去の出題回数"], 3)
        self.assertEqual(trends.loc["減少テーマ", "直近の出題回数"], 2)
        self.assertEqual(trends.loc["減少テーマ", "出題回割合の差(ポイント)"], -80.0)
        self.assertEqual(trends.loc["減少テーマ", "変化判定"], "減少")
        self.assertEqual(trends.loc["新出テーマ", "変化判定"], "新出候補")
        self.assertEqual(trends.loc["継続テーマ", "変化判定"], "継続（横ばい）")
        self.assertEqual(len(result["新出テーマ候補"]), 1)
        self.assertEqual(len(result["減少テーマ候補"]), 1)
        self.assertEqual(len(result["継続テーマ"]), 1)
        annual = result["年度別テーマ出現頻度"]
        self.assertEqual(
            annual.loc[
                annual["テーマ"].eq("継続テーマ") & annual["回"].eq(1),
                "テーマ出題問題数",
            ].iloc[0],
            1,
        )
        self.assertEqual(
            annual.loc[
                annual["テーマ"].eq("継続テーマ") & annual["回"].eq(11),
                "テーマ出題問題数",
            ].iloc[0],
            0,
        )

    def test_manual_alias_is_applied_to_historical_theme_trends(self):
        self.trend_sheet()
        question_review.save_theme_alias(
            self.database_path,
            "疾患",
            "新出テーマ",
            "継続テーマ",
            "同じ概念の表記として統合する。",
        )
        result = question_review.build_theme_change_analysis(
            self.database_path, 1, 13
        )
        trends = result["中項目別テーマ頻度比較"].set_index("テーマ")
        self.assertNotIn("新出テーマ", trends.index)
        self.assertEqual(trends.loc["継続テーマ", "直近の出題回数"], 7)
        annual = result["年度別テーマ出現頻度"]
        self.assertEqual(
            annual.loc[
                annual["テーマ"].eq("継続テーマ") & annual["回"].eq(6),
                "テーマ出題問題数",
            ].iloc[0],
            1,
        )

    def test_incomplete_round_is_not_treated_as_zero_or_new_theme_evidence(self):
        self.trend_sheet(incomplete_year=2)
        result = question_review.build_theme_change_analysis(
            self.database_path, 1, 13
        )
        trends = result["中項目別テーマ頻度比較"].set_index("テーマ")
        annual = result["年度別テーマ出現頻度"]
        incomplete = annual.loc[
            annual["回"].eq(2) & annual["テーマ"].eq("減少テーマ")
        ].iloc[0]

        self.assertEqual(incomplete["テーマ確認状態"], "未完了")
        self.assertTrue(pd.isna(incomplete["テーマ出題問題数"]))
        self.assertEqual(
            trends.loc["新出テーマ", "変化判定"],
            "比較保留（過去期間のレビュー回数不足）",
        )

    def test_surface_normalization_and_human_synonym_aliases(self):
        sheets = {
            "解剖学": pd.DataFrame([
                question_row(1, "肩関節の構造を問う。", major="関節", middle="肩関節")
            ]),
            "運動学": pd.DataFrame([
                question_row(1, "肩関節の運動を問う。", major="運動", middle="肩関節")
            ]),
        }
        question_review.sync_question_annotations(self.database_path, sheets)
        self.annotate_theme("第1回/解剖学/No.1", ["肩　関節"])
        self.annotate_theme("第1回/運動学/No.1", ["肩関節"])
        result = question_review.build_question_analysis(
            self.database_path, 1, 1
        )
        self.assertEqual(len(result["共通テーマランキング"]), 1)

        question_review.save_theme_alias(
            self.database_path,
            "疾患",
            "肩の関節",
            "肩関節",
            "この文脈では同一の解剖テーマとして扱う。",
        )
        alias_history = question_review.list_theme_alias_history(self.database_path)
        self.assertEqual(alias_history.iloc[0]["変更後統合先"], "肩関節")
        self.assertTrue(question_review.remove_theme_alias(
            self.database_path, "疾患", "肩の関節"
        ))
        self.assertTrue(question_review.list_theme_aliases(self.database_path).empty)
        self.assertFalse(question_review.list_theme_alias_history(self.database_path).empty)

    def test_alias_cycles_and_unconfirmed_canonical_targets_are_rejected(self):
        sheets = {
            "解剖学": pd.DataFrame([
                question_row(1, "肩関節を問う。", major="関節", middle="肩関節"),
                question_row(2, "肩の関節を問う。", major="関節", middle="肩関節"),
            ])
        }
        question_review.sync_question_annotations(self.database_path, sheets)
        self.annotate_theme("第1回/解剖学/No.1", ["肩関節"])
        self.annotate_theme("第2回/解剖学/No.1", ["肩の関節"])
        question_review.save_theme_alias(
            self.database_path, "疾患", "肩の関節", "肩関節", "同義語"
        )
        with self.assertRaisesRegex(ValueError, "循環参照"):
            question_review.save_theme_alias(
                self.database_path, "疾患", "肩関節", "肩の関節", "逆向き"
            )
        with self.assertRaisesRegex(ValueError, "人が確認済み"):
            question_review.save_theme_alias(
                self.database_path, "疾患", "未知語", "未確認テーマ", "根拠"
            )

    def test_candidate_phrases_remain_unconfirmed_and_include_question_ids(self):
        rows = [
            {
                "subject": "整形外科",
                "major": "外傷",
                "middle": "骨折",
                "exam_round": year,
                "problem_id": f"第{year}回/整形外科/No.1",
                "question": {
                    "問題": "橈骨骨折の特徴を問う。",
                    "選択肢": ["橈骨骨折", "選択肢二", "選択肢三", "選択肢四"],
                    "模範解答": "1",
                    "正答": "橈骨骨折",
                },
            }
            for year in (1, 2)
        ]
        candidates = question_review._extract_theme_candidates(
            rows, curated=["橈骨骨折"], stop=set()
        )
        candidate = candidates.loc[candidates["語句候補"].eq("橈骨骨折")].iloc[0]

        self.assertEqual(candidate["候補状態"], "未確認候補")
        self.assertEqual(candidate["出現問題数"], 2)
        self.assertIn("第1回/整形外科/No.1", candidate["問題ID"])
        self.assertIn("第2回/整形外科/No.1", candidate["問題ID"])


if __name__ == "__main__":
    unittest.main()
