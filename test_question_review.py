import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analysis
import question_review


def question_row(year, number, stem, figure="", case_marker="", major="領域", middle="項目"):
    return {
        "回": year,
        "問題番号": number,
        "大項目": major,
        "中項目": middle,
        "問題": stem,
        "選択肢1": "選択肢一",
        "選択肢2": "選択肢二",
        "選択肢3": "選択肢三",
        "選択肢4": "選択肢四",
        "模範解答": "1",
        "正答": "選択肢一",
        "図": figure,
        "症例問題": case_marker,
        "解説": "解説文",
    }


class QuestionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "review.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def question_sheets(self):
        return {
            "解剖学": pd.DataFrame([
                question_row(
                    1, 1, "患者の図を見て骨折の組合せとして正しいものはどれか。",
                    figure="figure.png", case_marker="症例",
                ),
                question_row(2, 1, "骨の名称はどれか。"),
                question_row(2, 2, "個別知識を説明せよ。"),
            ]),
            "運動学": pd.DataFrame([
                question_row(1, 1, "肩関節の動きを確認する問題です。"),
            ]),
        }

    def save_human_annotation(self, problem_id, subject, related_subject=None):
        related_subject = related_subject or (
            "運動学" if subject == "解剖学" else None
        )
        return question_review.save_annotation(
            self.database_path,
            problem_id,
            {
                "format_main": "症例問題",
                "format_secondary": ["正誤判定問題"],
                "format_rationale": "問題文と症例欄を確認した。",
                "format_status": "人間確認済み",
                "knowledge_level": "L3：適用",
                "knowledge_rationale": "症例の条件に知識を適用する。",
                "knowledge_status": "人間確認済み",
                "themes": [{"category": "解剖構造", "label": "肩関節"}],
                "theme_rationale": "設問の中心となる構造を原文で確認した。",
                "theme_status": "人間確認済み",
                "cross_subject_required": subject == "解剖学",
                "related_subjects": [related_subject] if related_subject else [],
                "cross_subject_rationale": (
                    f"肩関節の構造と{related_subject}の知識を組み合わせる必要がある。"
                    if subject == "解剖学"
                    else "この設問は出題科目内の知識で解ける。"
                ),
                "cross_subject_status": "人間確認済み",
            },
        )

    def test_format_suggestions_are_provisional_and_level_stays_unclassified(self):
        question_review.sync_question_annotations(
            self.database_path, self.question_sheets()
        )
        image_question = question_review.get_annotation(
            self.database_path, "第1回/解剖学/No.1"
        )
        simple_question = question_review.get_annotation(
            self.database_path, "第2回/解剖学/No.1"
        )
        unclassified_question = question_review.get_annotation(
            self.database_path, "第2回/解剖学/No.2"
        )

        self.assertEqual(image_question["suggested_format"], "画像問題")
        self.assertEqual(image_question["format_status"], "AI仮分類")
        self.assertIn("組合せ問題", image_question["format_secondary"])
        self.assertEqual(image_question["question"]["図"], "figure.png")
        self.assertEqual(image_question["knowledge_status"], "未分類")
        self.assertEqual(simple_question["format_main"], "単純知識問題")
        self.assertEqual(unclassified_question["format_status"], "未分類")

    def test_only_confirmed_theme_and_cross_subject_labels_build_structure(self):
        question_sheets = self.question_sheets()
        question_review.sync_question_annotations(self.database_path, question_sheets)
        for problem_id, subject in (
            ("第1回/解剖学/No.1", "解剖学"),
            ("第1回/運動学/No.1", "運動学"),
        ):
            self.save_human_annotation(problem_id, subject)
        question_review.sync_question_annotations(self.database_path, question_sheets)
        self.assertEqual(
            question_review.get_annotation(
                self.database_path, "第1回/解剖学/No.1"
            )["format_status"],
            "人間確認済み",
        )
        result = question_review.build_question_analysis(
            self.database_path, 1, 2
        )

        matrix = result["科目間共通テーマ行列"].set_index("科目")
        required_matrix = result["科目横断必須問題数行列"].set_index("科目")
        self.assertEqual(matrix.loc["解剖学", "運動学"], 1)
        self.assertEqual(required_matrix.loc["解剖学", "運動学"], 1)
        self.assertEqual(len(result["共通テーマランキング"]), 1)
        self.assertIn("原文で確認", result["共通テーマランキング"].iloc[0]["分類根拠"])
        self.assertEqual(
            result["中項目間の共通テーマネットワーク"].iloc[0]["共通確認済みテーマ数"],
            1,
        )
        self.assertEqual(len(result["科目横断が必要な確認済み問題"]), 1)
        self.assertEqual(
            result["問題形式別構成"]["人間確認済み問題数"].sum(), 2
        )
        self.assertEqual(
            result["知識レベル別構成"]["人間確認済み問題数"].sum(), 2
        )
        latest_format_period = result["期間別問題形式"].loc[
            result["期間別問題形式"]["期間"].str.startswith("直近10回")
        ].iloc[0]
        self.assertEqual(latest_format_period["未収録回数"], 0)
        self.assertTrue(analysis.to_excel_bytes(result).startswith(b"PK"))
        self.assertIn("問題別分類・確認詳細", analysis.to_markdown(result))

    def test_cross_subject_matrix_includes_related_subject_outside_period(self):
        question_sheets = self.question_sheets()
        question_sheets["病理学"] = pd.DataFrame([
            question_row(3, 1, "対象期間外の関連科目データ。")
        ])
        question_review.sync_question_annotations(self.database_path, question_sheets)
        self.save_human_annotation(
            "第1回/解剖学/No.1", "解剖学", related_subject="病理学"
        )

        result = question_review.build_question_analysis(
            self.database_path, 1, 2
        )
        matrix = result["科目横断必須問題数行列"].set_index("科目")

        self.assertEqual(matrix.loc["解剖学", "病理学"], 1)

    def test_changed_question_preserves_labels_and_requires_reconfirmation(self):
        question_sheets = self.question_sheets()
        question_review.sync_question_annotations(self.database_path, question_sheets)
        self.save_human_annotation("第1回/解剖学/No.1", "解剖学")
        question_sheets["解剖学"].loc[0, "問題"] += "改訂"
        sync_result = question_review.sync_question_annotations(
            self.database_path, question_sheets
        )

        annotation = question_review.get_annotation(
            self.database_path, "第1回/解剖学/No.1"
        )
        self.assertEqual(sync_result["内容変更数"], 1)
        self.assertEqual(annotation["format_status"], "再確認待ち")
        self.assertEqual(annotation["knowledge_status"], "再確認待ち")
        self.assertEqual(annotation["format_main"], "症例問題")
        self.assertEqual(annotation["knowledge_level"], "L3：適用")
        history = question_review.list_annotation_history(self.database_path)
        self.assertEqual(history.iloc[0]["変更理由"], "問題内容または画像・症例情報が変更")
        self.assertIn("改訂", history.iloc[0]["変更後問題文"])
        self.assertNotIn("改訂", history.iloc[0]["変更前問題文"])

    def test_removed_and_restored_question_keeps_labels_pending_review(self):
        question_sheets = self.question_sheets()
        question_review.sync_question_annotations(self.database_path, question_sheets)
        self.save_human_annotation("第1回/解剖学/No.1", "解剖学")
        question_sheets["解剖学"] = question_sheets["解剖学"].iloc[1:].reset_index(drop=True)
        question_review.sync_question_annotations(self.database_path, question_sheets)

        question_sheets["解剖学"] = pd.concat(
            [
                pd.DataFrame([
                    question_row(
                        1, 1, "患者の図を見て骨折の組合せとして正しいものはどれか。",
                        figure="figure.png", case_marker="症例",
                    )
                ]),
                question_sheets["解剖学"],
            ],
            ignore_index=True,
        )
        question_review.sync_question_annotations(self.database_path, question_sheets)
        annotation = question_review.get_annotation(
            self.database_path, "第1回/解剖学/No.1"
        )

        self.assertEqual(annotation["format_main"], "症例問題")
        self.assertEqual(annotation["format_status"], "再確認待ち")
        self.assertEqual(annotation["knowledge_level"], "L3：適用")
        self.assertEqual(annotation["theme_status"], "再確認待ち")
        history = question_review.list_annotation_history(self.database_path)
        deleted_history = history.loc[
            history["変更理由"].eq("問題が入力データから削除")
        ]
        self.assertFalse(deleted_history.empty)
        self.assertIn("患者", deleted_history.iloc[0]["変更前問題文"])

    def test_review_validation_and_note_history(self):
        question_review.sync_question_annotations(
            self.database_path, self.question_sheets()
        )
        with self.assertRaisesRegex(ValueError, "知識要求レベル・根拠"):
            question_review.save_annotation(
                self.database_path,
                "第2回/解剖学/No.2",
                {
                    "knowledge_level": "L2：理解",
                    "knowledge_status": "人間確認済み",
                    "knowledge_rationale": "",
                },
            )
        self.save_human_annotation("第1回/解剖学/No.1", "解剖学")
        history = question_review.list_annotation_history(
            self.database_path, problem_id="第1回/解剖学/No.1"
        )
        self.assertEqual(history.iloc[0]["変更後知識レベル"], "L3：適用")
        self.assertEqual(history.iloc[0]["変更後形式"], "症例問題")
        self.assertEqual(
            history.iloc[0]["変更後形式根拠"], "問題文と症例欄を確認した。"
        )

    def test_source_signature_and_theme_input_validation(self):
        question_review.sync_question_annotations(
            self.database_path,
            self.question_sheets(),
            source_signature="exam-file-v1",
        )
        self.assertEqual(
            question_review.get_source_signature(self.database_path), "exam-file-v1"
        )
        self.assertEqual(
            question_review.parse_theme_lines("疾患 | 骨折\n疾患 | 骨折"),
            [{"category": "疾患", "label": "骨折"}],
        )
        with self.assertRaisesRegex(ValueError, "テーマ区分"):
            question_review.parse_theme_lines("不明な区分 | テーマ")


if __name__ == "__main__":
    unittest.main()
