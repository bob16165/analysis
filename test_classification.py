import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analysis
import classification


def question_row(number, stem, figure="", case_marker="", explanation="解説"):
    return {
        "回": 1,
        "問題番号": number,
        "大項目": "領域",
        "中項目": "項目",
        "問題": stem,
        "選択肢1": "選択肢一",
        "選択肢2": "選択肢二",
        "選択肢3": "選択肢三",
        "選択肢4": "選択肢四",
        "模範解答": "1",
        "正答": "選択肢一",
        "図": figure,
        "症例問題": case_marker,
        "解説": explanation,
    }


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "review.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def question_sheets(self):
        return {
            "解剖学": pd.DataFrame([
                question_row(1, "橈骨の名称はどれか。"),
                question_row(
                    2,
                    "患者の所見と骨折の組合せとして正しいのはどれか。",
                    figure="figure.png",
                    case_marker="症例",
                ),
                question_row(3, "エネルギー代謝量の算出に必要なのはどれか。"),
            ])
        }

    def _confirmed_values(self, clinical, taxonomy):
        formats = {dimension: "非該当" for dimension in classification.FORMAT_DIMENSIONS}
        formats["臨床状況・症例提示"] = clinical
        return {
            "format_values": formats,
            "format_rationale": "問題文と選択肢を確認した。",
            "format_status": "人間確認済み",
            "taxonomy": taxonomy,
            "taxonomy_rationale": "正答に必要な認知処理を確認した。",
            "taxonomy_status": "人間確認済み",
        }

    def test_rule_batches_are_explicit_and_candidates_remain_provisional(self):
        classification.sync_questions(self.database_path, self.question_sheets())
        self.assertEqual(classification.get_progress(self.database_path)["ルール処理済み"], 0)

        job = classification.start_rule_job(self.database_path)
        first_batch = classification.run_rule_batch(
            self.database_path, job["job_id"], batch_size=1
        )
        self.assertEqual(first_batch["処理件数"], 1)
        self.assertEqual(first_batch["未処理件数"], 2)

        self.assertTrue(classification.pause_rule_job(self.database_path, job["job_id"]))
        with self.assertRaisesRegex(ValueError, "実行中ではありません"):
            classification.run_rule_batch(self.database_path, job["job_id"], batch_size=1)
        self.assertTrue(classification.resume_rule_job(self.database_path, job["job_id"]))
        classification.run_rule_batch(self.database_path, job["job_id"], batch_size=10)

        direct = classification.get_classification(
            self.database_path, "第1回/解剖学/No.1"
        )
        case = classification.get_classification(
            self.database_path, "第1回/解剖学/No.2"
        )
        calculation_wording = classification.get_classification(
            self.database_path, "第1回/解剖学/No.3"
        )
        self.assertEqual(direct["taxonomy"], "Ⅰ型")
        self.assertEqual(direct["taxonomy_status"], "ルール候補")
        self.assertEqual(case["format_values"]["画像使用"], "該当")
        self.assertEqual(case["format_values"]["組合せ問題"], "該当")
        self.assertEqual(case["taxonomy"], "判定保留")
        self.assertEqual(calculation_wording["format_values"]["計算問題"], "判定保留")
        self.assertEqual(
            classification.get_job_status(self.database_path, job["job_id"])["status"],
            "完了",
        )

    def test_manual_values_are_preserved_and_flagged_after_input_change(self):
        sheets = self.question_sheets()
        classification.sync_questions(self.database_path, sheets)
        job = classification.start_rule_job(self.database_path)
        classification.run_rule_batch(self.database_path, job["job_id"], batch_size=10)

        problem_id = "第1回/解剖学/No.2"
        classification.save_classification(
            self.database_path,
            problem_id,
            self._confirmed_values("該当", "Ⅲ型"),
        )
        sheets["解剖学"].loc[1, "解説"] = "解説を改訂"
        result = classification.sync_questions(self.database_path, sheets)

        saved = classification.get_classification(self.database_path, problem_id)
        self.assertEqual(result["内容変更数"], 1)
        self.assertEqual(saved["format_status"], "再確認待ち")
        self.assertEqual(saved["taxonomy_status"], "再確認待ち")
        self.assertEqual(saved["format_values"]["臨床状況・症例提示"], "該当")
        self.assertEqual(saved["taxonomy"], "Ⅲ型")
        rerun_job = classification.start_rule_job(self.database_path)
        classification.run_rule_batch(
            self.database_path, rerun_job["job_id"], batch_size=10
        )
        after_rerun = classification.get_classification(
            self.database_path, problem_id
        )
        self.assertEqual(after_rerun["format_status"], "再確認待ち")
        self.assertEqual(after_rerun["taxonomy_status"], "再確認待ち")
        self.assertEqual(
            after_rerun["format_values"]["臨床状況・症例提示"], "該当"
        )
        self.assertEqual(after_rerun["taxonomy"], "Ⅲ型")
        history = classification.list_classification_history(
            self.database_path, problem_id=problem_id
        )
        self.assertIn("問題内容", history.iloc[0]["変更理由"])

    def test_aggregate_ratios_show_denominators_and_hold_counts(self):
        sheets = self.question_sheets()
        classification.sync_questions(self.database_path, sheets)
        for problem_id, clinical, taxonomy in (
            ("第1回/解剖学/No.1", "該当", "Ⅰ型"),
            ("第1回/解剖学/No.2", "非該当", "判定保留"),
        ):
            classification.save_classification(
                self.database_path,
                problem_id,
                self._confirmed_values(clinical, taxonomy),
            )

        result = classification.build_classification_analysis(
            self.database_path, 1, 1, "解剖学"
        )
        form = result["問題形式別集計"]
        form_row = form.loc[
            form["集計軸"].eq("全体")
            & form["問題形式"].eq("臨床状況・症例提示")
        ].iloc[0]
        self.assertEqual(form_row["該当数"], 1)
        self.assertEqual(form_row["非該当数"], 1)
        self.assertEqual(form_row["判定済み分母"], 2)
        self.assertEqual(form_row["該当割合(%)"], 50.0)

        taxonomy = result["Taxonomy別集計"]
        taxonomy_row = taxonomy.loc[
            taxonomy["集計軸"].eq("全体") & taxonomy["Taxonomy"].eq("Ⅰ型")
        ].iloc[0]
        self.assertEqual(taxonomy_row["分類済み問題数"], 1)
        self.assertEqual(taxonomy_row["確認済み保留数"], 1)
        self.assertEqual(taxonomy_row["出題割合(%)"], 100.0)
        self.assertEqual(len(result["問題別分類・根拠"]), 3)
        self.assertEqual(
            set(result["問題形式別集計"]["集計軸"]),
            {"全体", "科目", "年度", "大項目", "中項目"},
        )
        self.assertTrue(analysis.to_excel_bytes(result).startswith(b"PK"))
        self.assertIn("Taxonomy別集計", analysis.to_markdown(result))

    def test_sync_does_not_run_rules_and_source_signature_is_persisted(self):
        classification.sync_questions(
            self.database_path,
            self.question_sheets(),
            source_signature="source-v1",
        )
        self.assertEqual(classification.get_progress(self.database_path)["未処理"], 3)
        self.assertEqual(
            classification.get_source_signature(self.database_path), "source-v1"
        )
        self.assertEqual(classification.get_job_status(self.database_path), None)

    def test_format_value_filter_without_dimension_matches_any_format(self):
        classification.sync_questions(self.database_path, self.question_sheets())
        job = classification.start_rule_job(self.database_path)
        classification.run_rule_batch(self.database_path, job["job_id"], batch_size=10)

        matches = classification.list_classification_questions(
            self.database_path,
            format_value="該当",
        )
        self.assertEqual(
            {row["problem_id"] for row in matches},
            {"第1回/解剖学/No.2"},
        )

    def test_errors_are_visible_and_retryable_without_reprocessing_successes(self):
        classification.sync_questions(self.database_path, self.question_sheets())
        problem_id = "第1回/解剖学/No.1"
        question = classification.get_classification(
            self.database_path, problem_id
        )["question"]
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "UPDATE classification_annotations SET question_json='{' WHERE problem_id=?",
                (problem_id,),
            )
            connection.commit()
        finally:
            connection.close()

        first_job = classification.start_rule_job(self.database_path)
        result = classification.run_rule_batch(
            self.database_path, first_job["job_id"], batch_size=10
        )
        self.assertEqual(result["エラー件数"], 1)
        self.assertEqual(
            classification.list_job_errors(
                self.database_path, first_job["job_id"]
            ).iloc[0]["問題ID"],
            problem_id,
        )

        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                "UPDATE classification_annotations SET question_json=? WHERE problem_id=?",
                (json.dumps(question, ensure_ascii=False), problem_id),
            )
            connection.commit()
        finally:
            connection.close()
        retry_job = classification.start_rule_job(
            self.database_path, retry_errors_only=True
        )
        self.assertEqual(retry_job["total_count"], 1)
        retry = classification.run_rule_batch(
            self.database_path, retry_job["job_id"], batch_size=10
        )
        self.assertEqual(retry["処理済み件数"], 1)
        self.assertEqual(
            classification.get_classification(
                self.database_path, problem_id
            )["rule_status"],
            "完了",
        )


if __name__ == "__main__":
    unittest.main()
