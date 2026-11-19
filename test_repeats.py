import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import repeats


def question_row(year, number, text, answer="1"):
    return {
        "回": year,
        "問題番号": number,
        "大項目": "領域",
        "中項目": "項目",
        "問題": text,
        "選択肢1": "選択肢一",
        "選択肢2": "選択肢二",
        "選択肢3": "選択肢三",
        "選択肢4": "選択肢四",
        "模範解答": answer,
        "正答": "選択肢一" if answer == "1" else "選択肢二",
    }


class RepeatReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "review.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def two_exact_questions(self):
        text = "共通の詳細な問題文で、重要な知識を確認する設問です。"
        return {
            "科目": pd.DataFrame([
                question_row(1, 1, text),
                question_row(2, 1, text),
            ])
        }

    def test_exact_candidates_keep_human_review_when_only_whitespace_changes(self):
        questions = self.two_exact_questions()
        first_sync = repeats.refresh_review_database(
            self.database_path, questions, source_signature="file-version-1"
        )
        candidates = repeats.list_review_candidates(self.database_path)

        self.assertEqual(first_sync["完全一致候補数"], 1)
        self.assertEqual(
            repeats.get_source_signature(self.database_path), "file-version-1"
        )
        self.assertEqual(candidates.loc[0, "auto_type"], "A")
        relation_id = candidates.loc[0, "relation_id"]
        repeats.save_relation_review(self.database_path, relation_id, "A", "同一問題")

        questions["科目"].loc[0, "問題"] = " 共通の詳細な問題文で、重要な知識を確認する設問です。 "
        repeats.refresh_review_database(self.database_path, questions)

        reviewed = repeats.list_review_candidates(self.database_path, "全て")
        self.assertEqual(reviewed.loc[0, "review_status"], "確認済み")
        self.assertEqual(reviewed.loc[0, "reviewed_type"], "A")
        self.assertTrue(repeats.list_question_changes(self.database_path).empty)

    def test_changed_question_requires_review_and_preserves_before_after(self):
        questions = self.two_exact_questions()
        repeats.refresh_review_database(self.database_path, questions)
        relation_id = repeats.list_review_candidates(self.database_path).loc[0, "relation_id"]
        repeats.save_relation_review(self.database_path, relation_id, "A", "人が確認")

        questions["科目"].loc[0, "問題"] = (
            "共通の詳細な問題文で、重要な知識を確認する設問です。追加の説明。"
        )
        repeats.refresh_review_database(self.database_path, questions)

        relation = repeats.list_review_candidates(self.database_path, "全て").iloc[0]
        self.assertEqual(relation["review_status"], "再確認待ち")
        self.assertEqual(relation["reviewed_type"], "A")
        detail = repeats.get_relation_detail(self.database_path, relation_id)
        changed_sides = [detail[side] for side in ("a", "b") if detail[side]["changed"]]
        self.assertEqual(len(changed_sides), 1)
        self.assertNotEqual(
            changed_sides[0]["previous"]["問題"], changed_sides[0]["current"]["問題"]
        )
        history = repeats.list_relation_history(self.database_path)
        self.assertTrue(history["reason"].str.contains("内容が変更").any())
        changes = repeats.list_question_changes(self.database_path)
        self.assertEqual(len(changes), 1)
        self.assertNotEqual(
            changes.loc[0, "変更前の問題文"], changes.loc[0, "変更後の問題文"]
        )

    def test_answer_change_also_requires_relation_review(self):
        questions = self.two_exact_questions()
        repeats.refresh_review_database(self.database_path, questions)
        relation_id = repeats.list_review_candidates(self.database_path).loc[0, "relation_id"]
        repeats.save_relation_review(self.database_path, relation_id, "A")

        questions["科目"].loc[0, "模範解答"] = "2"
        questions["科目"].loc[0, "正答"] = "選択肢二"
        repeats.refresh_review_database(self.database_path, questions)

        detail = repeats.get_relation_detail(self.database_path, relation_id)
        self.assertEqual(detail["relation"]["review_status"], "再確認待ち")
        changes = repeats.list_question_changes(self.database_path)
        self.assertEqual(changes.loc[0, "変更前の模範解答"], "1")
        self.assertEqual(changes.loc[0, "変更後の模範解答"], "2")

    def test_review_history_keeps_old_and_new_notes(self):
        questions = self.two_exact_questions()
        repeats.refresh_review_database(self.database_path, questions)
        relation_id = repeats.list_review_candidates(self.database_path).loc[0, "relation_id"]
        repeats.save_relation_review(self.database_path, relation_id, "A", "初回メモ")
        repeats.save_relation_review(self.database_path, relation_id, "B", "修正メモ")

        history = repeats.list_relation_history(self.database_path)
        latest = history.iloc[0]
        self.assertEqual(latest["previous_note"], "初回メモ")
        self.assertEqual(latest["new_note"], "修正メモ")
        self.assertEqual(latest["previous_type"], "A")
        self.assertEqual(latest["new_type"], "B")

    def test_existing_review_history_schema_is_migrated(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE relation_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    relation_id TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    previous_type TEXT,
                    new_type TEXT,
                    reason TEXT NOT NULL,
                    before_hash_a TEXT,
                    before_hash_b TEXT,
                    after_hash_a TEXT,
                    after_hash_b TEXT,
                    changed_at TEXT NOT NULL
                )
            """)

        repeats.get_source_signature(self.database_path)
        with sqlite3.connect(self.database_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(relation_history)")
            }
        self.assertIn("previous_note", columns)
        self.assertIn("new_note", columns)

    def test_manual_related_pair_is_saved_as_c_with_history(self):
        questions = {
            "科目": pd.DataFrame([
                question_row(1, 1, "解剖構造を確認する別の設問です。"),
                question_row(2, 2, "臨床判断を確認する別の設問です。"),
            ])
        }
        repeats.refresh_review_database(self.database_path, questions)
        relation_id = repeats.add_manual_related_pair(
            self.database_path,
            "第1回/科目/No.1",
            "第2回/科目/No.2",
            "同じ解剖構造の異なる知識",
        )

        detail = repeats.get_relation_detail(self.database_path, relation_id)
        self.assertEqual(detail["relation"]["reviewed_type"], "C")
        self.assertEqual(detail["relation"]["review_status"], "修正済み")
        self.assertEqual(len(repeats.list_relation_history(self.database_path)), 1)

    def test_automatic_similarity_threshold_is_applied(self):
        questions = {
            "科目": pd.DataFrame([
                question_row(1, 1, "長い共通の設問文で複数の医学知識を確認するための問題です。"),
                question_row(2, 1, "長い共通の設問文で複数の医学知識を確認するための問題です。追記。"),
            ])
        }
        repeats.refresh_review_database(self.database_path, questions, min_similarity=0.1)
        candidates = repeats.list_review_candidates(self.database_path)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.loc[0, "auto_type"], "B")

        high_threshold_database = Path(self.temp_dir.name) / "high-threshold.sqlite3"
        repeats.refresh_review_database(
            high_threshold_database, questions, min_similarity=0.99
        )
        self.assertTrue(repeats.list_review_candidates(high_threshold_database).empty)

    def test_same_exam_round_is_not_a_reappearance_candidate(self):
        questions = {
            "科目": pd.DataFrame([
                question_row(1, 1, "同じ問題文です。"),
                question_row(1, 2, "同じ問題文です。"),
            ])
        }
        repeats.refresh_review_database(self.database_path, questions)
        self.assertTrue(repeats.list_review_candidates(self.database_path).empty)


if __name__ == "__main__":
    unittest.main()
