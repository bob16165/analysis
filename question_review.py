"""Human-reviewed question classifications and cross-subject structure."""

import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import repeats
import terms


FORMATS = [
    "単純知識問題",
    "正誤判定問題",
    "組合せ問題",
    "症例問題",
    "画像問題",
    "臨床判断問題",
    "その他",
]
KNOWLEDGE_LEVELS = ["L1：想起", "L2：理解", "L3：適用", "L4：統合"]
THEME_DOMAINS = [
    "疾患",
    "解剖構造",
    "生理機能",
    "病態",
    "症状",
    "検査法",
    "治療法",
    "その他",
]
ANNOTATION_STATUSES = [
    "未分類", "AI仮分類", "人間確認済み", "修正済み", "再確認待ち",
]
CONFIRMED_STATUSES = ("人間確認済み", "修正済み")
CHOICE_COLUMNS = ["選択肢1", "選択肢2", "選択肢3", "選択肢4"]
THEME_MIN_COMPARISON_ROUNDS = 3
THEME_TREND_THRESHOLD_PP = 10.0
UNCLASSIFIED_MIDDLE_LABELS = {"", "記載なし"}
ANNOTATION_FIELDS = [
    "problem_id", "exam_round", "subject", "major", "middle", "problem_number",
    "content_hash", "question_json", "active", "suggested_format",
    "suggested_secondary_json", "suggestion_rationale", "format_main",
    "format_secondary_json", "format_rationale", "format_status",
    "knowledge_level", "knowledge_rationale", "knowledge_status", "themes_json",
    "theme_rationale", "theme_status", "cross_subject_required",
    "related_subjects_json", "cross_subject_rationale", "cross_subject_status",
    "updated_at",
]
REVIEW_LIST_COLUMNS = [
    "problem_id", "exam_round", "subject", "major", "middle", "problem_number",
    "suggested_format", "format_main", "format_status", "knowledge_level",
    "knowledge_status", "theme_status", "cross_subject_status",
]


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _normalized(value) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", _text(value)))


def _theme_identity(category: str, label: str) -> str:
    return json.dumps(
        [category, _normalized(label)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _active_theme_aliases(connection) -> dict[str, dict]:
    return {
        row["alias_key"]: _row_dict(row)
        for row in connection.execute(
            "SELECT * FROM theme_aliases WHERE active=1"
        ).fetchall()
    }


def _resolve_theme_label(category: str, label: str, aliases: dict[str, dict]) -> str:
    current = _text(label)
    visited = set()
    while True:
        key = _theme_identity(category, current)
        if key not in aliases:
            return current
        if key in visited:
            raise ValueError("テーマの表記統合ルールに循環参照があります。")
        visited.add(key)
        current = aliases[key]["canonical_label"]


def _format_suggestion(question: dict) -> tuple[Optional[str], list[str], str]:
    stem = _normalized(question.get("問題"))
    flags = []
    evidence = []

    if _text(question.get("図")):
        flags.append("画像問題")
        evidence.append("「図」列に画像参照がある")

    case_marker = _normalized(question.get("症例問題"))
    case_patterns = ("患者", "主訴", "現病歴", "来院", "受診")
    if case_marker == "症例" or any(pattern in stem for pattern in case_patterns):
        flags.append("症例問題")
        evidence.append(
            "Excelの症例問題欄または問題文に症例を示す記載がある"
        )

    if any(pattern in stem for pattern in ("組合せ", "組み合わせ")):
        flags.append("組合せ問題")
        evidence.append("問題文に「組合せ」の表現がある")

    clinical_patterns = (
        "最も考えられる", "最も適切", "行うべき", "とるべき", "優先する",
        "初期対応", "処置として",
    )
    if "症例問題" in flags and any(pattern in stem for pattern in clinical_patterns):
        flags.append("臨床判断問題")
        evidence.append("症例の状況に対する判断を問う表現がある")

    correctness_patterns = (
        "誤っている", "正しいもの", "正しいのは", "正しくない",
        "適切でない", "適切なのは",
    )
    if any(pattern in stem for pattern in correctness_patterns):
        flags.append("正誤判定問題")
        evidence.append("問題文に正誤を選択する表現がある")

    if not flags and any(pattern in stem for pattern in ("名称はどれか", "何か", "どれか")):
        flags.append("単純知識問題")
        evidence.append("問題文が用語・知識の直接選択を求める表現である")

    priority = [
        "画像問題", "組合せ問題", "臨床判断問題", "症例問題",
        "正誤判定問題", "単純知識問題",
    ]
    primary = next((category for category in priority if category in flags), None)
    if primary is None:
        return None, [], ""
    secondary = [category for category in priority if category in flags and category != primary]
    return primary, secondary, "仮分類ルール: " + "。".join(evidence)


def parse_theme_lines(text: str) -> list[dict[str, str]]:
    themes = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if "|" not in line:
            raise ValueError(f"{line_number}行目は「テーマ区分 | テーマ名」の形式で入力してください。")
        domain, label = (part.strip() for part in line.split("|", 1))
        if domain not in THEME_DOMAINS:
            raise ValueError(
                f"{line_number}行目のテーマ区分は{', '.join(THEME_DOMAINS)}から選んでください。"
            )
        if not label:
            raise ValueError(f"{line_number}行目のテーマ名が空です。")
        key = (domain, unicodedata.normalize("NFKC", label))
        if key not in seen:
            themes.append({"category": domain, "label": label})
            seen.add(key)
    return themes


@contextmanager
def _database(path):
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            _create_schema(connection)
            yield connection
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS question_annotations (
            problem_id TEXT PRIMARY KEY,
            exam_round INTEGER NOT NULL,
            subject TEXT NOT NULL,
            major TEXT,
            middle TEXT,
            problem_number TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            question_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            suggested_format TEXT,
            suggested_secondary_json TEXT NOT NULL DEFAULT '[]',
            suggestion_rationale TEXT NOT NULL DEFAULT '',
            format_main TEXT,
            format_secondary_json TEXT NOT NULL DEFAULT '[]',
            format_rationale TEXT NOT NULL DEFAULT '',
            format_status TEXT NOT NULL DEFAULT '未分類',
            knowledge_level TEXT,
            knowledge_rationale TEXT NOT NULL DEFAULT '',
            knowledge_status TEXT NOT NULL DEFAULT '未分類',
            themes_json TEXT NOT NULL DEFAULT '[]',
            theme_rationale TEXT NOT NULL DEFAULT '',
            theme_status TEXT NOT NULL DEFAULT '未分類',
            cross_subject_required INTEGER,
            related_subjects_json TEXT NOT NULL DEFAULT '[]',
            cross_subject_rationale TEXT NOT NULL DEFAULT '',
            cross_subject_status TEXT NOT NULL DEFAULT '未分類',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS question_annotation_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT NOT NULL,
            previous_json TEXT,
            new_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            changed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS question_review_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS theme_aliases (
            alias_key TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            alias_label TEXT NOT NULL,
            canonical_label TEXT NOT NULL,
            rationale TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS theme_alias_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias_key TEXT NOT NULL,
            previous_json TEXT,
            new_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            changed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS question_annotations_subject_round_idx
            ON question_annotations (subject, exam_round, active);
        CREATE INDEX IF NOT EXISTS question_annotations_format_status_idx
            ON question_annotations (format_status, active);
    """)


def _question_records(question_sheets: dict[str, pd.DataFrame]) -> list[dict]:
    base_records = {
        record["problem_id"]: record
        for record in repeats.collect_question_records(question_sheets)
    }
    output = []
    for subject, questions in question_sheets.items():
        for _, row in questions.iterrows():
            if pd.isna(pd.to_numeric(row.get("回"), errors="coerce")):
                continue
            problem_id = repeats.canonical_problem_id(row, subject)
            base = base_records[problem_id]
            figure = _text(row.get("図"))
            case_marker = _text(row.get("症例問題"))
            content_hash = hashlib.sha256(
                json.dumps(
                    [base["content_hash"], _normalized(figure), _normalized(case_marker)],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            content = json.loads(base["snapshot_json"])
            content.update({
                "図": figure,
                "症例問題": case_marker,
                "解説": _text(row.get("解説")),
            })
            suggested, secondary, rationale = _format_suggestion(content)
            output.append({
                "problem_id": problem_id,
                "exam_round": base["round"],
                "subject": subject,
                "major": _text(row.get("大項目")),
                "middle": _text(row.get("中項目")),
                "problem_number": base["problem_number"],
                "content_hash": content_hash,
                "question_json": json.dumps(content, ensure_ascii=False, sort_keys=True),
                "suggested_format": suggested,
                "suggested_secondary_json": json.dumps(secondary, ensure_ascii=False),
                "suggestion_rationale": rationale,
            })
    output.sort(key=lambda record: record["problem_id"])
    return output


def _row_dict(row) -> dict:
    return {key: row[key] for key in row.keys()}


def _insert_history(
    connection,
    problem_id: str,
    before: Optional[dict],
    after: dict,
    reason: str,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT INTO question_annotation_history
            (problem_id, previous_json, new_json, reason, changed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            problem_id,
            json.dumps(before, ensure_ascii=False, sort_keys=True) if before else None,
            json.dumps(after, ensure_ascii=False, sort_keys=True),
            reason,
            now,
        ),
    )


def _upsert_annotation(connection, record: dict, now: str) -> None:
    columns = ", ".join(ANNOTATION_FIELDS)
    placeholders = ", ".join("?" for _ in ANNOTATION_FIELDS)
    updates = ", ".join(
        f"{field}=excluded.{field}" for field in ANNOTATION_FIELDS if field != "problem_id"
    )
    values = {
        **record,
        "active": 1,
        "suggested_secondary_json": record["suggested_secondary_json"],
        "format_main": record["suggested_format"],
        "format_secondary_json": record["suggested_secondary_json"],
        "format_rationale": record["suggestion_rationale"],
        "format_status": "AI仮分類" if record["suggested_format"] else "未分類",
        "knowledge_level": None,
        "knowledge_rationale": "",
        "knowledge_status": "未分類",
        "themes_json": "[]",
        "theme_rationale": "",
        "theme_status": "未分類",
        "cross_subject_required": None,
        "related_subjects_json": "[]",
        "cross_subject_rationale": "",
        "cross_subject_status": "未分類",
        "updated_at": now,
    }
    connection.execute(
        f"INSERT INTO question_annotations ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(problem_id) DO UPDATE SET {updates}",
        tuple(values[field] for field in ANNOTATION_FIELDS),
    )


def sync_question_annotations(
    database_path,
    question_sheets: dict[str, pd.DataFrame],
    source_signature: Optional[str] = None,
) -> dict[str, int]:
    records = _question_records(question_sheets)
    current_ids = {record["problem_id"] for record in records}
    now = datetime.now(timezone.utc).isoformat()
    added = changed = removed = 0
    with _database(database_path) as connection:
        existing = {
            row["problem_id"]: row
            for row in connection.execute(
                "SELECT * FROM question_annotations"
            ).fetchall()
        }
        previous_active = {
            problem_id: row for problem_id, row in existing.items() if row["active"]
        }
        for record in records:
            old = existing.get(record["problem_id"])
            if old is None:
                _upsert_annotation(connection, record, now)
                added += 1
                continue

            old_data = _row_dict(old)
            if old["content_hash"] != record["content_hash"]:
                changed += 1
                updated = {
                    **old_data,
                    **record,
                    "active": 1,
                    "updated_at": now,
                }
                for status_field in (
                    "format_status", "knowledge_status", "theme_status",
                    "cross_subject_status",
                ):
                    if old_data[status_field] != "未分類":
                        updated[status_field] = "再確認待ち"
                _insert_history(
                    connection,
                    record["problem_id"],
                    _annotation_snapshot(old_data),
                    _annotation_snapshot(updated),
                    "問題内容または画像・症例情報が変更",
                    now,
                )
                assignments = ", ".join(
                    f"{field}=?" for field in ANNOTATION_FIELDS if field != "problem_id"
                )
                connection.execute(
                    f"UPDATE question_annotations SET {assignments} WHERE problem_id=?",
                    tuple(updated[field] for field in ANNOTATION_FIELDS if field != "problem_id")
                    + (record["problem_id"],),
                )
                continue

            connection.execute(
                """
                UPDATE question_annotations
                SET major=?, middle=?, question_json=?, active=1, updated_at=?
                WHERE problem_id=?
                """,
                (
                    record["major"], record["middle"], record["question_json"],
                    now, record["problem_id"],
                ),
            )

        for problem_id, old in previous_active.items():
            if problem_id in current_ids:
                continue
            removed += 1
            before = _row_dict(old)
            updated = {**before, "active": 0, "updated_at": now}
            for status_field in (
                "format_status", "knowledge_status", "theme_status",
                "cross_subject_status",
            ):
                if before[status_field] != "未分類":
                    updated[status_field] = "再確認待ち"
            _insert_history(
                connection,
                problem_id,
                _annotation_snapshot(before),
                _annotation_snapshot(updated),
                "問題が入力データから削除",
                now,
            )
            connection.execute(
                """
                UPDATE question_annotations
                SET active=0, format_status=?, knowledge_status=?, theme_status=?,
                    cross_subject_status=?, updated_at=?
                WHERE problem_id=?
                """,
                (
                    updated["format_status"], updated["knowledge_status"],
                    updated["theme_status"], updated["cross_subject_status"],
                    now, problem_id,
                ),
            )

        if source_signature is not None:
            connection.execute(
                """
                INSERT INTO question_review_metadata (key, value, updated_at)
                VALUES ('source_signature', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at
                """,
                (source_signature, now),
            )

        provisional = connection.execute(
            """
            SELECT COUNT(*) FROM question_annotations
            WHERE active=1 AND format_status='AI仮分類'
            """
        ).fetchone()[0]
        pending = connection.execute(
            """
            SELECT COUNT(*) FROM question_annotations
            WHERE active=1 AND (
                format_status='再確認待ち' OR knowledge_status='再確認待ち'
                OR theme_status='再確認待ち' OR cross_subject_status='再確認待ち'
            )
            """
        ).fetchone()[0]

    return {
        "問題数": len(records),
        "新規登録数": added,
        "内容変更数": changed,
        "入力データからの削除数": removed,
        "形式のAI仮分類数": provisional,
        "再確認待ち問題数": pending,
    }


def get_source_signature(database_path) -> Optional[str]:
    with _database(database_path) as connection:
        row = connection.execute(
            "SELECT value FROM question_review_metadata WHERE key='source_signature'"
        ).fetchone()
    return row["value"] if row else None


def get_review_scope(database_path) -> dict[str, list]:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT subject, exam_round
            FROM question_annotations
            WHERE active=1
            ORDER BY exam_round, subject
            """
        ).fetchall()
    return {
        "subjects": sorted({row["subject"] for row in rows}),
        "rounds": sorted({row["exam_round"] for row in rows}),
    }


def _confirmed_theme_labels(connection) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT themes_json FROM question_annotations
        WHERE active=1 AND theme_status IN (?, ?)
        ORDER BY problem_id
        """,
        CONFIRMED_STATUSES,
    ).fetchall()
    labels = {}
    for row in rows:
        for theme in json.loads(row["themes_json"]):
            key = _theme_identity(theme["category"], theme["label"])
            labels.setdefault(key, {
                "category": theme["category"],
                "label": theme["label"],
            })
    return sorted(labels.values(), key=lambda theme: (theme["category"], theme["label"]))


def list_confirmed_theme_labels(database_path) -> pd.DataFrame:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT problem_id, themes_json FROM question_annotations
            WHERE active=1 AND theme_status IN (?, ?)
            ORDER BY problem_id
            """,
            CONFIRMED_STATUSES,
        ).fetchall()
    grouped = {}
    for row in rows:
        for theme in json.loads(row["themes_json"]):
            key = _theme_identity(theme["category"], theme["label"])
            item = grouped.setdefault(
                key,
                {
                    "区分": theme["category"],
                    "テーマ名": theme["label"],
                    "問題ID": set(),
                },
            )
            item["問題ID"].add(row["problem_id"])
    return pd.DataFrame([
        {
            "区分": value["区分"],
            "テーマ名": value["テーマ名"],
            "問題数": len(value["問題ID"]),
            "問題ID": "、".join(sorted(value["問題ID"])),
        }
        for value in sorted(
            grouped.values(), key=lambda item: (item["区分"], item["テーマ名"])
        )
    ], columns=["区分", "テーマ名", "問題数", "問題ID"])


def list_theme_aliases(database_path) -> pd.DataFrame:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT category, alias_label, canonical_label, rationale, updated_at
            FROM theme_aliases
            WHERE active=1
            ORDER BY category, alias_label
            """
        ).fetchall()
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["category", "alias_label", "canonical_label", "rationale", "updated_at"],
    )


def save_theme_alias(
    database_path,
    category: str,
    alias_label: str,
    canonical_label: str,
    rationale: str,
) -> bool:
    category = _text(category)
    alias_label = _text(alias_label)
    canonical_label = _text(canonical_label)
    rationale = _text(rationale)
    if category not in THEME_DOMAINS:
        raise ValueError("テーマ区分の指定が不正です。")
    if not alias_label or not canonical_label or not rationale:
        raise ValueError("統合前テーマ・統合先テーマ・判断根拠を入力してください。")
    alias_key = _theme_identity(category, alias_label)
    canonical_key = _theme_identity(category, canonical_label)
    if alias_key == canonical_key:
        raise ValueError("統合前と統合先のテーマ名が同じです。")

    now = datetime.now(timezone.utc).isoformat()
    with _database(database_path) as connection:
        known_canonical_keys = {
            _theme_identity(theme["category"], theme["label"])
            for theme in _confirmed_theme_labels(connection)
        }
        if canonical_key not in known_canonical_keys:
            raise ValueError("統合先は人が確認済みのテーマから選んでください。")

        old_row = connection.execute(
            "SELECT * FROM theme_aliases WHERE alias_key=?",
            (alias_key,),
        ).fetchone()
        before = _row_dict(old_row) if old_row else None
        after = {
            "alias_key": alias_key,
            "category": category,
            "alias_label": alias_label,
            "canonical_label": canonical_label,
            "rationale": rationale,
            "active": 1,
            "updated_at": now,
        }
        aliases = _active_theme_aliases(connection)
        aliases[alias_key] = after
        _resolve_theme_label(category, alias_label, aliases)
        if before and all(
            before.get(field) == after[field]
            for field in ("category", "alias_label", "canonical_label", "rationale", "active")
        ):
            return False

        connection.execute(
            """
            INSERT INTO theme_aliases
                (alias_key, category, alias_label, canonical_label, rationale, active, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(alias_key) DO UPDATE SET
                category=excluded.category,
                alias_label=excluded.alias_label,
                canonical_label=excluded.canonical_label,
                rationale=excluded.rationale,
                active=1,
                updated_at=excluded.updated_at
            """,
            (alias_key, category, alias_label, canonical_label, rationale, now),
        )
        connection.execute(
            """
            INSERT INTO theme_alias_history
                (alias_key, previous_json, new_json, reason, changed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                alias_key,
                json.dumps(before, ensure_ascii=False, sort_keys=True) if before else None,
                json.dumps(after, ensure_ascii=False, sort_keys=True),
                "人によるテーマ表記統合",
                now,
            ),
        )
    return True


def remove_theme_alias(database_path, category: str, alias_label: str) -> bool:
    alias_key = _theme_identity(category, alias_label)
    now = datetime.now(timezone.utc).isoformat()
    with _database(database_path) as connection:
        old_row = connection.execute(
            "SELECT * FROM theme_aliases WHERE alias_key=? AND active=1",
            (alias_key,),
        ).fetchone()
        if old_row is None:
            return False
        before = _row_dict(old_row)
        after = {**before, "active": 0, "updated_at": now}
        connection.execute(
            "UPDATE theme_aliases SET active=0, updated_at=? WHERE alias_key=?",
            (now, alias_key),
        )
        connection.execute(
            """
            INSERT INTO theme_alias_history
                (alias_key, previous_json, new_json, reason, changed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                alias_key,
                json.dumps(before, ensure_ascii=False, sort_keys=True),
                json.dumps(after, ensure_ascii=False, sort_keys=True),
                "人によるテーマ表記統合の解除",
                now,
            ),
        )
    return True


def list_theme_alias_history(database_path, limit: int = 500) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("履歴表示件数は1以上にしてください。")
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT history_id, previous_json, new_json, reason, changed_at
            FROM theme_alias_history
            ORDER BY history_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    history = []
    for row in rows:
        before = json.loads(row["previous_json"]) if row["previous_json"] else {}
        after = json.loads(row["new_json"])
        history.append({
            "履歴ID": row["history_id"],
            "変更日時": row["changed_at"],
            "変更理由": row["reason"],
            "区分": after.get("category", before.get("category", "")),
            "表記ゆれ・同義語": after.get("alias_label", before.get("alias_label", "")),
            "変更前統合先": before.get("canonical_label", ""),
            "変更後統合先": after.get("canonical_label", ""),
            "変更後根拠": after.get("rationale", ""),
            "有効": bool(after.get("active", False)),
        })
    return pd.DataFrame(history)


def list_review_questions(
    database_path,
    subject: Optional[str] = None,
    status_field: str = "format_status",
    status: str = "全て",
    query: str = "",
    limit: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    if limit < 1 or offset < 0:
        raise ValueError("表示件数は1以上、開始位置は0以上にしてください。")
    status_columns = {
        "format_status", "knowledge_status", "theme_status", "cross_subject_status",
    }
    if status_field not in status_columns:
        raise ValueError("不明な分類状態です。")
    filters = ["active=1"]
    values = []
    if subject and subject != "全科目":
        filters.append("subject=?")
        values.append(subject)
    if status != "全て":
        if status not in ANNOTATION_STATUSES:
            raise ValueError("不明な分類状態です。")
        filters.append(f"{status_field}=?")
        values.append(status)
    if query.strip():
        filters.append(
            "(problem_id LIKE ? OR major LIKE ? OR middle LIKE ?)"
        )
        pattern = f"%{query.strip()}%"
        values.extend((pattern, pattern, pattern))
    columns = ", ".join(REVIEW_LIST_COLUMNS)
    sql = f"""
        SELECT {columns} FROM question_annotations
        WHERE {' AND '.join(filters)}
        ORDER BY exam_round DESC, subject, problem_number
        LIMIT ? OFFSET ?
    """
    values.extend((limit, offset))
    with _database(database_path) as connection:
        rows = connection.execute(sql, values).fetchall()
    return pd.DataFrame([dict(row) for row in rows], columns=REVIEW_LIST_COLUMNS)


def get_annotation(database_path, problem_id: str) -> dict:
    with _database(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM question_annotations WHERE problem_id=?",
            (problem_id,),
        ).fetchone()
    if row is None:
        raise ValueError("指定した問題の分類データがありません。")
    result = _row_dict(row)
    result["question"] = json.loads(result.pop("question_json"))
    for field in (
        "suggested_secondary_json", "format_secondary_json",
        "themes_json", "related_subjects_json",
    ):
        value_field = field.removesuffix("_json")
        result[value_field] = json.loads(result.pop(field))
    return result


def _annotation_snapshot(row: dict) -> dict:
    return {
        "question_id": row["problem_id"],
        "question": json.loads(row["question_json"]),
        "format": {
            "primary": row["format_main"],
            "secondary": json.loads(row["format_secondary_json"]),
            "rationale": row["format_rationale"],
            "status": row["format_status"],
        },
        "knowledge_level": {
            "level": row["knowledge_level"],
            "rationale": row["knowledge_rationale"],
            "status": row["knowledge_status"],
        },
        "themes": {
            "values": json.loads(row["themes_json"]),
            "rationale": row["theme_rationale"],
            "status": row["theme_status"],
        },
        "cross_subject": {
            "required": row["cross_subject_required"],
            "related_subjects": json.loads(row["related_subjects_json"]),
            "rationale": row["cross_subject_rationale"],
            "status": row["cross_subject_status"],
        },
    }


def save_annotation(database_path, problem_id: str, values: dict) -> bool:
    format_main = values.get("format_main") or None
    if format_main == "未分類":
        format_main = None
    format_secondary = list(dict.fromkeys(values.get("format_secondary", [])))
    knowledge_level = values.get("knowledge_level") or None
    if knowledge_level == "未分類":
        knowledge_level = None
    format_status = values.get("format_status", "未分類")
    knowledge_status = values.get("knowledge_status", "未分類")
    theme_status = values.get("theme_status", "未分類")
    cross_status = values.get("cross_subject_status", "未分類")
    statuses = (format_status, knowledge_status, theme_status, cross_status)
    if any(status not in ANNOTATION_STATUSES for status in statuses):
        raise ValueError("分類状態の指定が不正です。")
    if format_main and format_main not in FORMATS:
        raise ValueError("問題形式の指定が不正です。")
    if any(value not in FORMATS for value in format_secondary):
        raise ValueError("副分類の指定が不正です。")
    if format_main in format_secondary:
        format_secondary.remove(format_main)
    if knowledge_level and knowledge_level not in KNOWLEDGE_LEVELS:
        raise ValueError("知識要求レベルの指定が不正です。")
    themes = values.get("themes", [])
    for theme in themes:
        if theme.get("category") not in THEME_DOMAINS or not _text(theme.get("label")):
            raise ValueError("テーマ区分またはテーマ名の指定が不正です。")
    rationale_values = {
        "format": _text(values.get("format_rationale")),
        "knowledge": _text(values.get("knowledge_rationale")),
        "theme": _text(values.get("theme_rationale")),
        "cross": _text(values.get("cross_subject_rationale")),
    }
    if format_status in (*CONFIRMED_STATUSES, "AI仮分類") and (
        not format_main or not rationale_values["format"]
    ):
        raise ValueError("形式の分類・根拠を入力してください。")
    if knowledge_status in (*CONFIRMED_STATUSES, "AI仮分類") and (
        not knowledge_level or not rationale_values["knowledge"]
    ):
        raise ValueError("知識要求レベル・根拠を入力してください。")
    if theme_status in CONFIRMED_STATUSES and not rationale_values["theme"]:
        raise ValueError("テーマ分類またはテーマなしの確認根拠を入力してください。")
    required = values.get("cross_subject_required")
    if required not in (None, True, False):
        raise ValueError("科目横断の必要性は必要・不要・未判定から選んでください。")
    related_subjects = list(dict.fromkeys(values.get("related_subjects", [])))

    with _database(database_path) as connection:
        old = connection.execute(
            "SELECT * FROM question_annotations WHERE problem_id=? AND active=1",
            (problem_id,),
        ).fetchone()
        if old is None:
            raise ValueError("指定した問題が未同期か、現在の入力データにありません。")
        source_subject = old["subject"]
        allowed_subjects = {
            row["subject"]
            for row in connection.execute(
                "SELECT DISTINCT subject FROM question_annotations WHERE active=1"
            ).fetchall()
        }
        if source_subject in related_subjects:
            raise ValueError("関連科目に問題の出題科目自身は指定できません。")
        if any(subject not in allowed_subjects for subject in related_subjects):
            raise ValueError("関連科目に過去問データにない科目が含まれています。")
        if cross_status in CONFIRMED_STATUSES:
            if required is None or not rationale_values["cross"]:
                raise ValueError("科目横断の必要性と判断根拠を入力してください。")
            if required and not related_subjects:
                raise ValueError("複数科目の知識が必要な場合は関連科目を指定してください。")
            if not required and related_subjects:
                raise ValueError("科目横断が不要の場合、関連科目を空にしてください。")

        before = _row_dict(old)
        after = {
            **before,
            "format_main": format_main,
            "format_secondary_json": json.dumps(format_secondary, ensure_ascii=False),
            "format_rationale": rationale_values["format"],
            "format_status": format_status,
            "knowledge_level": knowledge_level,
            "knowledge_rationale": rationale_values["knowledge"],
            "knowledge_status": knowledge_status,
            "themes_json": json.dumps(themes, ensure_ascii=False, sort_keys=True),
            "theme_rationale": rationale_values["theme"],
            "theme_status": theme_status,
            "cross_subject_required": (
                None if required is None else int(required)
            ),
            "related_subjects_json": json.dumps(related_subjects, ensure_ascii=False),
            "cross_subject_rationale": rationale_values["cross"],
            "cross_subject_status": cross_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        changed = any(before[field] != after[field] for field in after if field != "updated_at")
        if not changed:
            return False
        _insert_history(
            connection,
            problem_id,
            _annotation_snapshot(before),
            _annotation_snapshot(after),
            "人による分類・修正",
            after["updated_at"],
        )
        fields = [
            "format_main", "format_secondary_json", "format_rationale", "format_status",
            "knowledge_level", "knowledge_rationale", "knowledge_status", "themes_json",
            "theme_rationale", "theme_status", "cross_subject_required",
            "related_subjects_json", "cross_subject_rationale", "cross_subject_status",
            "updated_at",
        ]
        assignments = ", ".join(f"{field}=?" for field in fields)
        connection.execute(
            f"UPDATE question_annotations SET {assignments} WHERE problem_id=?",
            tuple(after[field] for field in fields) + (problem_id,),
        )
    return True


def list_annotation_history(
    database_path, limit: int = 500, problem_id: Optional[str] = None
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("履歴表示件数は1以上にしてください。")
    with _database(database_path) as connection:
        if problem_id:
            rows = connection.execute(
                """
                SELECT history_id, problem_id, previous_json, new_json, reason, changed_at
                FROM question_annotation_history
                WHERE problem_id=?
                ORDER BY history_id DESC
                LIMIT ?
                """,
                (problem_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT history_id, problem_id, previous_json, new_json, reason, changed_at
                FROM question_annotation_history
                ORDER BY history_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    history = []
    for row in rows:
        before = json.loads(row["previous_json"]) if row["previous_json"] else {}
        after = json.loads(row["new_json"])
        before_question = before.get("question", {})
        after_question = after.get("question", {})
        before_format = before.get("format", {})
        after_format = after.get("format", {})
        before_knowledge = before.get("knowledge_level", {})
        after_knowledge = after.get("knowledge_level", {})
        before_themes = before.get("themes", {})
        after_themes = after.get("themes", {})
        before_cross = before.get("cross_subject", {})
        after_cross = after.get("cross_subject", {})
        history.append({
            "履歴ID": row["history_id"],
            "問題ID": row["problem_id"],
            "変更日時": row["changed_at"],
            "変更理由": row["reason"],
            "変更前問題文": before_question.get("問題", ""),
            "変更後問題文": after_question.get("問題", ""),
            "変更前選択肢": " / ".join(before_question.get("選択肢", [])),
            "変更後選択肢": " / ".join(after_question.get("選択肢", [])),
            "変更前模範解答": before_question.get("模範解答", ""),
            "変更後模範解答": after_question.get("模範解答", ""),
            "変更前正答": before_question.get("正答", ""),
            "変更後正答": after_question.get("正答", ""),
            "変更前図参照": before_question.get("図", ""),
            "変更後図参照": after_question.get("図", ""),
            "変更前症例欄": before_question.get("症例問題", ""),
            "変更後症例欄": after_question.get("症例問題", ""),
            "変更前形式": before_format.get("primary", ""),
            "変更後形式": after_format.get("primary", ""),
            "変更前副分類": "、".join(before_format.get("secondary", [])),
            "変更後副分類": "、".join(after_format.get("secondary", [])),
            "変更前形式根拠": before_format.get("rationale", ""),
            "変更後形式根拠": after_format.get("rationale", ""),
            "変更前形式状態": before_format.get("status", ""),
            "変更後形式状態": after_format.get("status", ""),
            "変更前知識レベル": before_knowledge.get("level", ""),
            "変更後知識レベル": after_knowledge.get("level", ""),
            "変更前知識レベル根拠": before_knowledge.get("rationale", ""),
            "変更後知識レベル根拠": after_knowledge.get("rationale", ""),
            "変更前知識レベル状態": before_knowledge.get("status", ""),
            "変更後知識レベル状態": after_knowledge.get("status", ""),
            "変更前テーマ": "、".join(
                f"{theme['category']}: {theme['label']}"
                for theme in before_themes.get("values", [])
            ),
            "変更後テーマ": "、".join(
                f"{theme['category']}: {theme['label']}"
                for theme in after_themes.get("values", [])
            ),
            "変更前テーマ根拠": before_themes.get("rationale", ""),
            "変更後テーマ根拠": after_themes.get("rationale", ""),
            "変更前テーマ状態": before_themes.get("status", ""),
            "変更後テーマ状態": after_themes.get("status", ""),
            "変更前科目横断": before_cross.get("required"),
            "変更後科目横断": after_cross.get("required"),
            "変更前関連科目": "、".join(before_cross.get("related_subjects", [])),
            "変更後関連科目": "、".join(after_cross.get("related_subjects", [])),
            "変更前科目横断根拠": before_cross.get("rationale", ""),
            "変更後科目横断根拠": after_cross.get("rationale", ""),
            "変更前科目横断状態": before_cross.get("status", ""),
            "変更後科目横断状態": after_cross.get("status", ""),
        })
    return pd.DataFrame(history)


def _decode_row(row) -> dict:
    record = _row_dict(row)
    record["question"] = json.loads(record.pop("question_json"))
    record["themes"] = json.loads(record.pop("themes_json"))
    record["related_subjects"] = json.loads(record.pop("related_subjects_json"))
    record["format_secondary"] = json.loads(record.pop("format_secondary_json"))
    record["suggested_secondary"] = json.loads(record.pop("suggested_secondary_json"))
    return record


def _category_summary(
    records: list[dict],
    value_field: str,
    status_field: str,
    categories: list[str],
    denominator: str,
) -> pd.DataFrame:
    confirmed = [row for row in records if row[status_field] in CONFIRMED_STATUSES]
    provisional = [row for row in records if row[status_field] == "AI仮分類"]
    unclassified = [row for row in records if row[status_field] == "未分類"]
    needs_review = [row for row in records if row[status_field] == "再確認待ち"]
    rows = []
    for category in categories:
        selected = [row for row in confirmed if row[value_field] == category]
        tentative = [row for row in provisional if row[value_field] == category]
        rows.append({
            "分類": category,
            "対象問題数": len(records),
            "人間確認済み問題数": len(selected),
            "確認済み分類数": len(confirmed),
            "確定分類内構成比(%)": (
                round(len(selected) / len(confirmed) * 100, 1) if confirmed else None
            ),
            "AI仮分類問題数": len(tentative),
            "未分類問題数": len(unclassified),
            "再確認待ち問題数": len(needs_review),
            "仮分類問題ID": "、".join(row["problem_id"] for row in tentative),
            "確定分類問題ID": "、".join(row["problem_id"] for row in selected),
            "構成比の分母": denominator,
        })
    return pd.DataFrame(rows)


def _period_distribution(
    records: list[dict],
    status_field: str,
    value_field: str,
    categories: list[str],
    latest_round: int,
    period_start: int,
    period_end: int,
    category_name: str,
) -> list[dict]:
    periods = [
        (f"選択期間（第{period_start}〜{period_end}回）", period_start, period_end),
        (
            f"直近5回（第{max(1, latest_round - 4)}〜{latest_round}回）",
            max(1, latest_round - 4),
            latest_round,
        ),
        (
            f"直近10回（第{max(1, latest_round - 9)}〜{latest_round}回）",
            max(1, latest_round - 9),
            latest_round,
        ),
    ]
    output = []
    for period_label, low, high in periods:
        period_records = [row for row in records if low <= row["exam_round"] <= high]
        confirmed = [
            row for row in period_records if row[status_field] in CONFIRMED_STATUSES
        ]
        provisional = [row for row in period_records if row[status_field] == "AI仮分類"]
        unclassified = [row for row in period_records if row[status_field] == "未分類"]
        needs_review = [
            row for row in period_records if row[status_field] == "再確認待ち"
        ]
        observed_rounds = {row["exam_round"] for row in period_records}
        for category in categories:
            selected = [row for row in confirmed if row[value_field] == category]
            tentative = [row for row in provisional if row[value_field] == category]
            output.append({
                "期間": period_label,
                "対象回数": high - low + 1,
                "収録回数": len(observed_rounds),
                "未収録回数": max(0, high - low + 1 - len(observed_rounds)),
                category_name: category,
                "対象問題数": len(period_records),
                "人間確認済み分類数": len(confirmed),
                "問題数": len(selected),
                "確定分類内構成比(%)": (
                    round(len(selected) / len(confirmed) * 100, 1) if confirmed else None
                ),
                "AI仮分類問題数": len(tentative),
                "未分類問題数": len(unclassified),
                "再確認待ち問題数": len(needs_review),
                "確定分類の問題ID": "、".join(row["problem_id"] for row in selected),
            })
    return output


def _matrix(subjects: list[str], edges: dict[tuple[str, str], int]) -> pd.DataFrame:
    table = {subject: {column: 0 for column in subjects} for subject in subjects}
    for (subject_a, subject_b), count in edges.items():
        table[subject_a][subject_b] = count
        table[subject_b][subject_a] = count
    return pd.DataFrame(
        [{"科目": subject, **table[subject]} for subject in subjects],
        columns=["科目", *subjects],
    )


def build_question_analysis(
    database_path,
    period_start: int,
    period_end: int,
    subject_filter: str = "全科目",
) -> dict[str, pd.DataFrame]:
    if period_start > period_end:
        raise ValueError("分析開始回は終了回以前にしてください。")
    with _database(database_path) as connection:
        all_rows = connection.execute(
            "SELECT * FROM question_annotations WHERE active=1"
        ).fetchall()
        theme_aliases = _active_theme_aliases(connection)
    decoded = [_decode_row(row) for row in all_rows]
    all_period_records = [
        row for row in decoded
        if period_start <= row["exam_round"] <= period_end
    ]
    classification_records = (
        all_period_records
        if subject_filter == "全科目"
        else [row for row in all_period_records if row["subject"] == subject_filter]
    )
    subjects = sorted({row["subject"] for row in all_period_records})
    latest_round = max(
        (row["exam_round"] for row in all_period_records), default=period_end
    )

    form_summary = _category_summary(
        classification_records, "format_main", "format_status", FORMATS,
        "問題形式の人間確認済み・修正済み問題数",
    )
    level_summary = _category_summary(
        classification_records, "knowledge_level", "knowledge_status", KNOWLEDGE_LEVELS,
        "知識要求レベルの人間確認済み・修正済み問題数",
    ).rename(columns={"分類": "知識要求レベル"})
    form_periods = _period_distribution(
        classification_records, "format_status", "format_main", FORMATS,
        latest_round, period_start, period_end, "問題形式",
    )
    level_periods = _period_distribution(
        classification_records, "knowledge_status", "knowledge_level", KNOWLEDGE_LEVELS,
        latest_round, period_start, period_end, "知識要求レベル",
    )

    form_year_rows = []
    level_year_rows = []
    for subject in sorted({row["subject"] for row in classification_records}):
        for year in range(period_start, period_end + 1):
            year_records = [
                row for row in classification_records
                if row["subject"] == subject and row["exam_round"] == year
            ]
            for value_field, status_field, categories, output in (
                ("format_main", "format_status", FORMATS, form_year_rows),
                ("knowledge_level", "knowledge_status", KNOWLEDGE_LEVELS, level_year_rows),
            ):
                confirmed = [
                    row for row in year_records
                    if row[status_field] in CONFIRMED_STATUSES
                ]
                for category in categories:
                    selected = [
                        row for row in confirmed if row[value_field] == category
                    ]
                    output.append({
                        "科目": subject,
                        "回": year,
                        "収録状態": "収録あり" if year_records else "未収録",
                        "分類": category,
                        "問題数": len(selected) if year_records else None,
                        "確認済み分類数": len(confirmed) if year_records else None,
                        "確定分類内構成比(%)": (
                            round(len(selected) / len(confirmed) * 100, 1)
                            if confirmed else None
                        ),
                        "問題ID": "、".join(row["problem_id"] for row in selected),
                    })

    by_subject = []
    for subject in subjects:
        subject_records = [row for row in all_period_records if row["subject"] == subject]
        by_subject.append({
            "科目": subject,
            "対象問題数": len(subject_records),
            "形式の確定分類数": sum(
                row["format_status"] in CONFIRMED_STATUSES for row in subject_records
            ),
            "形式のAI仮分類数": sum(
                row["format_status"] == "AI仮分類" for row in subject_records
            ),
            "知識レベルの確定分類数": sum(
                row["knowledge_status"] in CONFIRMED_STATUSES for row in subject_records
            ),
            "テーマの確定分類数": sum(
                row["theme_status"] in CONFIRMED_STATUSES for row in subject_records
            ),
            "科目横断の確定判断数": sum(
                row["cross_subject_status"] in CONFIRMED_STATUSES
                for row in subject_records
            ),
        })

    theme_groups = {}
    for row in all_period_records:
        if row["theme_status"] not in CONFIRMED_STATUSES:
            continue
        for theme in row["themes"]:
            label = _resolve_theme_label(
                theme["category"], theme["label"], theme_aliases
            )
            key = (theme["category"], _normalized(label))
            group = theme_groups.setdefault(
                key, {"label": label, "subjects": {}}
            )
            group["label"] = min(group["label"], label)
            group["subjects"].setdefault(row["subject"], []).append(row)

    theme_rank_rows = []
    theme_year_rows = []
    theme_subject_edges = {}
    item_edges = {}
    for (domain, _), group in sorted(theme_groups.items()):
        label = group["label"]
        subject_groups = group["subjects"]
        distinct_subjects = sorted(subject_groups)
        total_rows = [row for grouped in subject_groups.values() for row in grouped]
        if len(distinct_subjects) < 2:
            continue
        theme_rank_rows.append({
            "テーマ区分": domain,
            "共通テーマ": label,
            "科目数": len(distinct_subjects),
            "科目": "、".join(distinct_subjects),
            "問題数": len({row["problem_id"] for row in total_rows}),
            "問題ID": "、".join(sorted({row["problem_id"] for row in total_rows})),
            "分類根拠": " / ".join(sorted({
                row["theme_rationale"] for row in total_rows if row["theme_rationale"]
            })),
        })
        for subject, grouped_rows in subject_groups.items():
            by_year = {}
            for row in grouped_rows:
                by_year.setdefault(row["exam_round"], []).append(row)
            for year, year_rows in by_year.items():
                theme_year_rows.append({
                    "テーマ区分": domain,
                    "共通テーマ": label,
                    "科目": subject,
                    "回": year,
                    "問題数": len({row["problem_id"] for row in year_rows}),
                    "問題ID": "、".join(sorted({row["problem_id"] for row in year_rows})),
                    "分類根拠": " / ".join(sorted({
                        row["theme_rationale"] for row in year_rows
                        if row["theme_rationale"]
                    })),
                })

        for index, subject_a in enumerate(distinct_subjects):
            for subject_b in distinct_subjects[index + 1:]:
                edge_key = (subject_a, subject_b)
                theme_subject_edges.setdefault(edge_key, set()).add((domain, label))
                rows_a = subject_groups[subject_a]
                rows_b = subject_groups[subject_b]
                item_groups_a = {}
                item_groups_b = {}
                rationale_by_id_a = {}
                rationale_by_id_b = {}
                for row in rows_a:
                    item_groups_a.setdefault(
                        (row["major"], row["middle"]), set()
                    ).add(row["problem_id"])
                    rationale_by_id_a[row["problem_id"]] = row["theme_rationale"]
                for row in rows_b:
                    item_groups_b.setdefault(
                        (row["major"], row["middle"]), set()
                    ).add(row["problem_id"])
                    rationale_by_id_b[row["problem_id"]] = row["theme_rationale"]
                for item_a, ids_a in item_groups_a.items():
                    for item_b, ids_b in item_groups_b.items():
                        item_key = (subject_a, item_a[0], item_a[1],
                                    subject_b, item_b[0], item_b[1])
                        edge = item_edges.setdefault(
                            item_key,
                            {
                                "themes": set(), "ids_a": set(), "ids_b": set(),
                                "rationales": set(),
                            },
                        )
                        edge["themes"].add(f"{domain}: {label}")
                        edge["ids_a"].update(ids_a)
                        edge["ids_b"].update(ids_b)
                        edge["rationales"].update(
                            rationale_by_id_a[problem_id] for problem_id in ids_a
                            if rationale_by_id_a[problem_id]
                        )
                        edge["rationales"].update(
                            rationale_by_id_b[problem_id] for problem_id in ids_b
                            if rationale_by_id_b[problem_id]
                        )

    theme_matrix_edges = {
        pair: len(theme_set) for pair, theme_set in theme_subject_edges.items()
    }
    theme_network_rows = []
    for (subject_a, major_a, middle_a, subject_b, major_b, middle_b), edge in sorted(
        item_edges.items()
    ):
        theme_network_rows.append({
            "科目A": subject_a,
            "大項目A": major_a,
            "中項目A": middle_a,
            "科目B": subject_b,
            "大項目B": major_b,
            "中項目B": middle_b,
            "共通確認済みテーマ数": len(edge["themes"]),
            "共通確認済みテーマ": "、".join(sorted(edge["themes"])),
            "問題IDA": "、".join(sorted(edge["ids_a"])),
            "問題IDB": "、".join(sorted(edge["ids_b"])),
            "分類根拠": " / ".join(sorted(edge["rationales"])),
        })

    cross_required_rows = []
    for row in all_period_records:
        if (
            row["cross_subject_status"] not in CONFIRMED_STATUSES
            or row["cross_subject_required"] != 1
        ):
            continue
        for related_subject in row["related_subjects"]:
            cross_required_rows.append({
                "科目": row["subject"],
                "関連科目": related_subject,
                "回": row["exam_round"],
                "問題ID": row["problem_id"],
                "大項目": row["major"],
                "中項目": row["middle"],
                "判断根拠": row["cross_subject_rationale"],
                "確認済みテーマ": "、".join(
                    f"{theme['category']}: {theme['label']}"
                    for theme in row["themes"]
                    if row["theme_status"] in CONFIRMED_STATUSES
                ),
            })
    required_edges = {}
    required_year_rows = {}
    for row in cross_required_rows:
        pair = tuple(sorted((row["科目"], row["関連科目"])))
        required_edges.setdefault(pair, set()).add(row["問題ID"])
        by_year_key = (*pair, row["回"])
        required_year_rows.setdefault(by_year_key, set()).add(row["問題ID"])

    required_yearly = [
        {
            "科目A": subject_a,
            "科目B": subject_b,
            "回": year,
            "人間確認済み科目横断問題数": len(problem_ids),
            "問題ID": "、".join(sorted(problem_ids)),
        }
        for (subject_a, subject_b, year), problem_ids in sorted(required_year_rows.items())
    ]
    required_matrix_edges = {
        pair: len(problem_ids) for pair, problem_ids in required_edges.items()
    }
    cross_subjects = sorted(
        set(subjects)
        | {row["関連科目"] for row in cross_required_rows}
    )

    confirmed_format_count = sum(
        row["format_status"] in CONFIRMED_STATUSES for row in classification_records
    )
    confirmed_level_count = sum(
        row["knowledge_status"] in CONFIRMED_STATUSES for row in classification_records
    )
    conditions = pd.DataFrame([
        {"項目": "対象回", "内容": f"第{period_start}〜{period_end}回"},
        {"項目": "分類・知識レベルの科目フィルター", "内容": subject_filter},
        {"項目": "分類・知識レベルの対象問題数", "内容": str(len(classification_records))},
        {"項目": "テーマ・科目横断分析の全科目対象問題数", "内容": str(len(all_period_records))},
        {"項目": "有効なテーマ表記統合ルール数", "内容": str(len(theme_aliases))},
        {"項目": "問題形式構成比の分母", "内容": f"形式の人間確認済み・修正済み問題数 {confirmed_format_count}"},
        {"項目": "知識レベル構成比の分母", "内容": f"知識レベルの人間確認済み・修正済み問題数 {confirmed_level_count}"},
        {"項目": "テーマ分析の対象", "内容": "全科目の人間確認済み・修正済みテーマ（問題IDと根拠を保持）"},
        {"項目": "科目横断必須の対象", "内容": "必要性・関連科目・根拠を人が確認済みの問題のみ"},
        {"項目": "語句一致との区別", "内容": "既存の文字列一致はこの集計に含めず、探索指標として維持"},
        {"項目": "自動分類", "内容": "形式のルール候補はAI仮分類。確認済み集計には含めない"},
    ])
    format_status_counts = pd.DataFrame([
        {
            "分類対象問題数": len(classification_records),
            "形式の人間確認済み・修正済み": confirmed_format_count,
            "形式のAI仮分類": sum(
                row["format_status"] == "AI仮分類" for row in classification_records
            ),
            "形式の再確認待ち": sum(
                row["format_status"] == "再確認待ち" for row in classification_records
            ),
            "知識レベルの人間確認済み・修正済み": confirmed_level_count,
            "テーマの人間確認済み・修正済み": sum(
                row["theme_status"] in CONFIRMED_STATUSES for row in classification_records
            ),
            "科目横断判断の人間確認済み・修正済み": sum(
                row["cross_subject_status"] in CONFIRMED_STATUSES
                for row in classification_records
            ),
        }
    ])
    review_detail = pd.DataFrame([
        {
            "問題ID": row["problem_id"],
            "回": row["exam_round"],
            "科目": row["subject"],
            "大項目": row["major"],
            "中項目": row["middle"],
            "問題形式": row["format_main"] or "",
            "形式副分類": "、".join(row["format_secondary"]),
            "形式状態": row["format_status"],
            "形式根拠": row["format_rationale"],
            "知識要求レベル": row["knowledge_level"] or "",
            "知識レベル状態": row["knowledge_status"],
            "知識レベル根拠": row["knowledge_rationale"],
            "テーマ": "、".join(
                f"{theme['category']}: {theme['label']}" for theme in row["themes"]
            ),
            "テーマ状態": row["theme_status"],
            "テーマ根拠": row["theme_rationale"],
            "科目横断必須": (
                "未判定" if row["cross_subject_required"] is None
                else "必要" if row["cross_subject_required"] else "不要"
            ),
            "関連科目": "、".join(row["related_subjects"]),
            "科目横断状態": row["cross_subject_status"],
            "科目横断根拠": row["cross_subject_rationale"],
        }
        for row in all_period_records
    ])
    return {
        "分析条件": conditions,
        "分類状況": format_status_counts,
        "科目別レビュー進捗": pd.DataFrame(by_subject),
        "問題別分類・確認詳細": review_detail,
        "問題形式別構成": form_summary,
        "年度別問題形式": pd.DataFrame(form_year_rows),
        "期間別問題形式": pd.DataFrame(form_periods),
        "知識レベル別構成": level_summary,
        "年度別知識レベル": pd.DataFrame(level_year_rows),
        "期間別知識レベル": pd.DataFrame(level_periods),
        "共通テーマランキング": pd.DataFrame(theme_rank_rows),
        "科目間共通テーマ行列": _matrix(
            subjects, theme_matrix_edges
        ),
        "中項目間の共通テーマネットワーク": pd.DataFrame(theme_network_rows),
        "共通テーマ年度別推移": pd.DataFrame(theme_year_rows),
        "テーマ表記統合ルール": pd.DataFrame([
            {
                "テーマ区分": alias["category"],
                "統合前の表記": alias["alias_label"],
                "統合先テーマ": alias["canonical_label"],
                "判断根拠": alias["rationale"],
            }
            for alias in sorted(
                theme_aliases.values(),
                key=lambda item: (item["category"], item["alias_label"]),
            )
        ], columns=[
            "テーマ区分", "統合前の表記", "統合先テーマ", "判断根拠"
        ]),
        "科目横断が必要な確認済み問題": pd.DataFrame(cross_required_rows),
        "科目横断必須問題数行列": _matrix(
            cross_subjects, required_matrix_edges
        ),
        "科目横断必須の年度別推移": pd.DataFrame(required_yearly),
    }


def _theme_candidate_vocabulary() -> tuple[list[str], set[str]]:
    root = Path(__file__).parent
    dictionary = json.loads((root / "dictionary.json").read_text(encoding="utf-8"))
    important_terms = dictionary.get("重要語(疾患・所見)", {})
    curated = []
    medical_terms_path = root / "medical_terms.txt"
    if medical_terms_path.exists():
        curated.extend(
            line.strip()
            for line in medical_terms_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    curated.extend(important_terms)
    stop = {
        terms.norm(word)
        for word in dictionary.get("除外語", [])
        if terms.norm(word)
    }
    return curated, stop


def _extract_theme_candidates(
    records: list[dict],
    candidate_limit: int = 10,
    curated: Optional[list[str]] = None,
    stop: Optional[set[str]] = None,
) -> pd.DataFrame:
    columns = [
        "科目", "大項目", "中項目", "順位", "語句候補", "出現問題数",
        "対象問題数", "問題ID", "候補状態", "抽出上の注意",
    ]
    if candidate_limit < 1:
        raise ValueError("テーマ候補の表示数は1以上にしてください。")
    usable_records = [
        row for row in records
        if row["middle"] not in UNCLASSIFIED_MIDDLE_LABELS
    ]
    if not usable_records:
        return pd.DataFrame(columns=columns)
    if curated is None or stop is None:
        default_curated, default_stop = _theme_candidate_vocabulary()
        curated = default_curated if curated is None else curated
        stop = default_stop if stop is None else stop

    question_groups = {}
    for row in usable_records:
        key = (row["subject"], row["major"])
        question = row["question"]
        choices = question.get("選択肢", [])
        question_record = {
            "回": row["exam_round"],
            "問題ID": row["problem_id"],
            "大項目": row["major"],
            "中項目": row["middle"],
            "問題": question.get("問題", ""),
            "模範解答": question.get("模範解答", ""),
            "正答": question.get("正答", ""),
        }
        question_record.update({
            f"選択肢{index}": choices[index - 1] if index <= len(choices) else ""
            for index in range(1, len(CHOICE_COLUMNS) + 1)
        })
        question_groups.setdefault(key, []).append(question_record)

    result = []
    for (subject, major), group_records in sorted(question_groups.items()):
        questions = pd.DataFrame(group_records)
        middle_items = sorted(questions["中項目"].dropna().astype(str).unique())
        exact_stop = {terms.norm(major), *(terms.norm(item) for item in middle_items)}
        candidates = terms.topic_frequent_terms(
            questions,
            middle_items,
            curated,
            stop,
            exact_stop,
            top=candidate_limit,
        )
        for candidate in candidates.to_dict(orient="records"):
            result.append({
                "科目": subject,
                "大項目": major,
                "中項目": candidate["中項目"],
                "順位": candidate["順位"],
                "語句候補": candidate["語"],
                "出現問題数": candidate["出現問題数"],
                "対象問題数": candidate["対象問題数"],
                "問題ID": candidate["問題ID"],
                "候補状態": "未確認候補",
                "抽出上の注意": (
                    "問題文・全選択肢の頻出語。誤答にも現れ、問題の主題とは限らない。"
                ),
            })
    return pd.DataFrame(result, columns=columns)


def _theme_trend_status(
    has_past: bool,
    has_recent: bool,
    past_coverage: int,
    recent_coverage: int,
    delta_pp: Optional[float],
) -> str:
    if not has_past and not has_recent:
        return "比較保留（完全レビュー済み回での出題なし）"
    if has_past and has_recent:
        if (
            past_coverage < THEME_MIN_COMPARISON_ROUNDS
            or recent_coverage < THEME_MIN_COMPARISON_ROUNDS
        ):
            return "継続（比較データ不足）"
        if delta_pp is not None and delta_pp >= THEME_TREND_THRESHOLD_PP:
            return "増加"
        if delta_pp is not None and delta_pp <= -THEME_TREND_THRESHOLD_PP:
            return "減少"
        return "継続（横ばい）"
    if past_coverage < THEME_MIN_COMPARISON_ROUNDS:
        return "比較保留（過去期間のレビュー回数不足）"
    if recent_coverage < THEME_MIN_COMPARISON_ROUNDS:
        return "比較保留（直近期間のレビュー回数不足）"
    if has_recent and not has_past:
        return "新出候補"
    if has_past and not has_recent:
        return "減少（直近期間に出題なし）"
    return "比較保留"


def build_theme_change_analysis(
    database_path,
    period_start: int,
    period_end: int,
    candidate_limit: int = 10,
) -> dict[str, pd.DataFrame]:
    if period_start > period_end:
        raise ValueError("分析開始回は終了回以前にしてください。")
    if candidate_limit < 1:
        raise ValueError("テーマ候補の表示数は1以上にしてください。")
    with _database(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM question_annotations WHERE active=1"
        ).fetchall()
        aliases = _active_theme_aliases(connection)

    records = [_decode_row(row) for row in rows]
    period_records = [
        row for row in records
        if period_start <= row["exam_round"] <= period_end
    ]
    topic_records = [
        row for row in period_records
        if row["middle"] not in UNCLASSIFIED_MIDDLE_LABELS
    ]
    latest_round = max(
        (row["exam_round"] for row in period_records), default=period_end
    )
    recent_start = max(period_start, latest_round - 9)
    past_end = recent_start - 1
    recent_end = latest_round
    item_round_data = {}
    item_theme_labels = {}
    for row in topic_records:
        item_key = (row["subject"], row["major"], row["middle"])
        year_data = item_round_data.setdefault(item_key, {}).setdefault(
            row["exam_round"],
            {"all_ids": set(), "reviewed_ids": set(), "themes": {}},
        )
        year_data["all_ids"].add(row["problem_id"])
        if row["theme_status"] not in CONFIRMED_STATUSES:
            continue
        year_data["reviewed_ids"].add(row["problem_id"])
        for theme in row["themes"]:
            canonical_label = _resolve_theme_label(
                theme["category"], theme["label"], aliases
            )
            theme_key = (theme["category"], _normalized(canonical_label))
            label_map = item_theme_labels.setdefault(item_key, {})
            label_map[theme_key] = min(
                label_map.get(theme_key, canonical_label), canonical_label
            )
            year_data["themes"].setdefault(theme_key, set()).add(row["problem_id"])

    annual_rows = []
    trend_rows = []
    for item_key in sorted(item_round_data):
        subject, major, middle = item_key
        year_map = item_round_data[item_key]
        labels = item_theme_labels.get(item_key, {})
        complete_rounds = {
            year for year, year_data in year_map.items()
            if year_data["all_ids"]
            and year_data["reviewed_ids"] == year_data["all_ids"]
        }
        covered_past_rounds = sorted(
            year for year in complete_rounds if period_start <= year <= past_end
        )
        covered_recent_rounds = sorted(
            year for year in complete_rounds if recent_start <= year <= recent_end
        )

        for year in sorted(year_map):
            year_data = year_map[year]
            review_complete = (
                bool(year_data["all_ids"])
                and year_data["reviewed_ids"] == year_data["all_ids"]
            )
            for theme_key, label in sorted(labels.items()):
                tagged_ids = year_data["themes"].get(theme_key, set())
                annual_rows.append({
                    "科目": subject,
                    "大項目": major,
                    "中項目": middle,
                    "回": year,
                    "テーマ区分": theme_key[0],
                    "テーマ": label,
                    "テーマ表示名": f"{theme_key[0]}：{label}",
                    "テーマ確認状態": (
                        "全問題を確認済み" if review_complete else "未完了"
                    ),
                    "対象問題数": len(year_data["all_ids"]),
                    "テーマ確認済み問題数": len(year_data["reviewed_ids"]),
                    "テーマ出題問題数": len(tagged_ids) if review_complete else None,
                    "テーマ出題割合(%)": (
                        round(len(tagged_ids) / len(year_data["all_ids"]) * 100, 1)
                        if review_complete else None
                    ),
                    "問題ID": "、".join(sorted(tagged_ids)),
                })

        for theme_key, label in sorted(labels.items()):
            past_theme_rounds = [
                year for year in covered_past_rounds
                if year_map[year]["themes"].get(theme_key)
            ]
            recent_theme_rounds = [
                year for year in covered_recent_rounds
                if year_map[year]["themes"].get(theme_key)
            ]
            past_ids = {
                problem_id for year in past_theme_rounds
                for problem_id in year_map[year]["themes"][theme_key]
            }
            recent_ids = {
                problem_id for year in recent_theme_rounds
                for problem_id in year_map[year]["themes"][theme_key]
            }
            past_share = (
                len(past_theme_rounds) / len(covered_past_rounds) * 100
                if covered_past_rounds else None
            )
            recent_share = (
                len(recent_theme_rounds) / len(covered_recent_rounds) * 100
                if covered_recent_rounds else None
            )
            delta = (
                round(recent_share - past_share, 1)
                if recent_share is not None and past_share is not None else None
            )
            trend_rows.append({
                "科目": subject,
                "大項目": major,
                "中項目": middle,
                "テーマ区分": theme_key[0],
                "テーマ": label,
                "過去期間": (
                    f"第{period_start}〜{past_end}回"
                    if past_end >= period_start else "比較期間なし"
                ),
                "直近期間": f"第{recent_start}〜{recent_end}回",
                "過去の完全レビュー済み回数": len(covered_past_rounds),
                "直近の完全レビュー済み回数": len(covered_recent_rounds),
                "過去の出題回数": len(past_theme_rounds),
                "直近の出題回数": len(recent_theme_rounds),
                "過去の出題問題数": len(past_ids),
                "直近の出題問題数": len(recent_ids),
                "過去の出題回割合(%)": round(past_share, 1) if past_share is not None else None,
                "直近の出題回割合(%)": round(recent_share, 1) if recent_share is not None else None,
                "出題回割合の差(ポイント)": delta,
                "変化判定": (
                    "比較期間なし"
                    if past_end < period_start
                    else _theme_trend_status(
                        bool(past_ids), bool(recent_ids),
                        len(covered_past_rounds), len(covered_recent_rounds), delta,
                    )
                ),
                "過去の問題ID": "、".join(sorted(past_ids)),
                "直近の問題ID": "、".join(sorted(recent_ids)),
            })

    trend_frame = pd.DataFrame(trend_rows)
    empty_middle_rows = [
        {
            "問題ID": row["problem_id"],
            "回": row["exam_round"],
            "科目": row["subject"],
            "大項目": row["major"],
            "中項目": row["middle"],
        }
        for row in period_records
        if row["middle"] in UNCLASSIFIED_MIDDLE_LABELS
    ]
    candidate_terms = _extract_theme_candidates(
        period_records, candidate_limit=candidate_limit
    )
    alias_frame = pd.DataFrame([
        {
            "テーマ区分": alias["category"],
            "統合前の表記": alias["alias_label"],
            "統合先テーマ": alias["canonical_label"],
            "判断根拠": alias["rationale"],
        }
        for alias in sorted(
            aliases.values(),
            key=lambda item: (item["category"], item["alias_label"]),
        )
    ], columns=["テーマ区分", "統合前の表記", "統合先テーマ", "判断根拠"])
    confirmed_topic_count = sum(
        row["theme_status"] in CONFIRMED_STATUSES for row in topic_records
    )
    unclassified_middle_count = sum(
        row["middle"] in UNCLASSIFIED_MIDDLE_LABELS for row in period_records
    )
    conditions = pd.DataFrame([
        {"項目": "対象回", "内容": f"第{period_start}〜{period_end}回"},
        {"項目": "直近期間", "内容": f"第{recent_start}〜{recent_end}回（最大10回）"},
        {"項目": "過去期間", "内容": f"第{period_start}〜{past_end}回" if past_end >= period_start else "比較期間なし"},
        {"項目": "期間内の対象問題数", "内容": str(len(period_records))},
        {"項目": "中項目分類済み問題数", "内容": str(len(topic_records))},
        {"項目": "テーマ確認済み問題数", "内容": str(confirmed_topic_count)},
        {"項目": "中項目未設定・記載なしの問題数", "内容": str(unclassified_middle_count)},
        {"項目": "対象", "内容": "中項目が分類済みの過去問。中項目「記載なし」は別表"},
        {"項目": "テーマの出題数", "内容": "人間確認済み・修正済みテーマを問題ID単位で集計"},
        {"項目": "年度別出題割合の分母", "内容": "該当する中項目・出題回の全問題数"},
        {"項目": "期間比較の出題回割合の分母", "内容": "各期間でテーマ確認が完了した出題回数"},
        {"項目": "年度比較の収録条件", "内容": "その中項目・出題回の全問題でテーマ確認が完了した回のみ比較に使用"},
        {"項目": "増減判定", "内容": f"両期間に完全レビュー済み回が各{THEME_MIN_COMPARISON_ROUNDS}回以上ある場合、出題回割合の差{THEME_TREND_THRESHOLD_PP:g}ポイント以上を増減と判定"},
        {"項目": "テーマ語句候補", "内容": f"既存の頻出語抽出を再利用し、中項目ごとに上位{candidate_limit}語。未確認候補として提示し、確定テーマには含めない"},
        {"項目": "候補抽出の注意", "内容": "問題文・全選択肢を対象とし、誤答にも現れる語句を含む。人の確認が必要"},
        {"項目": "表記ゆれ", "内容": "NFKC・空白除去で正規化し、同義語は人が登録した統合ルールのみ適用"},
    ])
    decrease = trend_frame.loc[
        trend_frame["変化判定"].str.startswith("減少")
    ].reset_index(drop=True) if not trend_frame.empty else trend_frame.copy()
    newly_appeared = trend_frame.loc[
        trend_frame["変化判定"].eq("新出候補")
    ].reset_index(drop=True) if not trend_frame.empty else trend_frame.copy()
    continuous = trend_frame.loc[
        trend_frame["変化判定"].str.startswith("継続")
    ].reset_index(drop=True) if not trend_frame.empty else trend_frame.copy()
    return {
        "テーマ変化分析条件": conditions,
        "中項目別テーマ頻度比較": trend_frame,
        "新出テーマ候補": newly_appeared,
        "減少テーマ候補": decrease,
        "継続テーマ": continuous,
        "年度別テーマ出現頻度": pd.DataFrame(annual_rows),
        "テーマ語句候補（未確認）": candidate_terms,
        "中項目未設定問題": pd.DataFrame(
            empty_middle_rows,
            columns=["問題ID", "回", "科目", "大項目", "中項目"],
        ),
        "テーマ表記統合ルール": alias_frame,
    }
