"""Reappearance candidates and review persistence for exam questions."""

import hashlib
import itertools
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd


MIN_SIMILARITY = 0.85
MIN_SHARED_NGRAMS = 4
MIN_SHARED_STEM_NGRAMS = 4
MIN_STEM_SIMILARITY = 0.2
MAX_NGRAM_DOCUMENTS = 80
CHOICE_COLUMNS = ["選択肢1", "選択肢2", "選択肢3", "選択肢4"]
RELATION_COLUMNS = [
    "relation_id", "question_id_a", "question_id_b", "subject_a", "round_a",
    "problem_number_a", "subject_b", "round_b", "problem_number_b",
    "auto_type", "similarity", "evidence", "review_status", "reviewed_type",
    "review_note", "active_a", "active_b",
]
REVIEW_STATUSES = [
    "自動候補", "再確認待ち", "確認済み", "修正済み", "対象外",
]
REVIEW_TYPES = ["A", "B", "C", "対象外"]


def _text(value) -> str:
    return "" if pd.isna(value) else str(value)


def _normalized(value) -> str:
    text = unicodedata.normalize("NFKC", _text(value))
    return re.sub(r"\s+", "", text)


def _number_text(value) -> str:
    if pd.isna(value):
        raise ValueError("問題番号が未入力の問題があります。")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def canonical_problem_id(row: pd.Series, subject: str) -> str:
    year = pd.to_numeric(row.get("回"), errors="coerce")
    if pd.isna(year):
        raise ValueError("出題回を読み取れない問題があります。")
    return f"第{int(year)}回/{subject}/No.{_number_text(row.get('問題番号'))}"


def _make_record(row: pd.Series, subject: str) -> dict:
    problem_id = canonical_problem_id(row, subject)
    year = int(pd.to_numeric(row["回"], errors="raise"))
    choices = [_text(row.get(column)) for column in CHOICE_COLUMNS]
    content = {
        "問題": _text(row.get("問題")),
        "選択肢": choices,
        "模範解答": _text(row.get("模範解答")),
        "正答": _text(row.get("正答")),
    }
    normalized_content = [
        _normalized(content["問題"]),
        *[_normalized(choice) for choice in choices],
        _normalized(content["模範解答"]),
        _normalized(content["正答"]),
    ]
    content_hash = hashlib.sha256(
        json.dumps(normalized_content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    content_text = "|".join(normalized_content)
    return {
        "problem_id": problem_id,
        "round": year,
        "subject": subject,
        "problem_number": _number_text(row.get("問題番号")),
        "major": _text(row.get("大項目")),
        "middle": _text(row.get("中項目")),
        "content_hash": content_hash,
        "content_text": content_text,
        "stem_text": normalized_content[0],
        "snapshot_json": json.dumps(content, ensure_ascii=False, sort_keys=True),
    }


def collect_question_records(question_sheets: dict[str, pd.DataFrame]) -> list[dict]:
    records = []
    for subject, questions in question_sheets.items():
        for _, row in questions.iterrows():
            if pd.isna(pd.to_numeric(row.get("回"), errors="coerce")):
                continue
            records.append(_make_record(row, subject))
    records.sort(key=lambda record: record["problem_id"])
    ids = [record["problem_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("国家試験回・科目・問題番号の組み合わせが重複しています。")
    return records


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
        CREATE TABLE IF NOT EXISTS questions_current (
            problem_id TEXT PRIMARY KEY,
            exam_round INTEGER NOT NULL,
            subject TEXT NOT NULL,
            problem_number TEXT NOT NULL,
            major TEXT,
            middle TEXT,
            content_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS question_versions (
            problem_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            PRIMARY KEY (problem_id, content_hash)
        );
        CREATE TABLE IF NOT EXISTS question_changes (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT NOT NULL,
            before_hash TEXT,
            after_hash TEXT,
            before_json TEXT,
            after_json TEXT,
            reason TEXT NOT NULL,
            changed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS relation_reviews (
            relation_id TEXT PRIMARY KEY,
            question_id_a TEXT NOT NULL,
            question_id_b TEXT NOT NULL,
            auto_type TEXT,
            similarity REAL,
            evidence TEXT NOT NULL,
            candidate_hash_a TEXT,
            candidate_hash_b TEXT,
            reviewed_hash_a TEXT,
            reviewed_hash_b TEXT,
            review_status TEXT NOT NULL,
            reviewed_type TEXT,
            review_note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (question_id_a, question_id_b)
        );
        CREATE TABLE IF NOT EXISTS relation_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation_id TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            previous_type TEXT,
            new_type TEXT,
            previous_note TEXT,
            new_note TEXT,
            reason TEXT NOT NULL,
            before_hash_a TEXT,
            before_hash_b TEXT,
            after_hash_a TEXT,
            after_hash_b TEXT,
            changed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS relation_reviews_status_idx
            ON relation_reviews (review_status, similarity);
    """)
    history_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(relation_history)").fetchall()
    }
    for column in ("previous_note", "new_note"):
        if column not in history_columns:
            connection.execute(
                f"ALTER TABLE relation_history ADD COLUMN {column} TEXT"
            )


def _relation_id(question_id_a: str, question_id_b: str) -> tuple[str, str, str]:
    id_a, id_b = sorted((question_id_a, question_id_b))
    relation_id = hashlib.sha256(f"{id_a}\0{id_b}".encode("utf-8")).hexdigest()
    return relation_id, id_a, id_b


def _insert_question_version(connection, record: dict, now: str) -> None:
    connection.execute(
        """
        INSERT INTO question_versions
            (problem_id, content_hash, snapshot_json, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(problem_id, content_hash) DO UPDATE SET last_seen_at=excluded.last_seen_at
        """,
        (
            record["problem_id"], record["content_hash"], record["snapshot_json"], now, now,
        ),
    )


def _record_relation_history(
    connection,
    relation: sqlite3.Row,
    new_status: str,
    new_type: Optional[str],
    reason: str,
    after_hash_a: Optional[str] = None,
    after_hash_b: Optional[str] = None,
    note_change: Optional[tuple[Optional[str], Optional[str]]] = None,
) -> None:
    previous_note, new_note = note_change or (
        relation["review_note"], relation["review_note"]
    )
    connection.execute(
        """
        INSERT INTO relation_history
            (relation_id, previous_status, new_status, previous_type, new_type,
             previous_note, new_note, reason, before_hash_a, before_hash_b,
             after_hash_a, after_hash_b, changed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation["relation_id"], relation["review_status"], new_status,
            relation["reviewed_type"], new_type, previous_note, new_note, reason,
            relation["reviewed_hash_a"] or relation["candidate_hash_a"],
            relation["reviewed_hash_b"] or relation["candidate_hash_b"],
            after_hash_a, after_hash_b, datetime.now(timezone.utc).isoformat(),
        ),
    )


def _mark_relations_for_review(
    connection,
    problem_id: str,
    reason: str,
    new_content_hash: Optional[str] = None,
) -> None:
    relations = connection.execute(
        """
        SELECT * FROM relation_reviews
        WHERE question_id_a=? OR question_id_b=?
        """,
        (problem_id, problem_id),
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    for relation in relations:
        after_hashes = []
        for suffix in ("a", "b"):
            relation_problem_id = relation[f"question_id_{suffix}"]
            if relation_problem_id == problem_id:
                after_hashes.append(new_content_hash)
            else:
                current = connection.execute(
                    "SELECT content_hash FROM questions_current WHERE problem_id=?",
                    (relation_problem_id,),
                ).fetchone()
                after_hashes.append(current["content_hash"] if current else None)
        _record_relation_history(
            connection,
            relation,
            "再確認待ち",
            relation["reviewed_type"],
            reason,
            after_hashes[0],
            after_hashes[1],
        )
        connection.execute(
            "UPDATE relation_reviews SET review_status=?, updated_at=? WHERE relation_id=?",
            ("再確認待ち", now, relation["relation_id"]),
        )


def _ngram_set(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[index:index + 3] for index in range(len(text) - 2)}


def _sequence_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b, autojunk=False).ratio()


def _candidate_pairs(
    records: list[dict], min_similarity: float = MIN_SIMILARITY
) -> list[dict]:
    exact_groups = defaultdict(list)
    for index, record in enumerate(records):
        exact_groups[record["content_hash"]].append(index)

    candidates = {}
    for indices in exact_groups.values():
        if len(indices) < 2:
            continue
        for index_a, index_b in itertools.combinations(indices, 2):
            if records[index_a]["round"] == records[index_b]["round"]:
                continue
            pair = (index_a, index_b)
            candidates[pair] = {
                "auto_type": "A",
                "similarity": 1.0,
                "evidence": "正規化した問題文・選択肢・模範解答・正答が一致",
            }

    feature_records = [_ngram_set(record["content_text"]) for record in records]
    stem_feature_records = [_ngram_set(record["stem_text"]) for record in records]
    inverted = defaultdict(list)
    for index, features in enumerate(feature_records):
        for feature in features:
            inverted[feature].append(index)

    shared_features = defaultdict(int)
    for indices in inverted.values():
        if len(indices) > MAX_NGRAM_DOCUMENTS:
            continue
        for index_a, index_b in itertools.combinations(indices, 2):
            if records[index_a]["round"] == records[index_b]["round"]:
                continue
            if records[index_a]["content_hash"] == records[index_b]["content_hash"]:
                continue
            pair = (index_a, index_b)
            shared_features[pair] += 1

    for (index_a, index_b), shared_count in shared_features.items():
        if shared_count < MIN_SHARED_NGRAMS:
            continue
        score = _sequence_similarity(
            records[index_a]["content_text"], records[index_b]["content_text"]
        )
        if score < min_similarity:
            continue
        stems_a = stem_feature_records[index_a]
        stems_b = stem_feature_records[index_b]
        shared_stem = len(stems_a & stems_b)
        stem_score = _sequence_similarity(
            records[index_a]["stem_text"], records[index_b]["stem_text"]
        )
        if shared_stem < MIN_SHARED_STEM_NGRAMS or stem_score < MIN_STEM_SIMILARITY:
            continue
        candidates[(index_a, index_b)] = {
            "auto_type": "B",
            "similarity": round(score, 4),
            "evidence": (
                f"問題文・選択肢・模範解答・正答の文字列類似度={score:.4f}; "
                f"問題文の文字列類似度={stem_score:.4f}"
            ),
        }

    output = []
    for (index_a, index_b), candidate in candidates.items():
        record_a, record_b = records[index_a], records[index_b]
        relation_id, id_a, id_b = _relation_id(
            record_a["problem_id"], record_b["problem_id"]
        )
        by_id = {
            record_a["problem_id"]: record_a,
            record_b["problem_id"]: record_b,
        }
        first, second = by_id[id_a], by_id[id_b]
        output.append({
            "relation_id": relation_id,
            "question_id_a": id_a,
            "question_id_b": id_b,
            "auto_type": candidate["auto_type"],
            "similarity": candidate["similarity"],
            "evidence": candidate["evidence"],
            "candidate_hash_a": first["content_hash"],
            "candidate_hash_b": second["content_hash"],
        })
    output.sort(
        key=lambda row: (
            row["auto_type"] != "A",
            -(row["similarity"] or 0),
            row["question_id_a"],
            row["question_id_b"],
        )
    )
    return output


def refresh_review_database(
    database_path,
    question_sheets: dict[str, pd.DataFrame],
    min_similarity: float = MIN_SIMILARITY,
    source_signature: Optional[str] = None,
) -> dict:
    if not 0 <= min_similarity <= 1:
        raise ValueError("類似度の下限は0〜1で指定してください。")
    records = collect_question_records(question_sheets)
    candidates = _candidate_pairs(records, min_similarity)
    now = datetime.now(timezone.utc).isoformat()
    changed = 0
    removed = 0
    with _database(database_path) as connection:
        previous = {
            row["problem_id"]: row
            for row in connection.execute(
                "SELECT * FROM questions_current WHERE active=1"
            ).fetchall()
        }
        connection.execute("UPDATE questions_current SET active=0")
        current_ids = set()
        for record in records:
            problem_id = record["problem_id"]
            current_ids.add(problem_id)
            old = previous.get(problem_id)
            _insert_question_version(connection, record, now)
            if old is not None and old["content_hash"] != record["content_hash"]:
                changed += 1
                connection.execute(
                    """
                    INSERT INTO question_changes
                        (problem_id, before_hash, after_hash, before_json, after_json,
                         reason, changed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        problem_id, old["content_hash"], record["content_hash"],
                        old["snapshot_json"], record["snapshot_json"], "問題内容が変更",
                        now,
                    ),
                )
                _mark_relations_for_review(
                    connection, problem_id, "関連問題の内容が変更",
                    record["content_hash"],
                )

            connection.execute(
                """
                INSERT INTO questions_current
                    (problem_id, exam_round, subject, problem_number, major, middle,
                     content_hash, snapshot_json, active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(problem_id) DO UPDATE SET
                    exam_round=excluded.exam_round,
                    subject=excluded.subject,
                    problem_number=excluded.problem_number,
                    major=excluded.major,
                    middle=excluded.middle,
                    content_hash=excluded.content_hash,
                    snapshot_json=excluded.snapshot_json,
                    active=1,
                    updated_at=excluded.updated_at
                """,
                (
                    problem_id, record["round"], record["subject"],
                    record["problem_number"], record["major"], record["middle"],
                    record["content_hash"], record["snapshot_json"], now,
                ),
            )

        for problem_id, old in previous.items():
            if problem_id in current_ids:
                continue
            removed += 1
            connection.execute(
                """
                INSERT INTO question_changes
                    (problem_id, before_hash, after_hash, before_json, after_json,
                     reason, changed_at)
                VALUES (?, ?, NULL, ?, NULL, ?, ?)
                """,
                (
                    problem_id, old["content_hash"], old["snapshot_json"],
                    "問題が入力データから削除", now,
                ),
            )
            _mark_relations_for_review(
                connection, problem_id, "関連問題が入力データから削除"
            )

        for candidate in candidates:
            existing = connection.execute(
                "SELECT * FROM relation_reviews WHERE relation_id=?",
                (candidate["relation_id"],),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO relation_reviews
                        (relation_id, question_id_a, question_id_b, auto_type, similarity,
                         evidence, candidate_hash_a, candidate_hash_b, review_status,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '自動候補', ?, ?)
                    """,
                    (
                        candidate["relation_id"], candidate["question_id_a"],
                        candidate["question_id_b"], candidate["auto_type"],
                        candidate["similarity"], candidate["evidence"],
                        candidate["candidate_hash_a"], candidate["candidate_hash_b"],
                        now, now,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE relation_reviews
                    SET auto_type=?, similarity=?, evidence=?, candidate_hash_a=?,
                        candidate_hash_b=?, updated_at=?
                    WHERE relation_id=?
                    """,
                    (
                        candidate["auto_type"], candidate["similarity"],
                        candidate["evidence"], candidate["candidate_hash_a"],
                        candidate["candidate_hash_b"], now, candidate["relation_id"],
                    ),
                )

        pending_count = connection.execute(
            "SELECT COUNT(*) FROM relation_reviews WHERE review_status='再確認待ち'"
        ).fetchone()[0]
        if source_signature is not None:
            connection.execute(
                """
                INSERT INTO review_metadata (key, value, updated_at)
                VALUES ('source_signature', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at
                """,
                (source_signature, now),
            )

    return {
        "問題数": len(records),
        "完全一致候補数": sum(row["auto_type"] == "A" for row in candidates),
        "類似候補数": sum(row["auto_type"] == "B" for row in candidates),
        "内容変更問題数": changed,
        "削除問題数": removed,
        "再確認待ち関係数": pending_count,
    }


def get_source_signature(database_path) -> Optional[str]:
    with _database(database_path) as connection:
        row = connection.execute(
            "SELECT value FROM review_metadata WHERE key='source_signature'"
        ).fetchone()
    return row["value"] if row else None


def list_review_candidates(
    database_path,
    status: str = "未確認・再確認待ち",
    limit: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    if limit < 1 or offset < 0:
        raise ValueError("表示件数は1以上、開始位置は0以上にしてください。")
    status_filters = {
        "未確認・再確認待ち": ("自動候補", "再確認待ち"),
        "自動候補": ("自動候補",),
        "再確認待ち": ("再確認待ち",),
        "確認済み": ("確認済み",),
        "修正済み": ("修正済み",),
        "対象外": ("対象外",),
    }
    if status != "全て" and status not in status_filters:
        raise ValueError("不明な判定状態です。")

    with _database(database_path) as connection:
        if status == "全て":
            rows = connection.execute(
                """
                SELECT r.*, a.subject AS subject_a, a.exam_round AS round_a,
                       a.problem_number AS problem_number_a, a.active AS active_a,
                       b.subject AS subject_b, b.exam_round AS round_b,
                       b.problem_number AS problem_number_b, b.active AS active_b
                FROM relation_reviews r
                LEFT JOIN questions_current a ON a.problem_id=r.question_id_a
                LEFT JOIN questions_current b ON b.problem_id=r.question_id_b
                ORDER BY CASE r.review_status WHEN '再確認待ち' THEN 0
                         WHEN '自動候補' THEN 1 ELSE 2 END,
                         r.similarity DESC, r.relation_id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        else:
            values = status_filters[status]
            placeholders = ",".join("?" for _ in values)
            rows = connection.execute(
                f"""
                SELECT r.*, a.subject AS subject_a, a.exam_round AS round_a,
                       a.problem_number AS problem_number_a, a.active AS active_a,
                       b.subject AS subject_b, b.exam_round AS round_b,
                       b.problem_number AS problem_number_b, b.active AS active_b
                FROM relation_reviews r
                LEFT JOIN questions_current a ON a.problem_id=r.question_id_a
                LEFT JOIN questions_current b ON b.problem_id=r.question_id_b
                WHERE r.review_status IN ({placeholders})
                ORDER BY CASE r.review_status WHEN '再確認待ち' THEN 0 ELSE 1 END,
                         r.similarity DESC, r.relation_id
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
    if not rows:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    return pd.DataFrame([dict(row) for row in rows])[RELATION_COLUMNS]


def get_relation_detail(database_path, relation_id: str) -> dict:
    with _database(database_path) as connection:
        relation = connection.execute(
            "SELECT * FROM relation_reviews WHERE relation_id=?",
            (relation_id,),
        ).fetchone()
        if relation is None:
            raise ValueError("指定した関係候補が見つかりません。")

        sides = {}
        for suffix in ("a", "b"):
            problem_id = relation[f"question_id_{suffix}"]
            current = connection.execute(
                "SELECT * FROM questions_current WHERE problem_id=?",
                (problem_id,),
            ).fetchone()
            if current is None:
                raise ValueError(f"問題データが見つかりません: {problem_id}")
            reference_hash = (
                relation[f"reviewed_hash_{suffix}"]
                or relation[f"candidate_hash_{suffix}"]
            )
            previous = None
            if reference_hash:
                previous = connection.execute(
                    """
                    SELECT snapshot_json FROM question_versions
                    WHERE problem_id=? AND content_hash=?
                    """,
                    (problem_id, reference_hash),
                ).fetchone()
            sides[suffix] = {
                "problem_id": problem_id,
                "current": json.loads(current["snapshot_json"]),
                "previous": json.loads(previous["snapshot_json"]) if previous else None,
                "changed": bool(reference_hash and reference_hash != current["content_hash"]),
                "active": bool(current["active"]),
                "subject": current["subject"],
                "round": current["exam_round"],
                "problem_number": current["problem_number"],
            }
    return {"relation": dict(relation), "a": sides["a"], "b": sides["b"]}


def save_relation_review(
    database_path,
    relation_id: str,
    reviewed_type: str,
    note: str = "",
) -> None:
    if reviewed_type not in REVIEW_TYPES:
        raise ValueError(f"判定は{', '.join(REVIEW_TYPES)}から選択してください。")
    now = datetime.now(timezone.utc).isoformat()
    with _database(database_path) as connection:
        relation = connection.execute(
            "SELECT * FROM relation_reviews WHERE relation_id=?",
            (relation_id,),
        ).fetchone()
        if relation is None:
            raise ValueError("指定した関係候補が見つかりません。")
        if reviewed_type == "対象外":
            status = "対象外"
        elif reviewed_type == relation["auto_type"]:
            status = "確認済み"
        else:
            status = "修正済み"
        current_a = connection.execute(
            "SELECT content_hash FROM questions_current WHERE problem_id=?",
            (relation["question_id_a"],),
        ).fetchone()
        current_b = connection.execute(
            "SELECT content_hash FROM questions_current WHERE problem_id=?",
            (relation["question_id_b"],),
        ).fetchone()
        if current_a is None or current_b is None:
            raise ValueError("関係する問題の内容を確認できません。")
        _record_relation_history(
            connection, relation, status, reviewed_type, "人による判定",
            current_a["content_hash"], current_b["content_hash"],
            (relation["review_note"], note.strip()),
        )
        connection.execute(
            """
            UPDATE relation_reviews
            SET review_status=?, reviewed_type=?, review_note=?,
                reviewed_hash_a=?, reviewed_hash_b=?, updated_at=?
            WHERE relation_id=?
            """,
            (
                status, reviewed_type, note.strip(), current_a["content_hash"],
                current_b["content_hash"], now, relation_id,
            ),
        )


def add_manual_related_pair(
    database_path,
    question_id_a: str,
    question_id_b: str,
    note: str = "",
) -> str:
    if question_id_a == question_id_b:
        raise ValueError("異なる2問を指定してください。")
    relation_id, id_a, id_b = _relation_id(question_id_a, question_id_b)
    now = datetime.now(timezone.utc).isoformat()
    with _database(database_path) as connection:
        question_a = connection.execute(
            "SELECT content_hash, active FROM questions_current WHERE problem_id=?",
            (id_a,),
        ).fetchone()
        question_b = connection.execute(
            "SELECT content_hash, active FROM questions_current WHERE problem_id=?",
            (id_b,),
        ).fetchone()
        if question_a is None or question_b is None:
            raise ValueError("指定した問題IDはデータベースにありません。先に過去問を同期してください。")
        if not question_a["active"] or not question_b["active"]:
            raise ValueError("入力データから削除された問題は新規関係として登録できません。")
        relation = connection.execute(
            "SELECT * FROM relation_reviews WHERE relation_id=?",
            (relation_id,),
        ).fetchone()
        if relation is None:
            connection.execute(
                """
                INSERT INTO relation_reviews
                    (relation_id, question_id_a, question_id_b, auto_type, similarity,
                     evidence, candidate_hash_a, candidate_hash_b, reviewed_hash_a,
                     reviewed_hash_b, review_status, reviewed_type, review_note,
                     created_at, updated_at)
                VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, '修正済み', 'C', ?, ?, ?)
                """,
                (
                    relation_id, id_a, id_b, "人が関連問題として追加",
                    question_a["content_hash"], question_b["content_hash"],
                    question_a["content_hash"], question_b["content_hash"],
                    note.strip(), now, now,
                ),
            )
            relation = connection.execute(
                "SELECT * FROM relation_reviews WHERE relation_id=?",
                (relation_id,),
            ).fetchone()
            _record_relation_history(
                connection, relation, "修正済み", "C", "人が関連問題として追加",
                question_a["content_hash"], question_b["content_hash"],
                (None, note.strip()),
            )
        else:
            _record_relation_history(
                connection, relation, "修正済み", "C", "人が関連問題として追加",
                question_a["content_hash"], question_b["content_hash"],
                (relation["review_note"], note.strip()),
            )
            connection.execute(
                """
                UPDATE relation_reviews
                SET review_status='修正済み', reviewed_type='C', review_note=?,
                    reviewed_hash_a=?, reviewed_hash_b=?, updated_at=?
                WHERE relation_id=?
                """,
                (
                    note.strip(), question_a["content_hash"], question_b["content_hash"],
                    now, relation_id,
                ),
            )
    return relation_id


def list_relation_history(database_path, limit: int = 500) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("履歴表示件数は1以上にしてください。")
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT h.*, r.question_id_a, r.question_id_b
            FROM relation_history h
            JOIN relation_reviews r ON r.relation_id=h.relation_id
            ORDER BY h.history_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def list_question_changes(database_path, limit: int = 500) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("変更履歴の表示件数は1以上にしてください。")
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT change_id, problem_id, before_hash, after_hash, before_json,
                   after_json, reason, changed_at
            FROM question_changes
            ORDER BY change_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        before = json.loads(row["before_json"]) if row["before_json"] else None
        after = json.loads(row["after_json"]) if row["after_json"] else None
        result.append({
            "変更ID": row["change_id"],
            "問題ID": row["problem_id"],
            "変更理由": row["reason"],
            "変更日時": row["changed_at"],
            "変更前の問題文": before["問題"] if before else "",
            "変更後の問題文": after["問題"] if after else "",
            "変更前の選択肢": " / ".join(before["選択肢"]) if before else "",
            "変更後の選択肢": " / ".join(after["選択肢"]) if after else "",
            "変更前の模範解答": before["模範解答"] if before else "",
            "変更後の模範解答": after["模範解答"] if after else "",
            "変更前の正答": before["正答"] if before else "",
            "変更後の正答": after["正答"] if after else "",
            "変更前の内容ハッシュ": row["before_hash"],
            "変更後の内容ハッシュ": row["after_hash"],
        })
    return pd.DataFrame(result, columns=[
        "変更ID", "問題ID", "変更理由", "変更日時",
        "変更前の問題文", "変更後の問題文",
        "変更前の選択肢", "変更後の選択肢",
        "変更前の模範解答", "変更後の模範解答",
        "変更前の正答", "変更後の正答",
        "変更前の内容ハッシュ", "変更後の内容ハッシュ",
    ])


def list_current_questions(
    database_path, query: str = "", limit: int = 100
) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("問題表示件数は1以上にしてください。")
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT problem_id, exam_round, subject, problem_number, major, middle,
                   active
            FROM questions_current
            WHERE active=1 AND problem_id LIKE ?
            ORDER BY exam_round DESC, subject, problem_number
            LIMIT ?
            """,
            (f"%{query.strip()}%", limit),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_review_counts(database_path) -> pd.DataFrame:
    with _database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT review_status AS 判定状態, COUNT(*) AS 件数
            FROM relation_reviews
            GROUP BY review_status
            ORDER BY CASE review_status WHEN '再確認待ち' THEN 0
                     WHEN '自動候補' THEN 1 ELSE 2 END, review_status
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows], columns=["判定状態", "件数"])
