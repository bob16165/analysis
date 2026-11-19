"""Observed question history for current- and former-standard categories."""

import pandas as pd

from repeats import canonical_problem_id


DEFAULT_BINS = (1, 3, 5)


def _missing(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def _item_pair(major, middle):
    if _missing(major) or _missing(middle):
        return None
    return str(major).strip(), str(middle).strip()


def _standard_class(pair) -> str:
    if pair is None:
        return "未分類"
    if "記載なし" in pair:
        return "旧出題基準由来"
    return "現行出題基準"


def _frequency_bin(count: int, boundaries: tuple[int, int, int]) -> str:
    first, second, third = boundaries
    if count == 0:
        return "出題実績なし"
    if count <= first:
        return f"1〜{first}回"
    if count <= second:
        return f"{first + 1}〜{second}回"
    if count <= third:
        return f"{second + 1}〜{third}回"
    return f"{third + 1}回以上"


def _bin_names(boundaries: tuple[int, int, int]) -> list[str]:
    first, second, third = boundaries
    return [
        "出題実績なし",
        f"1〜{first}回",
        f"{first + 1}〜{second}回",
        f"{second + 1}〜{third}回",
        f"{third + 1}回以上",
    ]


def analyze_standard_history(
    questions: pd.DataFrame,
    frequency_items: pd.DataFrame,
    subject: str,
    expected_rounds: list[int],
    period_start: int,
    period_end: int,
    recent_count: int = 10,
    frequency_bin_boundaries: tuple[int, int, int] = DEFAULT_BINS,
) -> dict[str, pd.DataFrame]:
    if period_start > period_end:
        raise ValueError("分析開始回は終了回以前にしてください。")
    if recent_count < 1:
        raise ValueError("直近回数は1以上にしてください。")
    if (
        len(frequency_bin_boundaries) != 3
        or frequency_bin_boundaries[0] < 1
        or not (
            frequency_bin_boundaries[0]
            < frequency_bin_boundaries[1]
            < frequency_bin_boundaries[2]
        )
    ):
        raise ValueError("出題頻度区分の境界は1以上の昇順で3つ指定してください。")

    expected = sorted(set(int(round_number) for round_number in expected_rounds))
    available = set(
        pd.to_numeric(questions.get("回", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    in_period_expected = [year for year in expected if period_start <= year <= period_end]
    missing_rounds = [year for year in in_period_expected if year not in available]
    latest_round = max(expected) if expected else period_end
    recent_start = latest_round - recent_count + 1
    recent_expected = [year for year in expected if recent_start <= year <= latest_round]
    recent_missing = [year for year in recent_expected if year not in available]

    items = {}
    for _, row in frequency_items.iterrows():
        pair = _item_pair(row.get("大項目"), row.get("中項目"))
        if pair is None:
            continue
        items[pair] = {
            "source": "出題頻度表",
            "standard_class": _standard_class(pair),
            "period_ids": [],
            "period_rounds": set(),
            "recent_ids": [],
            "recent_rounds": set(),
            "all_ids": [],
            "all_rounds": set(),
        }

    unclassified_rows = []
    question_records = []
    for _, row in questions.iterrows():
        round_value = pd.to_numeric(row.get("回"), errors="coerce")
        if pd.isna(round_value):
            continue
        round_number = int(round_value)
        pair = _item_pair(row.get("大項目"), row.get("中項目"))
        problem_id = canonical_problem_id(row, subject)
        if pair is None:
            unclassified_rows.append({
                "回": round_number,
                "問題ID": problem_id,
                "大項目": row.get("大項目"),
                "中項目": row.get("中項目"),
                "問題": row.get("問題", ""),
            })
            question_records.append((round_number, problem_id, None))
            continue
        if pair not in items:
            items[pair] = {
                "source": "過去問のみ",
                "standard_class": _standard_class(pair),
                "period_ids": [],
                "period_rounds": set(),
                "recent_ids": [],
                "recent_rounds": set(),
                "all_ids": [],
                "all_rounds": set(),
            }
        item = items[pair]
        if item["source"] == "出題頻度表":
            item["source"] = "出題頻度表・過去問"
        item["all_ids"].append(problem_id)
        item["all_rounds"].add(round_number)
        if period_start <= round_number <= period_end:
            item["period_ids"].append(problem_id)
            item["period_rounds"].add(round_number)
        if recent_start <= round_number <= latest_round:
            item["recent_ids"].append(problem_id)
            item["recent_rounds"].add(round_number)
        question_records.append((round_number, problem_id, pair))

    item_rows = []
    for (major, middle), item in sorted(items.items()):
        item_rows.append({
            "出題基準区分": item["standard_class"],
            "大項目": major,
            "中項目": middle,
            "項目一覧の出典": item["source"],
            "対象期間出題数": len(item["period_ids"]),
            "対象期間出題回数": len(item["period_rounds"]),
            "対象期間の出題頻度区分": _frequency_bin(
                len(item["period_ids"]), frequency_bin_boundaries
            ),
            f"直近{recent_count}回出題数": len(item["recent_ids"]),
            f"直近{recent_count}回出題回数": len(item["recent_rounds"]),
            "直近期間の問題ID": "、".join(item["recent_ids"]),
            "対象期間の問題ID": "、".join(item["period_ids"]),
            "全期間の問題ID": "、".join(item["all_ids"]),
        })
    item_table = pd.DataFrame(item_rows, columns=[
        "出題基準区分", "大項目", "中項目", "項目一覧の出典",
        "対象期間出題数", "対象期間出題回数", "対象期間の出題頻度区分",
        f"直近{recent_count}回出題数", f"直近{recent_count}回出題回数",
        "直近期間の問題ID", "対象期間の問題ID", "全期間の問題ID",
    ])

    standard_rows = []
    bin_rows = []
    for standard in ("現行出題基準", "旧出題基準由来"):
        matching = [item for item in items.values() if item["standard_class"] == standard]
        represented = sum(bool(item["period_rounds"]) for item in matching)
        recent_represented = sum(bool(item["recent_rounds"]) for item in matching)
        ids = [
            problem_id
            for round_number, problem_id, pair in question_records
            if pair is not None
            and _standard_class(pair) == standard
            and period_start <= round_number <= period_end
        ]
        recent_ids = [
            problem_id
            for round_number, problem_id, pair in question_records
            if pair is not None
            and _standard_class(pair) == standard
            and recent_start <= round_number <= latest_round
        ]
        recent_rate = (
            round(recent_represented / len(matching) * 100, 1)
            if matching and not recent_missing else None
        )
        standard_rows.append({
            "出題基準区分": standard,
            "確認できた項目数": len(matching),
            "対象期間出題実績のある項目数": represented,
            "対象期間の項目出題実績率(%)": (
                round(represented / len(matching) * 100, 1) if matching else None
            ),
            "対象期間の問題数": len(ids),
            "直近期間の出題実績のある項目数": recent_represented,
            f"直近{recent_count}回の項目出題実績率(%)": recent_rate,
            f"直近{recent_count}回の問題数": len(recent_ids),
            "直近期間の集計状態": "未収録回あり" if recent_missing else "算出済み",
        })

        for bin_name in _bin_names(frequency_bin_boundaries):
            count = sum(
                _frequency_bin(len(item["period_ids"]), frequency_bin_boundaries)
                == bin_name
                for item in matching
            )
            bin_rows.append({
                "出題基準区分": standard,
                "対象期間の出題頻度区分": bin_name,
                "項目数": count,
                "区分内構成比(%)": (
                    round(count / len(matching) * 100, 1) if matching else None
                ),
            })

    annual_rows = []
    for round_number in in_period_expected:
        registered = round_number in available
        for standard in ("現行出題基準", "旧出題基準由来"):
            matches = [
                (problem_id, pair)
                for year, problem_id, pair in question_records
                if year == round_number
                and pair is not None
                and _standard_class(pair) == standard
            ]
            annual_rows.append({
                "出題基準区分": standard,
                "回": round_number,
                "収録状態": "収録あり" if registered else "未収録",
                "問題数": len(matches) if registered else None,
                "該当問題ID": "、".join(problem_id for problem_id, _ in matches),
            })

    unclassified = pd.DataFrame(unclassified_rows, columns=[
        "回", "問題ID", "大項目", "中項目", "問題",
    ])
    settings = pd.DataFrame([
        {"項目": "分析対象", "内容": subject},
        {"項目": "対象期間", "内容": f"第{period_start}〜{period_end}回"},
        {"項目": "出題基準区分", "内容": "「記載なし」を含む項目は旧出題基準由来、それ以外は現行出題基準"},
        {"項目": "項目母集団", "内容": "出題頻度表または分析対象の過去問で確認できた大項目・中項目の組み合わせ"},
        {"項目": "網羅率の扱い", "内容": "公式出題基準全体の網羅率は算出していません"},
        {"項目": "期間内の未収録回", "内容": "、".join(map(str, missing_rounds)) or "なし"},
        {"項目": "直近期間", "内容": f"第{recent_start}〜{latest_round}回"},
        {"項目": "直近期間の未収録回", "内容": "、".join(map(str, recent_missing)) or "なし"},
        {"項目": "未分類問題数", "内容": str(len(unclassified))},
        {"項目": "頻度区分", "内容": "、".join(_bin_names(frequency_bin_boundaries))},
    ])
    return {
        "出題基準区分別集計": pd.DataFrame(standard_rows),
        "項目別出題実績": item_table,
        "出題頻度区分別集計": pd.DataFrame(bin_rows),
        "年度別の区分別出題": pd.DataFrame(annual_rows),
        "分類未確定問題": unclassified,
        "分析条件": settings,
    }
