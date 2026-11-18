"""Descriptive trend analysis for historical exam questions."""

import statistics
from typing import Optional

import pandas as pd


MIN_REGRESSION_ROUNDS = 8
MIN_FITTED_CHANGE_PP = 5.0
MIN_R_SQUARED = 0.30


def _number_text(value) -> str:
    if pd.isna(value):
        raise ValueError("問題番号が未入力の問題があります。")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _problem_id(row: pd.Series, subject: str) -> str:
    year = int(row["回"])
    return f"第{year}回/{subject}/No.{_number_text(row['問題番号'])}"


def _period_stats(yearly: pd.DataFrame, start: int, end: int, label: str) -> dict:
    expected_rounds = max(0, end - start + 1)
    in_range = yearly[yearly["回"].between(start, end)]
    covered = in_range[in_range["収録状態"] == "収録あり"]
    missing_rounds = sorted(set(range(start, end + 1)) - set(covered["回"].astype(int)))
    measurable = covered[covered["分母問題数"] > 0]
    denominator = int(measurable["分母問題数"].sum())
    numerator = int(measurable["対象問題数"].sum())
    complete = not missing_rounds
    share = round(numerator / denominator * 100, 2) if complete and denominator else None
    if not complete:
        status = "未収録回あり"
    elif not denominator:
        status = "分母0のため算出不可"
    else:
        status = "算出済み"
    return {
        "期間": label,
        "開始回": start,
        "終了回": end,
        "対象出題数(収録分)": numerator,
        "分母問題数(算出対象回)": denominator,
        "分母対象": "大項目" if yearly["分母区分"].iloc[0] == "大項目" else "科目全体",
        "収録回数": len(covered),
        "対象回数": expected_rounds,
        "分母あり回数": len(measurable),
        "出題割合(%)": share,
        "集計状態": status,
        "未収録回": "、".join(map(str, missing_rounds)),
    }


def _regression(yearly: pd.DataFrame, start: int, end: int) -> dict:
    in_range = yearly[
        yearly["回"].between(start, end)
        & yearly["収録状態"].eq("収録あり")
        & yearly["分母問題数"].gt(0)
    ]
    n = len(in_range)
    expected = max(0, end - start + 1)
    source_rounds = yearly[
        yearly["回"].between(start, end) & yearly["収録状態"].eq("収録あり")
    ]
    source_complete = len(source_rounds) == expected

    result = {
        "回帰対象回数": n,
        "回帰傾き(pp/回)": None,
        "回帰推定差(pp)": None,
        "回帰R²": None,
        "回帰判定": "判定保留",
        "回帰判定理由": "",
    }
    if not source_complete:
        result["回帰判定理由"] = "対象範囲に未収録回がある"
        return result
    if n < MIN_REGRESSION_ROUNDS:
        result["回帰判定理由"] = f"有効回数が{MIN_REGRESSION_ROUNDS}回未満"
        return result

    xs = in_range["回"].astype(float).tolist()
    ys = in_range["出題割合(%)"].astype(float).tolist()
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ss_x
    intercept = mean_y - slope * mean_x
    fitted_change = slope * (max(xs) - min(xs))
    fitted = [intercept + slope * x for x in xs]
    ss_total = sum((y - mean_y) ** 2 for y in ys)
    ss_residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, fitted))
    r_squared = 1 - ss_residual / ss_total if ss_total else None
    result.update({
        "回帰傾き(pp/回)": round(slope, 4),
        "回帰推定差(pp)": round(fitted_change, 2),
        "回帰R²": round(r_squared, 4) if r_squared is not None else None,
    })

    if abs(fitted_change) < MIN_FITTED_CHANGE_PP:
        result["回帰判定"] = "横ばい"
        result["回帰判定理由"] = "回帰直線の期間全体の推定差が判定幅未満"
    elif r_squared is None or r_squared < MIN_R_SQUARED:
        result["回帰判定理由"] = f"回帰R²が{MIN_R_SQUARED}未満または算出不可"
    elif fitted_change > 0:
        result["回帰判定"] = "増加"
        result["回帰判定理由"] = "回帰推定差とR²が判定基準を満たす"
    else:
        result["回帰判定"] = "減少"
        result["回帰判定理由"] = "回帰推定差とR²が判定基準を満たす"
    return result


def _coefficient_of_variation(yearly: pd.DataFrame, start: int, end: int):
    range_rows = yearly[yearly["回"].between(start, end)]
    if range_rows["収録状態"].ne("収録あり").any():
        return None
    values = yearly.loc[
        yearly["回"].between(start, end)
        & yearly["収録状態"].eq("収録あり")
        & yearly["分母問題数"].gt(0),
        "出題割合(%)",
    ].astype(float).tolist()
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return round(statistics.pstdev(values) / mean * 100, 2)


def _summarize_area(yearly: pd.DataFrame, start: int, end: int, label: str) -> dict:
    selected = _period_stats(yearly, start, end, "選択期間")
    full_start = int(yearly.loc[yearly["収録状態"].eq("収録あり"), "回"].min())
    full_end = int(yearly.loc[yearly["収録状態"].eq("収録あり"), "回"].max())
    recent_end = full_end

    metrics = {
        "分析対象": label,
        "選択期間": f"第{start}〜{end}回",
        "選択期間出題数": selected["対象出題数(収録分)"],
        "選択期間分母": selected["分母問題数(算出対象回)"],
        "選択期間出題割合(%)": selected["出題割合(%)"],
        "前期後期差(pp)": None,
    }
    for period_label, period_start, period_end in (
        ("全期間", full_start, full_end),
        ("直近5回", recent_end - 4, recent_end),
        ("直近10回", recent_end - 9, recent_end),
        ("直近15回", recent_end - 14, recent_end),
    ):
        stats = _period_stats(yearly, period_start, period_end, period_label)
        metrics[f"{period_label}出題数"] = stats["対象出題数(収録分)"]
        metrics[f"{period_label}分母"] = stats["分母問題数(算出対象回)"]
        metrics[f"{period_label}出題割合(%)"] = stats["出題割合(%)"]
        metrics[f"{period_label}集計状態"] = stats["集計状態"]

    period_length = end - start + 1
    first_end = start + (period_length + 1) // 2 - 1
    second_start = first_end + 1
    first = _period_stats(yearly, start, first_end, "前期")
    second = _period_stats(yearly, second_start, end, "後期")
    if first["出題割合(%)"] is not None and second["出題割合(%)"] is not None:
        metrics["前期後期差(pp)"] = round(
            second["出題割合(%)"] - first["出題割合(%)"], 2
        )

    regression = _regression(yearly, start, end)
    metrics.update(regression)
    metrics["変動係数(%)"] = _coefficient_of_variation(yearly, start, end)
    return metrics, first, second


def analyze_trends(
    questions: pd.DataFrame,
    subject: str,
    major: str,
    selected_middle: Optional[str],
    denominator_basis: str,
    year_min: int,
    year_max: int,
    middle_items: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Return trend metrics, graph-ready annual data, and domain rankings."""
    if year_min > year_max:
        raise ValueError("分析開始回は終了回以前にしてください。")
    if denominator_basis not in {"大項目", "科目全体"}:
        raise ValueError("分母は「大項目」または「科目全体」を指定してください。")
    required = {"回", "問題番号", "大項目", "中項目"}
    missing_columns = required - set(questions.columns)
    if missing_columns:
        raise ValueError(f"問題データに必要な列がありません: {', '.join(sorted(missing_columns))}")

    source = questions.copy()
    source["回"] = pd.to_numeric(source["回"], errors="coerce")
    source = source[source["回"].notna()].copy()
    source["回"] = source["回"].astype(int)
    available_rounds = sorted(source["回"].unique().tolist())
    if not available_rounds:
        raise ValueError(f"{subject}に出題回を読み取れる問題がありません。")

    by_round = {int(year): group for year, group in source.groupby("回", sort=True)}
    data_start = min(year_min, min(available_rounds))
    data_end = max(year_max, max(available_rounds))
    years = list(range(data_start, data_end + 1))
    area_names = list(dict.fromkeys(middle_items or []))
    if not area_names:
        area_names = sorted(
            source.loc[source["大項目"].eq(major), "中項目"].dropna().astype(str).unique()
        )
    area_names = [name for name in area_names if name and name != "nan"]

    annual_by_area: dict[str, list[dict]] = {}
    selected_area_label = selected_middle or f"{major}（大項目全体）"
    targets = [selected_area_label] + [name for name in area_names if name != selected_middle]
    if selected_middle is None:
        targets = [selected_area_label] + area_names

    for target in targets:
        target_middle = target if target in area_names else None
        rows = []
        for year in years:
            round_questions = by_round.get(year)
            if round_questions is None:
                rows.append({
                    "回": year,
                    "分析対象": target,
                    "対象問題数": None,
                    "科目問題数": None,
                    "大項目問題数": None,
                    "分母問題数": None,
                    "出題割合(%)": None,
                    "分母区分": denominator_basis,
                    "収録状態": "未収録",
                    "問題ID": "",
                })
                continue

            major_questions = round_questions[round_questions["大項目"].eq(major)]
            if target_middle is None:
                target_questions = major_questions
            else:
                target_questions = major_questions[
                    major_questions["中項目"].astype(str).eq(target_middle)
                ]
            denominator = (
                len(major_questions)
                if denominator_basis == "大項目"
                else len(round_questions)
            )
            rate = round(len(target_questions) / denominator * 100, 2) if denominator else None
            ids = "、".join(
                _problem_id(row, subject)
                for _, row in target_questions.sort_values("問題番号").iterrows()
            )
            rows.append({
                "回": year,
                "分析対象": target,
                "対象問題数": len(target_questions),
                "科目問題数": len(round_questions),
                "大項目問題数": len(major_questions),
                "分母問題数": denominator,
                "出題割合(%)": rate,
                "分母区分": denominator_basis,
                "収録状態": "収録あり",
                "問題ID": ids,
            })
        annual_by_area[target] = rows

    all_yearly = pd.DataFrame(annual_by_area[selected_area_label])
    selected_yearly = all_yearly[all_yearly["回"].between(year_min, year_max)].copy()
    selected_metrics, first_half, second_half = _summarize_area(
        all_yearly, year_min, year_max, selected_area_label
    )
    selected_metrics["前期範囲"] = f"第{year_min}〜{first_half['終了回']}回"
    selected_metrics["後期範囲"] = (
        f"第{second_half['開始回']}〜{year_max}回"
        if second_half["開始回"] <= year_max else "該当なし"
    )
    summary = pd.DataFrame([selected_metrics])

    period_definitions = []
    full_start, full_end = min(available_rounds), max(available_rounds)
    for label, start, end in (
        ("全期間", full_start, full_end),
        ("直近5回", full_end - 4, full_end),
        ("直近10回", full_end - 9, full_end),
        ("直近15回", full_end - 14, full_end),
        ("選択期間", year_min, year_max),
    ):
        period_definitions.append((label, start, end))
    periods = pd.DataFrame([
        _period_stats(all_yearly, start, end, label)
        for label, start, end in period_definitions
    ])

    first_half["分析対象"] = selected_area_label
    second_half["分析対象"] = selected_area_label
    comparison = pd.DataFrame([first_half, second_half])
    difference = None
    if first_half["出題割合(%)"] is not None and second_half["出題割合(%)"] is not None:
        difference = round(second_half["出題割合(%)"] - first_half["出題割合(%)"], 2)
    comparison["前期後期差(pp)"] = difference

    area_rows = []
    for area in area_names:
        area_yearly = pd.DataFrame(annual_by_area[area])
        metrics, _, _ = _summarize_area(area_yearly, year_min, year_max, area)
        area_rows.append(metrics)
    areas = pd.DataFrame(area_rows)
    if not areas.empty:
        increase = areas[areas["回帰判定"] == "増加"].sort_values(
            "回帰推定差(pp)", ascending=False, kind="stable"
        )
        decrease = areas[areas["回帰判定"] == "減少"].sort_values(
            "回帰推定差(pp)", ascending=True, kind="stable"
        )
        volatility = areas[areas["変動係数(%)"].notna()].sort_values(
            "変動係数(%)", ascending=False, kind="stable"
        )
    else:
        increase = areas.copy()
        decrease = areas.copy()
        volatility = areas.copy()

    condition = pd.DataFrame({
        "項目": [
            "科目", "大項目", "分析対象", "分析対象回", "比較分母",
            "回帰の最低有効回数", "回帰推定差の増減基準", "回帰R²の基準",
            "回帰判定に必要な収録状態", "出題割合の集計方法",
        ],
        "内容": [
            subject, major, selected_area_label, f"第{year_min}〜{year_max}回",
            "大項目の問題数" if denominator_basis == "大項目" else "科目全体の問題数",
            MIN_REGRESSION_ROUNDS, f"±{MIN_FITTED_CHANGE_PP}ポイント",
            f"{MIN_R_SQUARED}以上", "対象期間に科目の未収録回がないこと",
            "期間内の対象問題数合計÷分母問題数合計（分母0の回は除外）",
        ],
    })

    return {
        "条件": condition,
        "選択領域指標": summary,
        "期間別出題割合": periods,
        "前期・後期比較": comparison,
        "年度別推移": selected_yearly,
        "増加領域ランキング": increase.reset_index(drop=True),
        "減少領域ランキング": decrease.reset_index(drop=True),
        "出題頻度変動ランキング": volatility.reset_index(drop=True),
        "中項目別指標": areas.reset_index(drop=True),
    }
