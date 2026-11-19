"""既存の分析結果から、期間比較の記事たたき台を作る。"""

import pandas as pd

import analysis
import classification
import repeats
import terms
import trends


PERIODS = [(1, 9), (10, 18), (19, 27), (28, 34)]


def _assigned(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & text.ne("記載なし")


def build_article_data(freq_file, past_file, subject: str, major: str,
                       classification_data=None, repeat_data=None) -> dict:
    frequency, year_columns = analysis.load_freq(freq_file, subject)
    if not major or major == "記載なし":
        raise ValueError("分析対象として分類済みの大項目を選択してください。")
    year_columns = [column for column in year_columns
                    if 1 <= int(column[1:-1]) <= 34]
    missing_columns = [f"第{year}回" for year in range(1, 35)
                       if f"第{year}回" not in year_columns]
    if missing_columns:
        raise ValueError("出題頻度表に第1〜34回の列が不足しています: " + "、".join(missing_columns))
    valid_frequency = frequency[_assigned(frequency["大項目"]) & _assigned(frequency["中項目"])]
    ranking = analysis.rank_middle(valid_frequency, year_columns, major, 3)
    ranking = ranking.drop_duplicates("中項目").head(3).copy()
    if ranking.empty:
        raise ValueError("対象大項目に分類済みの中項目がありません。")
    targets = ranking["中項目"].tolist()

    all_questions = analysis.load_questions(past_file, subject, 1, 34)
    classified = _assigned(all_questions["大項目"]) & _assigned(all_questions["中項目"])
    major_questions = all_questions[classified & all_questions["大項目"].eq(major)].copy()
    target_questions = major_questions[major_questions["中項目"].isin(targets)].copy()
    target_questions["_id"] = pd.Series(
        [repeats.canonical_problem_id(row, subject) for _, row in target_questions.iterrows()],
        index=target_questions.index, dtype="object",
    )
    if target_questions["_id"].duplicated().any():
        raise ValueError("対象問題の問題IDが重複しています。")
    covered = set(all_questions["回"].dropna().astype(int))
    period_rows = []
    for start, end in PERIODS:
        missing = [year for year in range(start, end + 1) if year not in covered]
        major_period = major_questions[major_questions["回"].between(start, end)]
        unassigned = all_questions[
            all_questions["回"].between(start, end)
            & all_questions["大項目"].eq(major) & ~_assigned(all_questions["中項目"])
        ]
        unknown_major = all_questions[
            all_questions["回"].between(start, end) & ~_assigned(all_questions["大項目"])
        ]
        for topic in targets:
            selected = major_period[major_period["中項目"].eq(topic)]
            denominator = len(major_period)
            period_rows.append({
                "期間": f"第{start}〜{end}回", "開始回": start, "終了回": end,
                "中項目": topic, "問題数": len(selected), "分母": denominator,
                "割合(%)": round(len(selected) / denominator * 100, 1)
                if denominator and not missing else None,
                "未収録回": missing, "中項目未対応": len(unassigned),
                "大項目未対応": len(unknown_major),
            })

    dictionary = analysis.load_dictionary()
    stop = {terms.norm(w) for w in dictionary.get("除外語", [])}
    exact_stop = {terms.norm(w) for w in valid_frequency.loc[
        valid_frequency["大項目"].eq(major), "中項目"
    ]}
    exact_stop.add(terms.norm(major))
    curated = analysis.load_medical_terms() + list(dictionary["重要語(疾患・所見)"])
    frequent = terms.topic_frequent_terms(
        target_questions, targets, curated, stop, exact_stop
    )

    taxonomy_rows = []
    for period in period_rows:
        selected = target_questions[
            target_questions["中項目"].eq(period["中項目"])
            & target_questions["回"].between(period["開始回"], period["終了回"])
        ]
        if classification_data is None:
            taxonomy_rows.append({**period, "利用可能": False})
            continue
        details = classification_data
        if details.empty:
            confirmed = details
        else:
            details = details[details["問題ID"].isin(selected["_id"])]
            if details["問題ID"].duplicated().any():
                raise ValueError("Taxonomyの問題IDが重複しています。")
            confirmed = details[details["Taxonomy状態"].isin(classification.CONFIRMED_STATUSES)]
            if not confirmed["Taxonomy"].isin(classification.TAXONOMIES).all():
                raise ValueError("人間確認済みTaxonomyに不正な分類値があります。")
        counts = {
            value: int(confirmed["Taxonomy"].eq(value).sum()) if not confirmed.empty else 0
            for value in classification.TAXONOMIES
        }
        taxonomy_rows.append({
            **period, "利用可能": True, **counts,
            "分類済み": sum(counts[value] for value in classification.TAXONOMIES[:3]),
            "未確認": len(selected) - len(confirmed),
        })

    trend_source = all_questions[
        ~all_questions["大項目"].eq(major) | _assigned(all_questions["中項目"])
    ]
    trend_result = None
    if not trend_source.empty:
        trend_result = trends.analyze_trends(
            trend_source, subject, major, None, "大項目", 25, 34,
            valid_frequency.loc[valid_frequency["大項目"].eq(major), "中項目"]
            .drop_duplicates().tolist(),
        )
    matched_repeats = None
    if repeat_data is not None:
        ids = set(target_questions["_id"])
        if repeat_data.empty:
            matched_repeats = repeat_data
        else:
            matched_repeats = repeat_data[
                repeat_data["active_a"].eq(1) & repeat_data["active_b"].eq(1)
                & repeat_data["subject_a"].eq(subject) & repeat_data["subject_b"].eq(subject)
                & repeat_data["round_a"].between(1, 34) & repeat_data["round_b"].between(1, 34)
                & repeat_data["round_a"].ne(repeat_data["round_b"])
                & (repeat_data["question_id_a"].isin(ids) | repeat_data["question_id_b"].isin(ids))
                & (((repeat_data["review_status"].isin(["確認済み", "修正済み"]))
                    & repeat_data["reviewed_type"].eq("A"))
                   | (repeat_data["review_status"].isin(["自動候補", "再確認待ち"])
                      & (repeat_data["auto_type"].eq("A") | repeat_data["reviewed_type"].eq("A"))))
            ].copy()
    return {
        "科目": subject, "大項目": major, "順位": ranking,
        "期間別": pd.DataFrame(period_rows), "頻出語": frequent,
        "Taxonomy": pd.DataFrame(taxonomy_rows), "トレンド": trend_result,
        "再出題": matched_repeats,
    }


def generate_article(result: dict, mea: str = "", next_theme: str = "",
                     mock_results: str = "") -> str:
    subject, major = result["科目"], result["大項目"]
    ranking = result["順位"]
    targets = ranking["中項目"].tolist()
    periods = result["期間別"]
    sections = [
        f"# {subject}「{major}」の出題傾向\n\n"
        "第1〜34回の出題頻度表で上位3中項目を選び、"
        "第1〜9・10〜18・19〜27・28〜34回を比較します。"
        "同数の場合は出題頻度表の順序で選びます。旧回も過去問の既存分類を使用し、"
        "現行基準への新たな対応付けは行っていません。",
    ]
    lines = ["| 中項目 | 第1〜34回の出題数（頻度表） |",
             "| --- | ---: |"]
    for _, row in ranking.iterrows():
        lines.append(f"| {row['中項目']} | {int(row['合計'])} |")
    sections.append("## 分析対象\n\n" + "\n".join(lines))
    lines = ["| 期間 | 中項目 | 問題数 / 分類済み大項目の問題数 | 出題割合 |",
             "| --- | --- | ---: | ---: |"]
    notes = []
    for _, row in periods.iterrows():
        share = "集計不可" if pd.isna(row["割合(%)"]) else f"{row['割合(%)']:.1f}%"
        lines.append(f"| {row['期間']} | {row['中項目']} | {row['問題数']} / {row['分母']} | {share} |")
        note = (
            f"{row['期間']}：大項目内の中項目未対応{row['中項目未対応']}問を分母から除外。"
            f"科目内の大項目未対応{row['大項目未対応']}問は対象大項目を特定できません。"
        )
        if row["未収録回"]:
            note += " 未収録：" + "、".join(f"第{year}回" for year in row["未収録回"])
        if note not in notes:
            notes.append(note)
    changes = []
    for topic in targets:
        rows = periods[periods["中項目"].eq(topic)]
        previous = rows[rows["開始回"].eq(19)]["割合(%)"].iloc[0]
        current = rows[rows["開始回"].eq(28)]["割合(%)"].iloc[0]
        if pd.notna(previous) and pd.notna(current):
            difference = round(current - previous, 1)
            changes.append(f"- {topic}：前期から現行期へ{difference:+.1f}ポイント（表示値の差）。")
        else:
            changes.append(f"- {topic}：未収録または分母不足のため前期との差は算出不可。")
    sections.append(
        "## 期間別の出題割合\n\n期間の問題数を合算した割合です。分母には選択した上位3中項目以外の"
        "分類済み中項目も含みます。未収録回のある期間と分母0の期間は割合を算出しません。\n\n"
        + "\n".join(lines) + "\n\n第19〜27回から第28〜34回への変化：\n"
        + "\n".join(changes) + "\n\n" + "\n".join(f"- {note}" for note in notes)
    )
    for topic in targets:
        topic_taxonomy = result["Taxonomy"][result["Taxonomy"]["中項目"].eq(topic)]
        lines = [f"## {topic}：Taxonomyと頻出語"]
        if not topic_taxonomy["利用可能"].all():
            lines.append("Taxonomy：現在の過去問に同期済みの分類を確認できません。")
        else:
            lines.append("| 期間 | Ⅰ型 | Ⅱ型 | Ⅲ型 | 分類済み分母 | 確認済み保留 | 未確認 |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _, row in topic_taxonomy.iterrows():
                lines.append(
                    f"| {row['期間']} | {row['Ⅰ型']} | {row['Ⅱ型']} | {row['Ⅲ型']} | "
                    f"{row['分類済み']} | {row['判定保留']} | {row['未確認']} |"
                )
        words = result["頻出語"][result["頻出語"]["中項目"].eq(topic)].sort_values("順位")
        if words.empty:
            lines.append("頻出語候補：条件を満たす語はありません。")
        else:
            lines.append("頻出語候補（第1〜34回）：")
            for _, row in words.iterrows():
                lines.append(f"- 「{row['語']}」：{int(row['出現問題数'])}問。該当問題：{row['問題ID']}")
        sections.append("\n\n".join(lines))
    sections.append(
        "Taxonomyは人間確認済み・修正済みのⅠ〜Ⅲ型のみを分類済み分母とし、"
        "確認済み保留と未確認を分けています。難易度ではありません。"
        "頻出語は問題文と全選択肢に2問以上出現した候補で、誤答も含みます。"
        "正答や問題の主題を意味するとは限らないため、問題IDから原資料を確認してください。"
    )
    trend_result = result["トレンド"]
    lines = ["## 近年のトレンド（第25〜34回）"]
    if trend_result is None:
        lines.append("分類済みの分析対象データがないため、判定できません。")
    else:
        for label, key in (("増加", "増加領域ランキング"), ("減少", "減少領域ランキング")):
            frame = trend_result[key]
            if frame.empty:
                lines.append(f"- {label}：既存の判定基準を満たす中項目はありません。")
            else:
                for _, row in frame.iterrows():
                    lines.append(
                        f"- {label}：{row['分析対象']}（回帰推定差 {row['回帰推定差(pp)']}ポイント、"
                        f"有効{int(row['回帰対象回数'])}回、R²={row['回帰R²']}）。"
                    )
    lines.append(
        "大項目内の出題割合について、有効8回以上・推定差±5ポイント以上・"
        "R²が0.30以上の場合に増減と判定します。基準を満たさない項目は増減を断定できず、"
        "将来の出題を予測するものではありません。"
    )
    sections.append("\n\n".join(lines))
    relations = result["再出題"]
    if relations is None:
        sections.append("## 再出題\n\n現在の過去問に同期済みの再出題レビューを確認できません。")
    elif relations.empty:
        sections.append("## 再出題\n\n対象中項目に関連する確認済みの同一問題・未確認候補はありません。"
                        "レビューや候補抽出の状況により、再出題がないとは断定できません。")
    else:
        lines = ["## 再出題"]
        for _, row in relations.iterrows():
            status = (
                "未確認候補" if row["review_status"] == "自動候補"
                else "再確認待ち" if row["review_status"] == "再確認待ち"
                else "人間確認済み"
            )
            lines.append(f"- {status}：{row['question_id_a']} ／ {row['question_id_b']}")
        lines.append("同一問題の候補と人間確認済みを区別しています。未確認候補は原資料で確認してください。")
        sections.append("\n\n".join(lines))
    if mock_results.strip():
        sections.append("## 自校の模試結果\n\n" + mock_results.strip())
    if mea.strip():
        sections.append("## MEAについて\n\n" + mea.strip())
    if next_theme.strip():
        sections.append("## 次回予告\n\n次回は「" + next_theme.strip() + "」を取り上げる予定です。")
    return "\n\n".join(sections) + "\n"
