import io
import json
import re
from pathlib import Path

import pandas as pd

import terms

DICT_PATH = Path(__file__).parent / "dictionary.json"
CHOICE_COLS = ["選択肢1", "選択肢2", "選択肢3", "選択肢4"]
RELATED_AXIS_SUBJECTS = ["リハ医学", "一般臨床", "整形外科", "柔道整復学"]
SUBJECT_LABELS = {"柔道整復学": "柔整理論"}


def norm(x) -> str:
    return terms.norm(x)


def load_dictionary() -> dict:
    return json.loads(DICT_PATH.read_text(encoding="utf-8"))


def save_dictionary(d: dict) -> None:
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def sheet_names(file) -> list[str]:
    return pd.ExcelFile(file).sheet_names


def load_freq(file, subject: str) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=subject)
    df = df[df["大項目"].notna() & df["中項目"].notna()].copy()
    year_cols = [c for c in df.columns if re.fullmatch(r"第\d+回", str(c))]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    return df, year_cols


def rank_middle(freq: pd.DataFrame, year_cols, major: str, top_n: int) -> pd.DataFrame:
    f = freq[freq["大項目"] == major].copy()
    f["合計"] = f[year_cols].sum(axis=1)
    f["出題された回数"] = (f[year_cols] > 0).sum(axis=1)
    f = f.sort_values("合計", ascending=False, kind="stable").reset_index(drop=True)
    f["順位"] = f["合計"].rank(method="min", ascending=False).astype(int)
    # 同順位は上位N位に入る限りすべて採用
    f["分析対象"] = f["順位"] <= top_n
    return f[["順位", "中項目", *year_cols, "合計", "出題された回数", "分析対象"]]


def load_questions(file, subject: str, year_min: int, year_max: int) -> pd.DataFrame:
    q = pd.read_excel(file, sheet_name=subject)
    q["回"] = pd.to_numeric(q["国家試験出題回"], errors="coerce")
    q = q[q["回"].between(year_min, year_max)].copy()
    q["正答番号"] = q["模範解答"].map(lambda v: re.findall(r"\d", str(v)))
    q["正答"] = q.apply(lambda r: "／".join(str(r[f"選択肢{i}"]) for i in r["正答番号"] if f"選択肢{i}" in q.columns), axis=1)
    q["問題ID"] = q["出題回数"]  # 例: 第28回No.25
    return q


def tag_question(r: pd.Series, d: dict) -> dict:
    stem = norm(r["問題"])
    choices = " ".join(norm(r[c]) for c in CHOICE_COLS)
    correct = norm(r["正答"])
    text = stem + choices
    axes = [ax for ax, kws in d["疾患・病態の軸"].items() if any(norm(k) in text for k in kws)]
    terms = [t for t in d["重要語(疾患・所見)"] if norm(t) in text]
    in_correct = [t for t in terms if norm(t) in correct]
    related = [d["重要語(疾患・所見)"][t] for t in in_correct if d["重要語(疾患・所見)"][t]]
    return {
        "関連軸": "、".join(axes),
        "重要語": "、".join(terms),
        "正答に含まれる重要語": "、".join(in_correct),
        "関連疾患": "、".join(related),
        "図あり": "あり" if pd.notna(r.get("図")) else "",
    }


def tag_questions(qs: pd.DataFrame, d: dict) -> pd.DataFrame:
    tag_columns = ["関連軸", "重要語", "正答に含まれる重要語", "関連疾患", "図あり"]
    tags = pd.DataFrame(
        [tag_question(r, d) for _, r in qs.iterrows()],
        index=qs.index,
        columns=tag_columns,
    )
    return pd.concat([qs, tags], axis=1)


def axis_summary(tagged: pd.DataFrame, d: dict) -> pd.DataFrame:
    rows = []
    for ax in d["疾患・病態の軸"]:
        hit = tagged[tagged["関連軸"].str.contains(ax, na=False)]
        rows.append([ax, len(hit), "、".join(hit["問題ID"])])
    return pd.DataFrame(rows, columns=["関連軸", "問題数", "該当問題"])


def article_tables(questions: pd.DataFrame, tagged: pd.DataFrame, subject: str,
                   major: str, targets: list[str], year_min: int, year_max: int) -> tuple:
    major_questions = questions[
        questions["大項目"].eq(major) & questions["回"].between(year_min, year_max)
    ]
    target_questions = major_questions[major_questions["中項目"].isin(targets)]
    years = list(range(year_min, year_max + 1))
    major_counts = major_questions.groupby("回").size().reindex(years, fill_value=0)
    target_counts = target_questions.groupby("回").size().reindex(years, fill_value=0)
    measurable_years = int((major_counts > 0).sum())

    summary = pd.DataFrame([{
        "科目": subject,
        "大項目": major,
        "分析対象回": f"第{year_min}〜{year_max}回",
        "対象中項目数": len(targets),
        "対象問題数": len(target_questions),
        "大項目の問題数": len(major_questions),
        "大項目内割合(%)": round(len(target_questions) / len(major_questions) * 100, 1)
        if len(major_questions) else None,
        "対象が出題された回数": int((target_counts > 0).sum()),
        "集計可能回数": measurable_years,
    }])

    topic_rows = []
    for topic in targets:
        topic_questions = target_questions[target_questions["中項目"].eq(topic)]
        topic_rows.append({
            "中項目": topic,
            "対象期間出題数": len(topic_questions),
            "出題回数": int(topic_questions["回"].nunique()),
            "大項目内割合(%)": round(len(topic_questions) / len(major_questions) * 100, 1)
            if len(major_questions) else None,
        })
    topics = pd.DataFrame(topic_rows, columns=[
        "中項目", "対象期間出題数", "出題回数", "大項目内割合(%)"
    ])

    trend = pd.DataFrame({
        "回": years,
        "対象中項目の問題数": target_counts.to_numpy(),
        "大項目の問題数": major_counts.to_numpy(),
    })
    trend["大項目内割合(%)"] = trend.apply(
        lambda row: round(row["対象中項目の問題数"] / row["大項目の問題数"] * 100, 1)
        if row["大項目の問題数"] else None,
        axis=1,
    )

    example_columns = [
        "中項目", "問題ID", "問題", *CHOICE_COLS, "模範解答", "正答",
        "関連軸", "重要語", "正答に含まれる重要語", "関連疾患",
    ]
    if tagged.empty:
        examples = pd.DataFrame(columns=example_columns)
    else:
        ordered = tagged.assign(
            _question_number=pd.to_numeric(tagged["問題番号"], errors="coerce")
        ).sort_values(
            ["中項目", "回", "_question_number"],
            ascending=[True, False, True],
            kind="stable",
        )
        examples = ordered.drop_duplicates("中項目")[example_columns].reset_index(drop=True)

    return summary, topics, trend, examples


def cross_subject_analysis(question_sheets: dict[str, pd.DataFrame], source_subject: str,
                           extracted_terms: pd.DataFrame, subjects=None) -> tuple:
    """集計した語句が他科目の問題に現れる割合と、該当問題を返す。"""
    subject_list = list(question_sheets) if subjects is None else subjects
    words = []
    if "語" in extracted_terms:
        words = [(str(word), norm(word)) for word in extracted_terms["語"] if norm(word)]

    summary_rows = []
    yearly_rows = []
    period_rows = []
    question_rows = []
    for subject in subject_list:
        if subject == source_subject or subject not in question_sheets:
            continue
        questions = question_sheets[subject]
        matched_by_year = {}
        subject_name = SUBJECT_LABELS.get(subject, subject)

        for _, question in questions.iterrows():
            year = pd.to_numeric(question.get("回"), errors="coerce")
            if pd.isna(year):
                continue
            values = [question.get("問題", "")] + [question.get(col, "") for col in CHOICE_COLS]
            text = norm(" ".join("" if pd.isna(value) else str(value) for value in values))
            matched_words = [word for word, normalized in words if normalized in text]
            year = int(year)
            matched_by_year.setdefault(year, [0, 0])
            matched_by_year[year][0] += 1
            matched_by_year[year][1] += bool(matched_words)
            if matched_words:
                question_rows.append({
                    "科目": subject_name,
                    "回": year,
                    "問題ID": question.get("問題ID", ""),
                    "一致語句": "、".join(matched_words),
                    "問題": question.get("問題", ""),
                })

        total = sum(counts[0] for counts in matched_by_year.values())
        hits = sum(counts[1] for counts in matched_by_year.values())
        hit_terms = {
            term for row in question_rows
            if row["科目"] == subject_name
            for term in row["一致語句"].split("、")
        }
        summary_rows.append({
            "科目": subject_name,
            "対象問題数(全回)": total,
            "一致問題数(全回)": hits,
            "一致率(%)": round(hits / total * 100, 1) if total else 0.0,
            "一致語句数": len(hit_terms),
        })

        for year in sorted(matched_by_year):
            year_total, year_hits = matched_by_year[year]
            yearly_rows.append({
                "科目": subject_name,
                "回": year,
                "対象問題数": year_total,
                "一致問題数": year_hits,
                "一致率(%)": round(year_hits / year_total * 100, 1),
            })

        for period, low, high in (("第1〜24回", 1, 24), ("第25〜34回(直近10年)", 25, 34)):
            period_total = sum(
                counts[0] for year, counts in matched_by_year.items() if low <= year <= high
            )
            period_hits = sum(
                counts[1] for year, counts in matched_by_year.items() if low <= year <= high
            )
            period_rows.append({
                "科目": subject_name,
                "期間": period,
                "対象問題数": period_total,
                "一致問題数": period_hits,
                "一致率(%)": round(period_hits / period_total * 100, 1) if period_total else 0.0,
            })

    return (
        pd.DataFrame(summary_rows, columns=[
            "科目", "対象問題数(全回)", "一致問題数(全回)", "一致率(%)", "一致語句数"
        ]),
        pd.DataFrame(yearly_rows, columns=[
            "科目", "回", "対象問題数", "一致問題数", "一致率(%)"
        ]),
        pd.DataFrame(period_rows, columns=[
            "科目", "期間", "対象問題数", "一致問題数", "一致率(%)"
        ]),
        pd.DataFrame(question_rows, columns=["科目", "回", "問題ID", "一致語句", "問題"]),
    )


MEDICAL_TERMS_PATH = Path(__file__).parent / "medical_terms.txt"


def load_medical_terms() -> list[str]:
    if not MEDICAL_TERMS_PATH.exists():
        return []
    return [l.strip() for l in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_medical_terms(text: str) -> None:
    MEDICAL_TERMS_PATH.write_text(text.strip() + "\n", encoding="utf-8")


def analyze(freq_file, past_file, subject, major, top_n, year_min=25, year_max=34, min_hits=2, min_conc=0.5):
    freq, year_cols = load_freq(freq_file, subject)
    ranking = rank_middle(freq, year_cols, major, top_n)
    targets = ranking.loc[ranking["分析対象"], "中項目"].tolist()
    d = load_dictionary()
    # 語の出題回数は第1回からの全問題で数える。分析対象の抽出だけ対象回で絞る
    question_sheets = {
        name: load_questions(past_file, name, 1, 999)
        for name in sheet_names(past_file)
    }
    all_q = question_sheets[subject]
    qs = all_q[(all_q["大項目"] == major) & all_q["中項目"].isin(targets) & all_q["回"].between(year_min, year_max)]
    tagged = tag_questions(qs, d).sort_values(["中項目", "回", "問題ID"])
    article_summary, article_topics, article_trend, article_examples = article_tables(
        all_q, tagged, subject, major, targets, year_min, year_max
    )
    stop = {terms.norm(w) for w in d.get("除外語", [])}
    # 選択した大項目・その中項目名と同一の語は分析対象外
    exact_stop = {terms.norm(w) for w in pd.concat([pd.Series([major]), freq.loc[freq["大項目"] == major, "中項目"]]).dropna().unique()}
    related = {terms.norm(k): v for k, v in d["重要語(疾患・所見)"].items()}
    curated = load_medical_terms() + list(d["重要語(疾患・所見)"])
    freq_terms = terms.related_terms(all_q, major, targets, curated, stop, exact_stop, related, min_hits, min_conc)
    cross_summary, cross_yearly, cross_periods, cross_questions = cross_subject_analysis(
        question_sheets, subject, freq_terms
    )
    related_axis = cross_summary[
        cross_summary["科目"].isin(SUBJECT_LABELS.get(s, s) for s in RELATED_AXIS_SUBJECTS)
    ].reset_index(drop=True)
    return {
        "条件": pd.DataFrame(
            {"項目": ["科目", "大項目", "対象回(分析対象の問題)", "上位N", "分析対象の中項目", "語の出題数の集計範囲", "科目横断の判定", "比較期間"],
             "内容": [subject, major, f"第{year_min}〜{year_max}回", str(top_n), "、".join(targets), "第1回〜第34回の全問題(同大項目／科目全体)", f"抽出語{len(freq_terms)}語が比較先の問題文・選択肢に含まれるか", "第1〜24回と第25〜34回(直近10年)"]}),
        "記事向け要約": article_summary,
        "中項目別の出題状況": article_topics,
        "対象中項目の年度推移": article_trend,
        "記事向け問題例": article_examples,
        "解釈上の注意": pd.DataFrame({"内容": [
            "出題数・割合は指定期間の過去問を集計した記述値であり、将来の出題を予測するものではありません。",
            "科目横断は抽出語句が問題文・選択肢に現れるかを数えた指標です。複数科目の知識が必要な問題だとは判定しません。",
            "問題例は対象中項目ごとに指定期間内の直近回から自動選定しています。記事に掲載する前に問題文・正答・集計条件を確認してください。",
        ]}),
        "頻度ランキング": ranking,
        "関連軸集計": related_axis,
        "科目横断集計": cross_summary,
        "科目横断の年度推移": cross_yearly,
        "科目横断の時期比較": cross_periods,
        "科目横断該当問題": cross_questions,
        "疾患・病態軸集計": axis_summary(tagged, d),
        "対象問題一覧": tagged[["中項目", "問題ID", "問題", *CHOICE_COLS, "模範解答", "正答", "関連軸", "重要語", "正答に含まれる重要語", "関連疾患", "図あり"]],
        "関連語句(全回集計)": freq_terms,
    }


def search_term(past_file, subject, word):
    return terms.search_term(load_questions(past_file, subject, 1, 999), word)


def diagnose_term(freq_file, past_file, subject, major, top_n, word, min_hits=2, min_conc=0.5):
    freq, year_cols = load_freq(freq_file, subject)
    ranking = rank_middle(freq, year_cols, major, top_n)
    targets = ranking.loc[ranking["分析対象"], "中項目"].tolist()
    d = load_dictionary()
    all_q = load_questions(past_file, subject, 1, 999)
    stop = {terms.norm(w) for w in d.get("除外語", [])}
    exact_stop = {terms.norm(w) for w in pd.concat([pd.Series([major]), freq.loc[freq["大項目"] == major, "中項目"]]).dropna().unique()}
    curated = load_medical_terms() + list(d["重要語(疾患・所見)"])
    return terms.diagnose_term(all_q, word, major, targets, curated, stop, exact_stop, min_hits, min_conc)


def to_excel_bytes(result: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for i, (name, df) in enumerate(result.items()):
            df.to_excel(w, sheet_name=f"{i}_{name}"[:31], index=False)
            ws = w.book[f"{i}_{name}"[:31]]
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col) * 1.6
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width))
    return buf.getvalue()


def to_markdown(result: dict) -> str:
    out = ["# 出題傾向分析データ（レポート作成用）", ""]
    for name, df in result.items():
        out += [f"## {name}", "", df.fillna("").astype(str).to_csv(index=False, sep="|"), ""]
    return "\n".join(out)
