import io
import json
import re
from pathlib import Path

import pandas as pd

import terms

DICT_PATH = Path(__file__).parent / "dictionary.json"
CHOICE_COLS = ["選択肢1", "選択肢2", "選択肢3", "選択肢4"]


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
    tags = pd.DataFrame([tag_question(r, d) for _, r in qs.iterrows()], index=qs.index)
    return pd.concat([qs, tags], axis=1)


def axis_summary(tagged: pd.DataFrame, d: dict) -> pd.DataFrame:
    rows = []
    for ax in d["疾患・病態の軸"]:
        hit = tagged[tagged["関連軸"].str.contains(ax, na=False)]
        rows.append([ax, len(hit), "、".join(hit["問題ID"])])
    return pd.DataFrame(rows, columns=["関連軸", "問題数", "該当問題"])


MEDICAL_TERMS_PATH = Path(__file__).parent / "medical_terms.txt"


def load_medical_terms() -> list[str]:
    if not MEDICAL_TERMS_PATH.exists():
        return []
    return [l.strip() for l in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_medical_terms(text: str) -> None:
    MEDICAL_TERMS_PATH.write_text(text.strip() + "\n", encoding="utf-8")


def analyze(freq_file, past_file, subject, major, top_n, year_min=28, year_max=34, min_hits=2, min_conc=0.5):
    freq, year_cols = load_freq(freq_file, subject)
    ranking = rank_middle(freq, year_cols, major, top_n)
    targets = ranking.loc[ranking["分析対象"], "中項目"].tolist()
    d = load_dictionary()
    # 語の出題回数は第1回からの全問題で数える。分析対象の抽出だけ対象回で絞る
    all_q = load_questions(past_file, subject, 1, 999)
    qs = all_q[(all_q["大項目"] == major) & all_q["中項目"].isin(targets) & all_q["回"].between(year_min, year_max)]
    tagged = tag_questions(qs, d).sort_values(["中項目", "回", "問題ID"])
    stop = {terms.norm(w) for w in d.get("除外語", [])}
    # 選択した大項目・その中項目名と同一の語は分析対象外
    exact_stop = {terms.norm(w) for w in pd.concat([pd.Series([major]), freq.loc[freq["大項目"] == major, "中項目"]]).dropna().unique()}
    related = {terms.norm(k): v for k, v in d["重要語(疾患・所見)"].items()}
    curated = load_medical_terms() + list(d["重要語(疾患・所見)"])
    freq_terms = terms.related_terms(all_q, major, targets, curated, stop, exact_stop, related, min_hits, min_conc)
    return {
        "条件": pd.DataFrame(
            {"項目": ["科目", "大項目", "対象回(分析対象の問題)", "上位N", "分析対象の中項目", "語の出題数の集計範囲"],
             "内容": [subject, major, f"第{year_min}〜{year_max}回", str(top_n), "、".join(targets), "第1回〜第34回の全問題(同大項目／科目全体)"]}),
        "頻度ランキング": ranking,
        "関連軸集計": axis_summary(tagged, d),
        "対象問題一覧": tagged[["中項目", "問題ID", "問題", *CHOICE_COLS, "模範解答", "正答", "関連軸", "重要語", "正答に含まれる重要語", "関連疾患", "図あり"]],
        "関連語句(全回集計)": freq_terms,
    }


def search_term(past_file, subject, word):
    return terms.search_term(load_questions(past_file, subject, 1, 999), word)


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
