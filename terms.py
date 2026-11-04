"""医療用語を分割せずに抽出し、大項目・中項目との関連の強さと全回での出題数を集計する。

候補語は過去問全体の文字n-gramから作り(形態素解析に依存しない)、
同じ出題数で長い語に含まれる断片(閉包でない語)を落とす。
"""
import re
from collections import defaultdict

import pandas as pd

CHOICE_IDX = "1234"
_PAREN_LATIN = re.compile(r"[\(（][A-Za-z0-9\s\-'.,・/]+[\)）]")
_RUN = re.compile(r"[一-龥ァ-ヶー・]+")
MAX_LEN = 12
RECENT_MIN = 25  # 中項目が付与されている第25〜34回(直近10年)。第1〜24回は大項目のみ
STEM, CORRECT, WRONG = 1, 2, 4


def norm(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = _PAREN_LATIN.sub("", s)
    return re.sub(r"[\s　]", "", s)


def is_generic(term: str, stop: set[str]) -> bool:
    """除外語そのもの、または除外語だけで構成される語。"""
    if term in stop:
        return True
    rest = term
    for s in sorted(stop, key=len, reverse=True):
        rest = rest.replace(s, "")
    return not rest


def _grams(text: str) -> set[str]:
    out = set()
    for run in _RUN.findall(text):
        n = len(run)
        for i in range(n):
            for j in range(i + 2, min(n, i + MAX_LEN) + 1):
                g = run[i:j]
                if g[0] in "ー・" or g[-1] == "・":
                    continue
                out.add(g)
    return out


def question_texts(r) -> dict:
    ca = set(re.findall(r"\d", str(r["模範解答"])))
    return {
        STEM: [norm(r["問題"])],
        CORRECT: [norm(r[f"選択肢{i}"]) for i in CHOICE_IDX if i in ca],
        WRONG: [norm(r[f"選択肢{i}"]) for i in CHOICE_IDX if i not in ca],
    }


def build_index(all_q: pd.DataFrame, curated: list[str]):
    """語 -> {行位置: 出現ビット(問題文/正答/誤答)}。curatedは平仮名を含む語も部分一致で追加。
    あわせて、語が単独の語(平仮名・記号で区切られた連続部分そのもの)として出た問題の集合も返す。"""
    idx = defaultdict(dict)
    standalone = defaultdict(set)
    cur = [norm(c) for c in curated if norm(c)]
    for pos, (_, r) in enumerate(all_q.iterrows()):
        for bit, texts in question_texts(r).items():
            seen = set()
            for t in texts:
                seen |= _grams(t)
                seen |= {c for c in cur if c in t}
                for run in _RUN.findall(t):
                    standalone[run].add(pos)
            for g in seen:
                idx[g][pos] = idx[g].get(pos, 0) | bit
    return idx, standalone


def closed_terms(idx: dict, min_df: int = 2) -> set[str]:
    """出題数が同じ長い語に含まれる断片を除いた語集合。"""
    df = {w: len(v) for w, v in idx.items() if len(v) >= min_df}
    dropped = set()
    for w, n in df.items():
        for sub in (w[1:], w[:-1]):
            if len(sub) >= 2 and df.get(sub) == n:
                dropped.add(sub)
    # 平仮名を含む語(手動登録語)に含まれる断片も除く
    for w in [w for w in df if not _RUN.fullmatch(w)]:
        n = df[w]
        for g in df:
            if g != w and len(g) < len(w) and g in w and df[g] == n:
                dropped.add(g)
    return set(df) - dropped


def related_terms(all_q: pd.DataFrame, major: str, targets: list[str], curated: list[str],
                  stop: set[str], exact_stop: set[str], related: dict,
                  min_hits: int = 2, min_conc: float = 0.5, top: int = 100) -> pd.DataFrame:
    """対象中項目に出る語のうち、大項目・中項目との関連が強いものを出題数つきで返す。"""
    all_q = all_q.reset_index(drop=True)
    idx, standalone = build_index(all_q, curated)
    cand = closed_terms(idx)
    cur = {norm(c) for c in curated}
    in_major = (all_q["大項目"] == major).to_numpy()
    in_target = (in_major & all_q["中項目"].isin(targets)).to_numpy()
    kai = all_q["回"].to_numpy()
    mid = all_q["中項目"].to_numpy()
    rows = []
    for w in cand:
        if w in exact_stop or is_generic(w, stop):
            continue
        d = idx[w]
        n_all = len(d)
        # 断片(例:「パー」「骨神経」)を避けるため、単独の語として出る割合が半分以上の語だけ採用
        if w not in cur and len(standalone.get(w, ())) < 0.5 * n_all:
            continue
        pm = [p for p in d if in_major[p]]
        pt = [p for p in d if in_target[p]]
        if len(pt) < min_hits:
            continue
        conc_major = len(pm) / n_all
        if conc_major < min_conc:
            continue
        years = sorted({int(kai[p]) for p in pm})
        by_mid = pd.Series([mid[p] for p in pm]).value_counts()
        rows.append({
            "語": w,
            "関連疾患": related.get(w, ""),
            "対象中項目での出題数(全回)": len(pt),
            "対象中項目での集中度": round(len(pt) / n_all, 2),
            "大項目内の出題数(全回)": len(pm),
            "大項目での集中度": round(conc_major, 2),
            "最も多い中項目": f"{by_mid.index[0]}({by_mid.iloc[0]})",
            "うち第1〜24回": sum(1 for p in pm if kai[p] < RECENT_MIN),
            "うち第25〜34回(直近10年)": sum(1 for p in pm if kai[p] >= RECENT_MIN),
            "うち正答として": sum(1 for p in pm if d[p] & CORRECT),
            "出題された回の数": len(years),
            "出題された回": "、".join(map(str, years)),
            "科目全体の出題数(全回)": n_all,
        })
    cols = ["語", "関連疾患", "対象中項目での出題数(全回)", "対象中項目での集中度", "大項目内の出題数(全回)",
            "大項目での集中度", "最も多い中項目", "うち第1〜24回", "うち第25〜34回(直近10年)", "うち正答として",
            "出題された回の数", "出題された回", "科目全体の出題数(全回)"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["対象中項目での出題数(全回)", "対象中項目での集中度", "大項目内の出題数(全回)"],
                        ascending=False, kind="stable")
    return df.head(top).reset_index(drop=True)


def search_term(all_q: pd.DataFrame, word: str) -> pd.DataFrame:
    """語句が問題文・選択肢に含まれる全問題を返す(検証用)。"""
    w = norm(word)
    rows = []
    for _, r in all_q.iterrows():
        t = question_texts(r)
        roles = [name for bit, name in ((STEM, "問題文"), (CORRECT, "正答"), (WRONG, "誤答"))
                 if any(w in x for x in t[bit])]
        if roles:
            rows.append([r["問題ID"], r["大項目"], r["中項目"], "・".join(roles), r["問題"]])
    return pd.DataFrame(rows, columns=["問題ID", "大項目", "中項目", "登場箇所", "問題"])


def diagnose_term(all_q: pd.DataFrame, word: str, major: str, targets: list[str], curated: list[str],
                  stop: set[str], exact_stop: set[str], min_hits: int = 2, min_conc: float = 0.5) -> list[str]:
    """語が抽出表に出ない理由を返す(手作業で見つけた語との差の確認用)。"""
    w = norm(word)
    all_q = all_q.reset_index(drop=True)
    idx, standalone = build_index(all_q, curated)
    d = idx.get(w, {})
    if not d:
        return ["過去問(問題文・選択肢)に出現しない(表記ゆれ・括弧内の英語表記の除去・全角半角を確認)"]
    reasons = []
    if w in exact_stop or is_generic(w, stop):
        reasons.append("除外語または大項目/中項目名と同一のため除外")
    if w not in closed_terms(idx):
        reasons.append("出題数が同じ、より長い語の断片として扱われ除外(長い語の方で抽出されている可能性)")
    cur = {norm(c) for c in curated}
    if w not in cur and len(standalone.get(w, ())) < 0.5 * len(d):
        reasons.append("単独の語としての出現が半分未満(断片扱い)。medical_terms.txtへの登録で解消")
    in_major = (all_q["大項目"] == major).to_numpy()
    in_target = in_major & all_q["中項目"].isin(targets).to_numpy()
    n_t = sum(1 for p in d if in_target[p])
    n_m = sum(1 for p in d if in_major[p])
    if n_t < min_hits:
        reasons.append(f"対象中項目での出題数が{n_t}問で下限{min_hits}未満")
    if n_m / len(d) < min_conc:
        reasons.append(f"大項目への集中度が{n_m / len(d):.2f}で下限{min_conc}未満(他の大項目でも多く出る語)")
    return reasons or ["条件は満たす。上位100件の表示件数で切れている可能性"]
