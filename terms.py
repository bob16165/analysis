"""医療用語を分割せずに抽出し、全回(第1回〜)での出題回数を集計する。"""
import re
from collections import defaultdict

import pandas as pd

CHOICE_IDX = "1234"
_PAREN_LATIN = re.compile(r"[\(（][A-Za-z0-9\s\-'.,・/]+[\)）]")
_DASH = re.compile(r"[―—─]{2,}")
_TOKEN = re.compile(r"[一-龥ァ-ヶー・]{2,14}")


def norm(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = _PAREN_LATIN.sub("", s)
    return re.sub(r"[\s　]", "", s)


def _is_generic(term: str, stop: set[str]) -> bool:
    if term in stop:
        return True
    # 除外語だけで構成される語(例: 症状所見)も除く
    rest = term
    for s in sorted(stop, key=len, reverse=True):
        rest = rest.replace(s, "")
    return not rest


def build_vocab(all_q: pd.DataFrame, curated: list[str], dict_terms: list[str], stop: set[str], exact_stop: set[str] = frozenset()) -> set[str]:
    """医療用語の語彙。手動リスト＋辞書＋過去問の短い選択肢(それ自体が用語であることが多い)。"""
    from collections import Counter

    c = Counter()
    for _, r in all_q.iterrows():
        seen = set()
        for i in CHOICE_IDX:
            for part in _DASH.split(norm(r[f"選択肢{i}"])):
                if 2 <= len(part) <= 16 and not re.search(r"[。、，,．.0-9０-９]", part) and re.search(r"[一-龥ァ-ヶ]", part):
                    seen.add(part)
        c.update(seen)
    vocab = {t for t, n in c.items() if n >= 2}
    vocab |= {norm(t) for t in curated + dict_terms + sorted(exact_stop) if norm(t)}
    return {t for t in vocab if not _is_generic(t, stop)}


def make_extractor(vocab: set[str], stop: set[str], exact_stop: set[str] = frozenset()):
    ordered = sorted(vocab, key=len, reverse=True)
    pat = re.compile("|".join(map(re.escape, ordered))) if ordered else None

    def extract(text: str) -> set[str]:
        found = set()
        rest = text
        if pat:
            found = set(pat.findall(text))
            rest = pat.sub("|", text)
        for tok in _TOKEN.findall(rest):
            if not _is_generic(tok, stop):
                found.add(tok)
        return {t for t in found if not _is_generic(t, stop) and t not in exact_stop}

    return extract


def _texts(r) -> dict:
    ca = set(re.findall(r"\d", str(r["模範解答"])))
    return {
        "stem": norm(r["問題"]),
        "correct": [norm(r[f"選択肢{i}"]) for i in CHOICE_IDX if i in ca],
        "wrong": [norm(r[f"選択肢{i}"]) for i in CHOICE_IDX if i not in ca],
    }


def frequent_terms(target_q: pd.DataFrame, all_q: pd.DataFrame, extract, major: str,
                   related: dict, min_hits: int = 2, top: int = 60) -> pd.DataFrame:
    """対象問題で頻出する語を抽出し、同大項目・科目全体の全回での出題数を付ける。"""
    hits = defaultdict(lambda: {"stem": set(), "correct": set(), "wrong": set()})
    for _, r in target_q.iterrows():
        t = _texts(r)
        for w in extract(t["stem"]):
            hits[w]["stem"].add(r["問題ID"])
        for role in ("correct", "wrong"):
            for s in t[role]:
                for w in extract(s):
                    hits[w][role].add(r["問題ID"])
    cands = {w: v for w, v in hits.items() if len(v["stem"] | v["correct"] | v["wrong"]) >= min_hits}

    pre = [(r, _texts(r)) for _, r in all_q.iterrows()]
    rows = []
    for w, v in cands.items():
        tq = v["stem"] | v["correct"] | v["wrong"]
        stat = {"同": defaultdict(set), "全": defaultdict(set)}
        for r, t in pre:
            inq = w in t["stem"] or any(w in s for s in t["correct"] + t["wrong"])
            if not inq:
                continue
            role = "正答" if any(w in s for s in t["correct"]) else "他"
            for scope in ("全", "同") if r["大項目"] == major else ("全",):
                stat[scope]["all"].add(r["問題ID"])
                if r["回"] <= 27:
                    stat[scope]["old"].add(r["問題ID"])
                else:
                    stat[scope]["new"].add(r["問題ID"])
                if role == "正答":
                    stat[scope]["ans"].add(r["問題ID"])
                stat[scope]["years"].add(int(r["回"]))
        s, a = stat["同"], stat["全"]
        years = sorted(s["years"])
        rows.append([
            w, related.get(w, ""), len(tq), len(v["stem"]), len(v["correct"]), len(v["wrong"]),
            len(s["all"]), len(s["old"]), len(s["new"]), len(s["ans"]), len(years),
            "、".join(map(str, years)), len(a["all"]), len(a["old"]), len(a["new"]),
        ])
    cols = ["語", "関連疾患", "対象問題での登場数", "問題文", "正答選択肢", "誤答選択肢",
            f"{major}・全回の出題数", "うち第1〜27回", "うち第28〜34回", "うち正答として", "出題された回の数", "出題された回",
            "科目全体・全回の出題数", "科目全体 第1〜27回", "科目全体 第28〜34回"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["対象問題での登場数", f"{major}・全回の出題数"], ascending=False, kind="stable")
    return df.head(top).reset_index(drop=True)
