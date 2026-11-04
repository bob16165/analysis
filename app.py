import json
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as an

st.set_page_config(page_title="出題傾向分析", layout="wide")
st.title("国試 出題傾向分析")

APP_DIR = Path(__file__).parent
SEARCH_DIRS = [APP_DIR, Path.home() / "Desktop"]


def find_default(name: str):
    return next((d / name for d in SEARCH_DIRS if (d / name).exists()), None)


with st.sidebar:
    st.header("入力ファイル")
    freq_up = st.file_uploader("出題頻度.xlsx", type="xlsx")
    past_up = st.file_uploader("過去問.xlsx", type="xlsx")
    freq_file = freq_up or find_default("出題頻度.xlsx")
    past_file = past_up or find_default("過去問.xlsx")
    st.caption("未指定時はこのフォルダ→デスクトップの順で探します")

if not (freq_file and past_file):
    st.info("2つのExcelを指定してください。")
    st.stop()

tab_run, tab_search, tab_dict = st.tabs(["分析", "語句検索", "辞書の編集"])

with tab_run:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    subject = c1.selectbox("科目", an.sheet_names(freq_file))
    freq, year_cols = an.load_freq(freq_file, subject)
    major = c2.selectbox("大項目", list(dict.fromkeys(freq["大項目"])))
    top_n = c3.number_input("上位N位まで（同順位は含む）", 1, 10, 3)
    y_lo, y_hi = c4.slider("分析対象の回", 1, 40, (25, 34))
    min_hits = c5.number_input("語の最低出題数(対象中項目・全回)", 1, 10, 2)
    min_conc = c6.slider("大項目への集中度の下限", 0.0, 1.0, 0.5, 0.05,
                         help="その語が出る全問題のうち、選択した大項目の問題が占める割合。下げるほど他領域にも出る語が増える")

    st.caption(
        "科目横断は、抽出語が比較先科目の問題文・選択肢に含まれる割合を集計します。"
        "語句の一致は関連の手掛かりであり、複数科目の知識が必須の問題だと証明するものではありません。"
    )

    if st.button("分析する", type="primary"):
        st.session_state["result"] = an.analyze(freq_file, past_file, subject, major, int(top_n), y_lo, y_hi, int(min_hits), float(min_conc))

    result = st.session_state.get("result")
    if result:
        for name, df in result.items():
            st.subheader(name)
            st.dataframe(df, width="stretch", hide_index=True)
            if name == "科目横断の年度推移" and not df.empty:
                st.line_chart(df.pivot(index="回", columns="科目", values="一致率(%)"))
        c1, c2 = st.columns(2)
        c1.download_button("Excelをダウンロード", an.to_excel_bytes(result), "出題傾向分析.xlsx")
        c2.download_button("AIレポート用Markdown", an.to_markdown(result), "出題傾向分析.md")

with tab_search:
    st.caption("語句が問題文・選択肢に含まれる全問題(第1回〜)を、科目内で検索します。抽出漏れの確認用です。")
    s_subject = st.selectbox("科目", an.sheet_names(past_file), key="s_subject")
    word = st.text_input("語句")
    if word:
        hit = an.search_term(past_file, s_subject, word)
        yrs = hit["問題ID"].str.extract(r"第(\d+)回")[0].astype(float)
        st.write(f"{len(hit)}問(第1〜24回: {int((yrs <= 24).sum())}問 / 第25〜34回: {int((yrs >= 25).sum())}問)")
        if "result" in st.session_state:
            st.write("**抽出表に出ない理由(上で選択中の科目・大項目・条件で判定)**")
            for reason in an.diagnose_term(freq_file, past_file, subject, major, int(top_n), word, int(min_hits), float(min_conc)):
                st.write("- " + reason)
        st.dataframe(hit, width="stretch", hide_index=True)

with tab_dict:
    st.caption("除外語・軸の判定キーワード・重要語→関連疾患の対応(dictionary.json)と、分割したくない医療用語(medical_terms.txt)を編集できます。保存後に再分析してください。")
    terms_text = st.text_area("medical_terms.txt(1行1語)", "\n".join(an.load_medical_terms()), height=250)
    if st.button("医療用語を保存"):
        an.save_medical_terms(terms_text)
        st.success("保存しました")
    d = an.load_dictionary()
    text = st.text_area("dictionary.json", json.dumps(d, ensure_ascii=False, indent=2), height=500)
    if st.button("保存"):
        try:
            an.save_dictionary(json.loads(text))
            st.success("保存しました")
        except json.JSONDecodeError as e:
            st.error(f"JSONエラー: {e}")
