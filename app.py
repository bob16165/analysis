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

tab_run, tab_dict = st.tabs(["分析", "辞書の編集"])

with tab_run:
    c1, c2, c3, c4, c5 = st.columns(5)
    subject = c1.selectbox("科目", an.sheet_names(freq_file))
    freq, year_cols = an.load_freq(freq_file, subject)
    major = c2.selectbox("大項目", list(dict.fromkeys(freq["大項目"])))
    top_n = c3.number_input("上位N位まで（同順位は含む）", 1, 10, 3)
    y_lo, y_hi = c4.slider("分析対象の回", 1, 40, (28, 34))
    min_hits = c5.number_input("頻出語の最低登場問題数", 1, 10, 2)

    if st.button("分析する", type="primary"):
        st.session_state["result"] = an.analyze(freq_file, past_file, subject, major, int(top_n), y_lo, y_hi, int(min_hits))

    result = st.session_state.get("result")
    if result:
        for name, df in result.items():
            st.subheader(name)
            st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Excelをダウンロード", an.to_excel_bytes(result), "出題傾向分析.xlsx")
        c2.download_button("AIレポート用Markdown", an.to_markdown(result), "出題傾向分析.md")

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
