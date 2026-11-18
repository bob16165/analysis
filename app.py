import json
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as an
import article
import trends

st.set_page_config(page_title="出題傾向分析", layout="wide")
st.title("国試 出題傾向分析")

APP_DIR = Path(__file__).parent
SEARCH_DIRS = [APP_DIR, Path.home() / "Desktop"]


def find_default(name: str):
    return next((d / name for d in SEARCH_DIRS if (d / name).exists()), None)


def file_signature(file) -> str:
    if isinstance(file, (str, Path)):
        content = Path(file).read_bytes()
    elif hasattr(file, "getvalue"):
        content = file.getvalue()
    else:
        raise TypeError("入力ファイルの内容を読み取れません。")
    return hashlib.sha256(content).hexdigest()


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

tab_run, tab_search, tab_dict, tab_trend = st.tabs(
    ["分析", "語句検索", "辞書の編集", "出題トレンド"]
)

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
        st.session_state.pop("article_draft", None)

    result = st.session_state.get("result")
    if result:
        article_summary = result["記事向け要約"].iloc[0]
        st.subheader("記事の骨子")
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "対象中項目の出題数",
            f"{article_summary['対象問題数']} / {article_summary['大項目の問題数']}",
        )
        share = article_summary["大項目内割合(%)"]
        metric_columns[1].metric(
            "大項目内の割合",
            f"{share}%" if article_summary["大項目の問題数"] else "—",
        )
        metric_columns[2].metric(
            "対象が出題された回数",
            f"{article_summary['対象が出題された回数']} / {article_summary['集計可能回数']}",
        )
        metric_columns[3].metric("対象中項目数", article_summary["対象中項目数"])
        st.caption(
            f"{article_summary['科目']} / {article_summary['大項目']} / "
            f"{article_summary['分析対象回']}。割合は各回の大項目の問題数を分母にしています。"
        )
        if not article_summary["大項目の問題数"]:
            st.warning("選択した期間にこの大項目の過去問がないため、割合を算出できません。")

        st.subheader("中項目別の出題状況")
        st.dataframe(result["中項目別の出題状況"], width="stretch", hide_index=True)
        st.subheader("対象中項目の年度推移")
        trend = result["対象中項目の年度推移"]
        st.line_chart(trend.set_index("回")["大項目内割合(%)"])
        st.dataframe(trend, width="stretch", hide_index=True)
        st.subheader("中項目ごとの年度推移")
        topic_trend = result["中項目別の年度推移"]
        if not topic_trend.empty:
            topic_chart = topic_trend.pivot(index="回", columns="中項目", values="問題数")
            unavailable = topic_trend.groupby("回")["大項目の問題数"].first().eq(0)
            topic_chart.loc[unavailable, :] = float("nan")
            st.line_chart(topic_chart)
            st.caption("大項目の問題がない回は集計不可としてグラフを欠損値にしています。")
        st.dataframe(topic_trend, width="stretch", hide_index=True)
        st.subheader("中項目別の頻出語（対象期間）")
        st.caption(
            "問題文・全選択肢に2問以上出現した抽出候補から中項目ごとに上位3語を表示します。"
            "同じ問題内の繰り返しは1問として数え、誤答も含みます。原資料で確認してください。"
        )
        st.dataframe(result["中項目別の頻出語"], width="stretch", hide_index=True)
        with st.expander("記事で確認する問題例（各中項目の直近回から自動選定）"):
            st.caption("記事へ掲載する前に、問題文・正答・集計条件を原資料で確認してください。")
            st.dataframe(result["記事向け問題例"], width="stretch", hide_index=True)

        st.info(
            "出題割合は過去問の記述値で、将来の出題予測ではありません。"
            "科目横断は抽出語句の一致を数えた指標であり、複数科目の知識が必要な問題とは限りません。"
        )

        article_keys = {
            "記事向け要約", "中項目別の出題状況", "対象中項目の年度推移",
            "記事向け問題例", "中項目別の年度推移", "中項目別の頻出語", "解釈上の注意",
        }
        with st.expander("詳細データ（全表）"):
            for name, df in result.items():
                if name in article_keys:
                    continue
                st.subheader(name)
                st.dataframe(df, width="stretch", hide_index=True)
                if name == "科目横断の年度推移" and not df.empty:
                    st.line_chart(df.pivot(index="回", columns="科目", values="一致率(%)"))
        c1, c2 = st.columns(2)
        c1.download_button("Excelをダウンロード", an.to_excel_bytes(result), "出題傾向分析.xlsx")
        c2.download_button("AIレポート用Markdown", an.to_markdown(result), "出題傾向分析.md")

with tab_trend:
    st.caption(
        "出題数・出題割合・回帰傾向・変動係数は別々の指標です。"
        "割合の分母を表示し、分母0の回は割合の計算から除外します。"
    )
    trend_questions = an.load_questions(past_file, subject, 1, 999)
    available_rounds = sorted(
        pd.to_numeric(trend_questions["回"], errors="coerce").dropna().astype(int).unique()
    )
    middle_items = (
        freq.loc[freq["大項目"].eq(major), "中項目"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if not available_rounds:
        st.warning("選択科目に分析可能な過去問がありません。")
    else:
        c1, c2, c3 = st.columns(3)
        middle_choice = c1.selectbox(
            "分析する領域",
            ["大項目全体", *middle_items],
            key="trend_middle",
        )
        denominator_options = (
            ["科目全体"]
            if middle_choice == "大項目全体"
            else ["大項目", "科目全体"]
        )
        denominator_basis = c2.selectbox(
            "出題割合の分母",
            denominator_options,
            key="trend_denominator",
        )
        period_choice = c3.selectbox(
            "分析期間",
            ["全期間", "直近5回", "直近10回", "直近15回", "任意範囲"],
            key="trend_period",
        )

        latest_round = max(available_rounds)
        first_round = min(available_rounds)
        recent_counts = {"直近5回": 5, "直近10回": 10, "直近15回": 15}
        if period_choice == "全期間":
            y_lo, y_hi = first_round, latest_round
        elif period_choice in recent_counts:
            y_hi = latest_round
            y_lo = y_hi - recent_counts[period_choice] + 1
        else:
            c1, c2 = st.columns(2)
            y_lo = c1.number_input(
                "開始回", min_value=1, max_value=40, value=first_round, key="trend_year_min"
            )
            y_hi = c2.number_input(
                "終了回", min_value=1, max_value=40, value=latest_round, key="trend_year_max"
            )

        st.caption(
            "増減判定は、有効8回以上・回帰推定差±5ポイント以上・R² 0.30以上を条件とします。"
            "前後差だけでは増減を判定せず、条件未達や収録回の欠損は判定保留にします。"
            "変動係数は平均出題割合が0%または有効データ不足の場合、算出しません。"
        )
        if y_lo > y_hi:
            st.error("開始回は終了回以前にしてください。")
        else:
            trend_signature = hashlib.sha256(
                f"{file_signature(freq_file)}:{file_signature(past_file)}".encode("ascii")
            ).hexdigest()
            trend_key = (
                trend_signature, subject, major, middle_choice, denominator_basis, y_lo, y_hi
            )
            if st.button("トレンドを分析", type="primary"):
                st.session_state["trend_result"] = trends.analyze_trends(
                    trend_questions,
                    subject,
                    major,
                    None if middle_choice == "大項目全体" else middle_choice,
                    denominator_basis,
                    int(y_lo),
                    int(y_hi),
                    middle_items,
                )
                st.session_state["trend_result_key"] = trend_key

            trend_result = st.session_state.get("trend_result")
            if trend_result and st.session_state.get("trend_result_key") == trend_key:
                st.subheader("選択領域の指標")
                st.dataframe(
                    trend_result["選択領域指標"], width="stretch", hide_index=True
                )
                st.subheader("期間別の出題割合")
                st.dataframe(
                    trend_result["期間別出題割合"], width="stretch", hide_index=True
                )
                st.subheader("年度別出題割合")
                yearly = trend_result["年度別推移"]
                st.line_chart(yearly.set_index("回")["出題割合(%)"])
                st.dataframe(yearly, width="stretch", hide_index=True)
                st.caption(
                    "年度別推移表の問題IDは「国家試験回／科目／問題番号」で構成しています。"
                    "分母0の回は出題割合を空欄にしています。"
                )
                st.subheader("前期・後期比較")
                st.bar_chart(
                    trend_result["前期・後期比較"]
                    .set_index("期間")["出題割合(%)"]
                )
                st.dataframe(
                    trend_result["前期・後期比較"], width="stretch", hide_index=True
                )
                st.subheader("増加・減少・変動ランキング")
                ranking_tabs = st.tabs(
                    ["増加領域", "減少領域", "出題頻度の変動"]
                )
                for ranking_tab, key in zip(
                    ranking_tabs,
                    ["増加領域ランキング", "減少領域ランキング", "出題頻度変動ランキング"],
                ):
                    with ranking_tab:
                        st.dataframe(
                            trend_result[key], width="stretch", hide_index=True
                        )
                c1, c2 = st.columns(2)
                c1.download_button(
                    "トレンド分析をExcelでダウンロード",
                    an.to_excel_bytes(trend_result),
                    "出題トレンド分析.xlsx",
                )
                c2.download_button(
                    "トレンド分析をMarkdownでダウンロード",
                    an.to_markdown(trend_result),
                    "出題トレンド分析.md",
                )
            elif trend_result:
                st.info("分析条件または入力データが変わりました。再分析してください。")

        st.subheader("note記事のたたき台")
        st.caption(
            "テンプレートを使った文章生成です。掲載前に文章、数値、問題の出典・正答、"
            "権利関係を確認してください。問題文や選択肢の全文は記事に含めません。"
        )
        mea = st.text_area("MEAの紹介文（任意）", key="article_mea")
        mock_results = st.text_area("自校の模試結果・集計条件（任意）", key="article_mock")
        next_theme = st.text_input("次回の分析テーマ（任意）", key="article_next")
        if st.button("記事のたたき台を作成"):
            try:
                st.session_state["article_draft"] = article.generate_article(
                    result, mea=mea, next_theme=next_theme, mock_results=mock_results
                )
            except ValueError as e:
                st.error(str(e))
        if "article_draft" in st.session_state:
            draft = st.text_area("記事本文（編集できます）", key="article_draft", height=600)
            count = len(draft)
            st.caption(f"文字数：{count:,}字（改行・記号を含む。目安：3,000〜4,000字）")
            if not 3000 <= count <= 4000:
                st.warning("文字数が目安の範囲外です。内容を確認し、必要に応じて編集してください。")
            if "［" in draft and "］" in draft:
                st.warning("未入力の項目があります。公開前に［　］内を入力してください。")
            st.download_button("記事をMarkdownでダウンロード", draft, "note記事.md", mime="text/markdown")

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
