import json
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as an
import article
import repeats as repeat_analysis
import standard_history as sh
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

tab_run, tab_search, tab_dict, tab_trend, tab_repeats, tab_standard = st.tabs(
    ["分析", "語句検索", "辞書の編集", "出題トレンド", "再出題・類題", "出題基準別実績"]
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

with tab_repeats:
    review_database = APP_DIR / "analysis_review.sqlite3"
    current_question_file_signature = file_signature(past_file)
    st.caption(
        "問題文・選択肢・模範解答・正答の正規化一致をA候補、文字3-gramで絞った文字列類似度"
        f"（初期閾値{repeat_analysis.MIN_SIMILARITY:.2f}）をB候補として提示します。"
        "類似度だけで同一問題・類題とは確定せず、人が確認した判定と履歴をSQLiteに保存します。"
    )
    st.warning(
        "問題データを更新した場合は再同期してください。内容に変更があった問題に関係する判定は"
        "再確認待ちとなり、人による判定は自動処理で上書きされません。"
    )
    if review_database.exists():
        saved_signature = repeat_analysis.get_source_signature(review_database)
        if saved_signature and saved_signature != current_question_file_signature:
            st.warning("過去問.xlsxが前回同期時から変更されています。判定前に再同期してください。")
        elif not saved_signature:
            st.warning("前回同期したExcelを特定できません。最新データを同期してください。")
    if st.button("過去問を同期して候補を抽出", type="primary"):
        question_sheets = {
            name: an.load_questions(past_file, name, 1, 999)
            for name in an.sheet_names(past_file)
        }
        try:
            sync_result = repeat_analysis.refresh_review_database(
                review_database,
                question_sheets,
                source_signature=current_question_file_signature,
            )
            st.session_state["repeat_sync_result"] = sync_result
        except ValueError as exc:
            st.error(str(exc))

    if "repeat_sync_result" in st.session_state:
        st.subheader("直近の同期結果")
        st.dataframe(
            pd.DataFrame([st.session_state["repeat_sync_result"]]),
            width="stretch",
            hide_index=True,
        )

    if review_database.exists():
        st.subheader("保存済みの候補と判定")
        review_counts = repeat_analysis.get_review_counts(review_database)
        if not review_counts.empty:
            st.dataframe(review_counts, width="stretch", hide_index=True)
        review_status = st.selectbox(
            "表示する判定状態",
            ["未確認・再確認待ち", *repeat_analysis.REVIEW_STATUSES, "全て"],
            key="repeat_review_status",
        )
        review_offset = st.number_input(
            "候補一覧の開始位置",
            min_value=0,
            max_value=100000,
            value=0,
            step=100,
            key="repeat_review_offset",
        )
        candidates = repeat_analysis.list_review_candidates(
            review_database, review_status, limit=100, offset=int(review_offset)
        )
        st.dataframe(candidates, width="stretch", hide_index=True)
        repeat_export = {
            "抽出条件": pd.DataFrame([
                {"項目": "対象", "内容": "同期済みの全過去問"},
                {"項目": "A候補", "内容": "正規化した問題文・選択肢・模範解答・正答が一致"},
                {
                    "項目": "B候補",
                    "内容": (
                        f"文字列類似度{repeat_analysis.MIN_SIMILARITY:.2f}以上、"
                        f"本文共通3-gram{repeat_analysis.MIN_SHARED_NGRAMS}個以上、"
                        f"問題文共通3-gram{repeat_analysis.MIN_SHARED_STEM_NGRAMS}個以上、"
                        f"問題文類似度{repeat_analysis.MIN_STEM_SIMILARITY:.2f}以上"
                    ),
                },
                {"項目": "表示状態", "内容": review_status},
                {"項目": "表示開始位置", "内容": int(review_offset)},
                {"項目": "表示件数上限", "内容": 100},
                {"項目": "自動判定に関する注意", "内容": "候補は人が確認するまで確定扱いしない"},
            ]),
            "候補一覧（表示中）": candidates,
            "問題データ変更履歴": repeat_analysis.list_question_changes(review_database),
            "関係判定履歴": repeat_analysis.list_relation_history(review_database),
        }
        repeat_export_col1, repeat_export_col2 = st.columns(2)
        repeat_export_col1.download_button(
            "再出題・類題候補をExcelでダウンロード",
            an.to_excel_bytes(repeat_export),
            "再出題・類題候補.xlsx",
        )
        repeat_export_col2.download_button(
            "再出題・類題候補をMarkdownでダウンロード",
            an.to_markdown(repeat_export),
            "再出題・類題候補.md",
        )
        if candidates.empty:
            st.info("この条件の候補はありません。")
        else:
            candidate_by_id = {
                row["relation_id"]: row for row in candidates.to_dict(orient="records")
            }

            def candidate_label(relation_id):
                row = candidate_by_id[relation_id]
                return (
                    f"{row['question_id_a']} ⇄ {row['question_id_b']} / "
                    f"{row['review_status']} / 自動候補 {row['auto_type'] or 'なし'}"
                )

            selected_relation_id = st.selectbox(
                "確認する関係",
                list(candidate_by_id),
                format_func=candidate_label,
                key="repeat_selected_relation",
            )
            detail = repeat_analysis.get_relation_detail(
                review_database, selected_relation_id
            )
            relation = detail["relation"]
            st.caption(
                f"自動候補: {relation['auto_type'] or 'なし'} / "
                f"類似度: {relation['similarity'] if relation['similarity'] is not None else '—'} / "
                f"根拠: {relation['evidence']}"
            )
            for side_key, side_label in (("a", "問題A"), ("b", "問題B")):
                side = detail[side_key]
                with st.expander(f"{side_label}: {side['problem_id']}", expanded=True):
                    if not side["active"]:
                        st.warning("現在の入力データにはこの問題がありません。")
                    if side["changed"] and side["previous"]:
                        st.warning("前回確認時から問題内容が変更されています。")
                        st.write("変更前", side["previous"])
                    st.write("現在", side["current"])

            review_options = ["未判定", *repeat_analysis.REVIEW_TYPES]
            default_review = (
                relation["reviewed_type"] or relation["auto_type"] or "未判定"
            )
            review_value = st.selectbox(
                "人による判定（A:同一問題 / B:類似問題 / C:関連問題）",
                review_options,
                index=review_options.index(default_review),
                key=f"repeat_review_type_{selected_relation_id}",
            )
            review_note = st.text_area(
                "判定根拠・メモ",
                value=relation["review_note"],
                key=f"repeat_review_note_{selected_relation_id}",
            )
            if st.button("判定を保存", key=f"repeat_save_{selected_relation_id}"):
                if review_value == "未判定":
                    st.error("保存する判定を選択してください。")
                else:
                    repeat_analysis.save_relation_review(
                        review_database, selected_relation_id, review_value, review_note
                    )
                    st.success("判定と履歴を保存しました。")

        with st.expander("関連問題（C）を手動で追加"):
            query = st.text_input(
                "問題IDの一部で検索（例: 第28回/整形外科）",
                key="repeat_manual_query",
            )
            st.dataframe(
                repeat_analysis.list_current_questions(
                    review_database, query=query, limit=100
                ),
                width="stretch",
                hide_index=True,
            )
            manual_col_a, manual_col_b = st.columns(2)
            manual_id_a = manual_col_a.text_input("問題ID A", key="repeat_manual_id_a")
            manual_id_b = manual_col_b.text_input("問題ID B", key="repeat_manual_id_b")
            manual_note = st.text_input("関連性の根拠", key="repeat_manual_note")
            if st.button("C:関連問題として追加", key="repeat_add_manual"):
                try:
                    repeat_analysis.add_manual_related_pair(
                        review_database, manual_id_a.strip(), manual_id_b.strip(), manual_note
                    )
                    st.success("関連問題と履歴を保存しました。")
                except ValueError as exc:
                    st.error(str(exc))

        with st.expander("判定履歴"):
            st.dataframe(
                repeat_analysis.list_relation_history(review_database),
                width="stretch",
                hide_index=True,
            )
        with st.expander("問題データの変更前後"):
            st.dataframe(
                repeat_analysis.list_question_changes(review_database),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("まだレビュー用データベースがありません。過去問を同期してください。")

with tab_standard:
    st.caption(
        "「記載なし」を含む大項目・中項目は旧出題基準由来、それ以外は現行出題基準として扱います。"
        "既存の分類・頻度集計は変更せず、頻度表または過去問で確認できた項目の実績を比較します。"
        "頻度区分は対象期間の問題数、出題回数は異なる国家試験回の数です。"
    )
    st.info(
        "公式出題基準の全項目が揃っていないため、公式基準全体に対する網羅率は算出しません。"
        "以下の項目出題実績率の分母は、出題頻度表または過去問から確認できた項目数です。"
    )
    standard_subject = st.selectbox(
        "科目", an.sheet_names(freq_file), key="standard_subject"
    )
    standard_freq, standard_year_columns = an.load_freq(freq_file, standard_subject)
    standard_questions = an.load_questions(past_file, standard_subject, 1, 999)
    expected_standard_rounds = [
        int(column[1:-1]) for column in standard_year_columns
    ]
    if not expected_standard_rounds:
        st.warning("出題頻度表に対象回の列がありません。")
    else:
        standard_col1, standard_col2, standard_col3 = st.columns(3)
        standard_col4, standard_col5, standard_col6 = st.columns(3)
        min_standard_round = min(expected_standard_rounds)
        max_standard_round = max(expected_standard_rounds)
        period_start = standard_col1.number_input(
            "開始回",
            min_value=min_standard_round,
            max_value=max_standard_round,
            value=min_standard_round,
            key="standard_period_start",
        )
        period_end = standard_col2.number_input(
            "終了回",
            min_value=min_standard_round,
            max_value=max_standard_round,
            value=max_standard_round,
            key="standard_period_end",
        )
        recent_count = standard_col3.number_input(
            "直近の回数", min_value=1, max_value=40, value=10, key="standard_recent_count"
        )
        bin_first = standard_col4.number_input(
            "第1区分の上限", min_value=1, max_value=100, value=1, key="standard_bin_first"
        )
        bin_second = standard_col5.number_input(
            "第2区分の上限", min_value=2, max_value=100, value=3, key="standard_bin_second"
        )
        bin_third = standard_col6.number_input(
            "第3区分の上限", min_value=3, max_value=100, value=5, key="standard_bin_third"
        )
        if st.button("出題基準区分別に集計", type="primary"):
            if not (bin_first < bin_second < bin_third):
                st.error("頻度区分の上限は小さい順に設定してください。")
            elif period_start > period_end:
                st.error("開始回は終了回以前にしてください。")
            else:
                standard_result = sh.analyze_standard_history(
                    standard_questions,
                    standard_freq,
                    standard_subject,
                    expected_standard_rounds,
                    int(period_start),
                    int(period_end),
                    int(recent_count),
                    (int(bin_first), int(bin_second), int(bin_third)),
                )
                st.session_state["standard_result"] = standard_result
                st.session_state["standard_result_key"] = (
                    file_signature(freq_file),
                    file_signature(past_file),
                    standard_subject,
                    int(period_start),
                    int(period_end),
                    int(recent_count),
                    int(bin_first),
                    int(bin_second),
                    int(bin_third),
                )

        current_standard_key = (
            file_signature(freq_file),
            file_signature(past_file),
            standard_subject,
            int(period_start),
            int(period_end),
            int(recent_count),
            int(bin_first),
            int(bin_second),
            int(bin_third),
        )
        standard_result = st.session_state.get("standard_result")
        if standard_result and st.session_state.get("standard_result_key") == current_standard_key:
            st.subheader("出題基準区分別の実績")
            st.dataframe(
                standard_result["出題基準区分別集計"],
                width="stretch",
                hide_index=True,
            )
            standard_summary = standard_result["出題基準区分別集計"]
            if not standard_summary.empty:
                st.bar_chart(
                    standard_summary.set_index("出題基準区分")[
                        "対象期間の項目出題実績率(%)"
                    ]
                )
            st.subheader("出題頻度区分別の項目数")
            st.dataframe(
                standard_result["出題頻度区分別集計"],
                width="stretch",
                hide_index=True,
            )
            frequency_chart = standard_result["出題頻度区分別集計"].pivot(
                index="対象期間の出題頻度区分",
                columns="出題基準区分",
                values="項目数",
            )
            st.bar_chart(frequency_chart)
            st.subheader("年度別の区分別出題")
            annual_standard = standard_result["年度別の区分別出題"]
            if not annual_standard.empty:
                annual_chart = annual_standard.pivot(
                    index="回", columns="出題基準区分", values="問題数"
                )
                st.line_chart(annual_chart)
            st.dataframe(annual_standard, width="stretch", hide_index=True)
            st.subheader("項目別出題実績と問題ID")
            st.dataframe(
                standard_result["項目別出題実績"],
                width="stretch",
                hide_index=True,
            )
            with st.expander("分類未確定の問題"):
                st.dataframe(
                    standard_result["分類未確定問題"],
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("集計条件"):
                st.dataframe(
                    standard_result["分析条件"],
                    width="stretch",
                    hide_index=True,
                )
            standard_download_col1, standard_download_col2 = st.columns(2)
            standard_download_col1.download_button(
                "区分別実績をExcelでダウンロード",
                an.to_excel_bytes(standard_result),
                "出題基準区分別実績.xlsx",
            )
            standard_download_col2.download_button(
                "区分別実績をMarkdownでダウンロード",
                an.to_markdown(standard_result),
                "出題基準区分別実績.md",
            )
        elif standard_result:
            st.info("分析条件または入力データが変わりました。再分析してください。")

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
