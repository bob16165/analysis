"""分析結果から、確認・加筆を前提とした記事のたたき台を作る。"""

import pandas as pd


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _number(value) -> str:
    text = _text(value)
    return text[:-2] if text.endswith(".0") else text


def generate_article(result: dict, mea: str = "", next_theme: str = "",
                     mock_results: str = "") -> str:
    summary = result["記事向け要約"].iloc[0]
    topics = result["中項目別の出題状況"].sort_values(
        "対象期間出題数", ascending=False, kind="stable"
    )
    trend = result["中項目別の年度推移"]
    examples = result["記事向け問題例"]
    topic_terms = result["中項目別の頻出語"]
    subject = _text(summary["科目"])
    major = _text(summary["大項目"])
    period = _text(summary["分析対象回"])
    denominator = int(summary["大項目の問題数"])
    measurable = int(summary["集計可能回数"])
    if not denominator:
        raise ValueError("対象期間に大項目の問題がないため、記事を生成できません。")
    if topics.empty:
        raise ValueError("分析対象の中項目がないため、記事を生成できません。")

    ranking = []
    for rank, (_, row) in enumerate(topics.iterrows(), 1):
        ranking.append(
            f"{rank}. {row['中項目']}：{int(row['対象期間出題数'])}問"
            f"（大項目全体の{_number(row['大項目内割合(%)'])}%、"
            f"{measurable}回中{int(row['出題回数'])}回で出題）"
        )

    available = trend[trend["大項目の問題数"] > 0]
    recurrence = []
    for topic in topics["中項目"]:
        rows = available[available["中項目"].eq(topic)]
        present = rows.loc[rows["問題数"] > 0, "回"].astype(int).tolist()
        absent = rows.loc[rows["問題数"] == 0, "回"].astype(int).tolist()
        def years(values):
            return "、".join(f"第{year}回" for year in values) if values else "なし"
        recurrence.append(f"- {topic}：出題 {years(present)}／出題なし {years(absent)}")

    pivot = trend.pivot(index="回", columns="中項目", values="問題数")
    yearly_lines = ["| 試験回 | 大項目の問題数 | " + " | ".join(map(str, pivot.columns)) + " |",
                    "| --- | ---: | " + " | ".join("---:" for _ in pivot.columns) + " |"]
    for year, row in pivot.iterrows():
        total = int(trend.loc[trend["回"].eq(year), "大項目の問題数"].iloc[0])
        values = [str(int(row[topic])) if total else "集計不可" for topic in pivot.columns]
        yearly_lines.append(f"| 第{year}回 | {total} | " + " | ".join(values) + " |")

    example_lines = []
    for _, row in examples.iterrows():
        example_lines.append(
            f"- {row['中項目']}：{row['問題ID']}／模範解答番号："
            f"{_text(row['模範解答']) or '要確認'}／確認された重要語："
            f"{_text(row['重要語']) or '自動抽出なし（原資料で確認）'}"
        )
    if not example_lines:
        example_lines.append("- 対象期間の問題例はありません。原資料を確認してください。")

    term_lines = []
    for topic in topics["中項目"]:
        topic_rows = topic_terms[topic_terms["中項目"].eq(topic)].sort_values("順位")
        words = [
            f"「{row['語']}」{int(row['出現問題数'])}問"
            for _, row in topic_rows.iterrows()
        ]
        term_lines.append(f"- {topic}：" + ("、".join(words) if words else "条件を満たす語なし"))

    yearly_summary = result["対象中項目の年度推移"]
    measured = yearly_summary[yearly_summary["大項目の問題数"] > 0]
    midpoint = len(measured) // 2
    if midpoint:
        first, second = measured.iloc[:midpoint], measured.iloc[midpoint:]
        def comparison(part):
            count = int(part["対象中項目の問題数"].sum())
            total = int(part["大項目の問題数"].sum())
            return f"{count}/{total}問（{count / total * 100:.1f}%）"
        change = (
            f"集計可能な回を前半{len(first)}回と後半{len(second)}回に分けると、"
            f"対象中項目の合計はそれぞれ{comparison(first)}、{comparison(second)}です。"
            "前後の問題数を合算した割合であり、各回の割合の単純平均ではありません。"
        )
    else:
        change = "集計可能な回が少ないため、期間を前後に分けた比較は控えます。"

    mea_text = mea.strip() or "［MEAの機能・提供内容を入力してください］"
    next_text = next_theme.strip() or "［次回の分析テーマを入力してください］"
    mock_text = mock_results.strip() or "［自校の模試結果・対象回・集計条件を入力してください］"

    return f"""# {subject}「{major}」の出題傾向を読む

国家試験の対策を考えるとき、出題数の多い領域から優先して学ぶのは自然な判断です。ただし、合計の多さだけを見ていると、毎回問われている領域と、特定の回に集中している領域を同じものとして扱ってしまうかもしれません。

今回は{period}の{subject}「{major}」を対象に、中項目の出題状況を整理します。教員にとっては授業や模試の振り返りに、学生にとっては学習計画を見直すための材料になればと思います。過去の出題を記述する分析であり、次回の出題を予測するものではありません。

## 第1章　出題数と割合――まずは全体像をつかむ

対象期間の「{major}」は{denominator}問。そのうち分析対象の中項目は{int(summary['対象問題数'])}問、全体の{_number(summary['大項目内割合(%)'])}%を占めました。対象としているのは出題頻度表の全期間で上位に入った中項目です。以下は、それらを今回の対象期間の問題数で並べ直したもので、同数の場合は一覧上の順序で表示しています。

{chr(10).join(ranking)}

割合の分母は科目全体ではなく、対象期間の「{major}」の問題数です。また、同順位を含めて選ぶため、分析対象は3領域とは限りません。上位に入らなかった中項目も分母に含まれるので、表示した割合の合計は必ずしも100%になりません。

数の多い領域は、学習時間を配分する際の一つの手掛かりになります。一方、少ない領域を学習対象から外す理由にはなりません。どこに時間をかけるかと、どこを理解しておくかは、分けて考えたいところです。

## 第2章　年度別推移――同じ構成が続くとは限らない

各回の問題数を並べると、合計だけでは分からない出題のばらつきが見えてきます。次の表はグラフを作る際にも使える基礎データです。アプリ画面では中項目ごとの推移をグラフで確認できます。

{chr(10).join(yearly_lines)}

大項目の問題がある回について、出題の有無を整理すると次のようになります。

{chr(10).join(recurrence)}

「出題なし」は対象資料の該当回にその中項目の問題がなかったことを表します。大項目の問題自体がない回は分母がなく、出題の有無を比較できる回には含めていません。資料の収録状況や分類も、あわせて確認が必要です。

継続して出題されている領域があれば、基本事項を定着させる教材として扱いやすいでしょう。一方、出題のない回がある領域では、直近の一回だけで重要度を判断しないことが大切です。問題数の増減は、領域の重要性が変わったことを直接示すわけではありません。

## 第3章　見落とされやすい変化――合計の内側を確かめる

{change}

ここで注意したいのは、前半と後半で比較した数値が違っていても、それだけで「増加傾向」や「重視されるようになった」と結論づけられないことです。対象回数が限られているうえ、各回の問題数や、対象に含まれない中項目の出題数によっても割合は変わります。差が小さい場合には、数問の違いで印象が変わることもあります。

もう一つの視点は、同じ中項目の中で何を問うているかです。例えば、名称を知っていれば解ける問題と、所見や検査結果を組み合わせて判断する問題では、必要な学習の深さが異なります。中項目の名前と件数が同じでも、問題文を確認しなければ、その違いは見えません。

教員が検討する際には、年度別の表を見ながら問題例を読み、授業で扱う知識と判断過程が対応しているかを確認するとよいでしょう。学生にとっても、正解したかどうかだけでなく、選択肢を選んだ根拠を説明できるかを振り返ることが、理解を確かめる手掛かりになります。

今回の数値から言えることと、問題を読んだうえで立てる仮説は区別しておきます。分類の変更や資料の欠落があれば比較の意味も変わるため、気になる差が見つかったときほど、元の問題と集計条件に立ち返ることが必要です。

## 第4章　実際の出題内容を確認する

まず、対象期間の各中項目で繰り返し現れた語句を見てみます。問題文と全選択肢を対象に、2問以上に現れた候補から上位3語を表示しています。同じ問題で複数回現れても1問として数えます。

{chr(10).join(term_lines)}

語句は表記を正規化して機械的に抽出した候補で、同数の場合は語順で並べています。誤答の選択肢も含むため、出現回数は正答数や主題の頻度を意味しません。抽出漏れや不自然な語もあり得るので、元の問題で確認してください。

数字の背景を確かめるため、各中項目の対象期間内で最も新しい回から、問題例を一つずつ挙げます。重要語は辞書による機械的な抽出で、問題の主題や正答そのものを保証するものではありません。

{chr(10).join(example_lines)}

各問題は出典の問題番号から原資料を参照し、問題文、選択肢、模範解答と図の有無を確認してください。掲載に当たっては利用条件と権利を確認し、疾患名・病態・検査法や正答の根拠を自分の言葉で補足します。正答番号が複数ある問題や、図が判断に必要な問題は、番号だけを見ても内容を十分に理解できません。

具体的な問題を読む際は、正しい選択肢だけでなく、他の選択肢がなぜ適切でないのかも確認します。教員は誤答が生じた理由を授業内容と照らし合わせ、学生は知識の不足と読み違いを分けて整理すると、次の学習につなげやすくなります。

## 第5章　自校の模試結果と照合する

国家試験でよく出題された領域と、自校の学生が十分に理解できている領域は、必ずしも一致しません。出題頻度と模試の正答率を組み合わせることで、授業や復習の優先順位を検討できます。

{mock_text}

照合する際は、国試と模試で中項目の分類が対応しているか、模試の問題数が何問か、受験者の範囲が同じかを確認します。少数の問題から得られた正答率は変動しやすく、問題の難易度も揃っているとは限りません。数値の差をそのまま学生の能力や授業の効果とみなさず、誤答の内容や授業での扱いと合わせて読む必要があります。

学生自身も、全体の正答率だけで安心したり落ち込んだりするのではなく、自分が間違えた問題を領域ごとに見直せます。なぜその答えを選んだかを書き出し、基礎知識と判断のどちらに課題があるかを確かめると、復習の方向が定まりやすくなります。

## 第6章　分析を教育活動につなげるために

出題傾向の集計は、表を作ること自体が目的ではありません。問題を読み直し、自校の結果と照合し、授業や学習の改善につなげるところに意味があります。

そのための手段として、MEAについても紹介します。{mea_text}

利用を検討する際には、集計の対象や分類、結果をどのように教育活動で使えるかを確認したいところです。ツールの数値だけで判断を完結させず、教員と学生が問題の内容を振り返る材料として活用することが重要です。

## 次回予告

次回は「{next_text}」を取り上げる予定です。今回と同じように出題数を確認したうえで、領域ごとの違いや問題の内容を見ていきます。国試対策を一回ごとの結果で終わらせず、学習と授業を継続的に見直すための材料として、引き続き整理していきます。
"""
