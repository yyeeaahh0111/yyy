
import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="關聯規則分析工具", layout="wide")
st.title("萬用關聯規則分析系統")


def parse_transaction_cell(cell, separator=","):
    if pd.isna(cell):
        return []
    return [item.strip() for item in str(cell).split(separator) if item.strip()]


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


uploaded_file = st.file_uploader("請上傳 Excel 檔案", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.write("### 原始資料預覽（前 5 筆）")
        st.dataframe(df.head(), use_container_width=True)

        all_columns = df.columns.tolist()
        target_col = st.selectbox(
            "請選擇包含『交易項目』的欄位（例如：game）",
            all_columns
        )

        separator = st.selectbox(
            "請選擇交易項目的分隔符號",
            [",", ";", "|", "/"],
            index=0
        )

        st.sidebar.header("演算法參數")
        min_support = st.sidebar.slider("最小支持度 (Support)", 0.01, 0.5, 0.05)
        min_confidence = st.sidebar.slider("最小信心度 (Confidence)", 0.1, 1.0, 0.5)
        min_lift = st.sidebar.slider("最小提升度 (Lift)", 1.0, 5.0, 1.0)

        transactions = df[target_col].apply(lambda x: parse_transaction_cell(x, separator))
        transactions = transactions[transactions.apply(len) > 0]

        if len(transactions) == 0:
            st.warning("⚠️ 選定欄位中沒有可用的交易資料，請確認欄位內容與分隔符號。")
        else:
            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_array, columns=te.columns_)

            st.write("---")
            st.subheader("🎮 指定 A 遊戲，查看共玩榜單")

            game_list = sorted(df_encoded.columns.tolist())

            if len(game_list) == 0:
                st.warning("⚠️ 沒有可分析的遊戲項目。")
            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    selected_game = st.selectbox("請選擇 A 遊戲", game_list)

                with col2:
                    top_n = st.number_input(
                        "顯示前幾名",
                        min_value=5,
                        max_value=min(50, len(game_list)),
                        value=min(10, len(game_list)),
                        step=1
                    )

                if st.button("產生共玩榜單"):
                    players_who_play_A = df_encoded[df_encoded[selected_game]]
                    total_A_players = len(players_who_play_A)

                    if total_A_players == 0:
                        st.warning(f"⚠️ 沒有玩家玩過「{selected_game}」")
                    else:
                        result_rows = []

                        for game_b in df_encoded.columns:
                            if game_b == selected_game:
                                continue

                            both_count = int(players_who_play_A[game_b].sum())
                            ratio = both_count / total_A_players if total_A_players > 0 else 0

                            result_rows.append({
                                "遊戲A": selected_game,
                                "遊戲B": game_b,
                                "A玩家數": total_A_players,
                                "A且B玩家數": both_count,
                                "比例": ratio,
                                "百分比(%)": round(ratio * 100, 2)
                            })

                        ranking_df = pd.DataFrame(result_rows)
                        ranking_df = ranking_df.sort_values(
                            by=["比例", "A且B玩家數"],
                            ascending=False
                        ).head(int(top_n))

                        st.success(
                            f"✅ 在玩過「{selected_game}」的 {total_A_players} 位玩家中，以下是最常一起遊玩的遊戲。"
                        )

                        st.dataframe(
                            ranking_df[["遊戲A", "遊戲B", "A玩家數", "A且B玩家數", "百分比(%)"]],
                            use_container_width=True
                        )

                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_df = ranking_df.sort_values("百分比(%)", ascending=True)
                        ax.barh(plot_df["遊戲B"], plot_df["百分比(%)"])
                        ax.set_xlabel("百分比 (%)")
                        ax.set_ylabel("遊戲 B")
                        ax.set_title(f"玩過 {selected_game} 的玩家，也玩過其他遊戲的比例")

                        for i, v in enumerate(plot_df["百分比(%)"]):
                            ax.text(v + 0.5, i, f"{v}%", va="center")

                        st.pyplot(fig)

                        csv_data = convert_df_to_csv(ranking_df)
                        st.download_button(
                            label="📥 下載共玩榜單 CSV",
                            data=csv_data,
                            file_name=f"{selected_game}_coplay_ranking.csv",
                            mime="text/csv"
                        )

            st.write("---")
            st.subheader("📊 Apriori 關聯規則分析")

            if st.button("開始執行關聯規則分析"):
                frequent_itemsets = apriori(
                    df_encoded,
                    min_support=min_support,
                    use_colnames=True
                )

                if frequent_itemsets.empty:
                    st.error("❌ 支持度設定太高，找不到任何頻繁項目集。")
                else:
                    rules = association_rules(
                        frequent_itemsets,
                        metric="lift",
                        min_threshold=min_lift
                    )
                    rules = rules[rules["confidence"] >= min_confidence]

                    if rules.empty:
                        st.warning("⚠️ 找不到符合條件的關聯規則。")
                    else:
                        display_df = rules[
                            ["antecedents", "consequents", "support", "confidence", "lift"]
                        ].copy()

                        display_df["antecedents"] = display_df["antecedents"].apply(lambda x: ", ".join(list(x)))
                        display_df["consequents"] = display_df["consequents"].apply(lambda x: ", ".join(list(x)))
                        display_df["support"] = (display_df["support"] * 100).round(2)
                        display_df["confidence"] = (display_df["confidence"] * 100).round(2)
                        display_df["lift"] = display_df["lift"].round(2)

                        display_df = display_df.rename(columns={
                            "antecedents": "前項",
                            "consequents": "後項",
                            "support": "支持度(%)",
                            "confidence": "信心度(%)",
                            "lift": "提升度"
                        })

                        st.success(f"✅ 分析完成！找到 {len(display_df)} 條規則")
                        st.dataframe(
                            display_df.sort_values("提升度", ascending=False),
                            use_container_width=True
                        )

    except Exception as e:
        st.error(f"執行發生錯誤：{e}")
else:
    st.info("💡 請先上傳 Excel 檔案開始分析。")
