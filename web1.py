import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# =========================
# matplotlib 繁體中文字型設定
# =========================
plt.rcParams["font.family"] = [
    "Microsoft JhengHei",
    "PingFang TC",
    "Noto Sans TC",
    "Heiti TC",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="關聯規則分析工具",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 自訂 CSS（繁中字型 + UI 美化）
# =========================
st.markdown("""
<style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"], div, p, span, label,
    input, button, textarea, select, h1, h2, h3, h4, h5, h6 {
        font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", sans-serif !important;
    }

    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #4f8bf9 0%, #7b61ff 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(79, 139, 249, 0.20);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .hero p {
        margin-top: 0.6rem;
        font-size: 1rem;
        opacity: 0.95;
    }

    .section-card {
        background: white;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 1.2rem;
    }

    .small-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #e5edf8;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 800;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f8bf9 0%, #7b61ff 100%);
        color: white;
        box-shadow: 0 8px 18px rgba(79, 139, 249, 0.22);
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 1px solid #e5e7eb;
    }

    .stDataFrame, .stTable {
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        border: none;
        height: 1px;
        background: #dbeafe;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 工具函式
# =========================
def parse_transaction_cell(cell, separator=","):
    if pd.isna(cell):
        return []
    return [item.strip() for item in str(cell).split(separator) if item.strip()]


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data
def load_excel(file):
    return pd.read_excel(file)


# =========================
# 標題區
# =========================
st.markdown("""
<div class="hero">
    <h1>🎮 萬用關聯規則分析系統</h1>
    <p>
        上傳 Excel 後，即可快速分析玩家共玩趨勢與 Apriori 關聯規則，
        找出最常一起出現的遊戲組合與推薦線索。
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# 側邊欄設定
# =========================
st.sidebar.markdown("## ⚙️ 分析設定")
st.sidebar.caption("請先上傳資料，再調整演算法參數")

min_support = st.sidebar.slider("最小支持度（Support）", 0.01, 0.5, 0.05)
min_confidence = st.sidebar.slider("最小信賴度（Confidence）", 0.1, 1.0, 0.5)
min_lift = st.sidebar.slider("最小提升度（Lift）", 1.0, 5.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.info(
    "支持度越高，代表規則越常出現；信賴度越高，代表前項推導後項的可信程度越高；提升度大於 1 表示有正向關聯。"
)

# =========================
# 資料上傳區
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="small-title">📂 資料上傳</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("請上傳 Excel 檔案", type=["xlsx", "xls"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    try:
        df = load_excel(uploaded_file)

        # 基本資訊
        total_rows = len(df)
        total_cols = len(df.columns)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">資料筆數</div>
                <div class="metric-value">{total_rows}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">欄位數量</div>
                <div class="metric-value">{total_cols}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">檔案狀態</div>
                <div class="metric-value">已載入</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 原始資料預覽
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">👀 原始資料預覽</div>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 欄位設定
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">🧩 交易欄位設定</div>', unsafe_allow_html=True)

        all_columns = df.columns.tolist()
        colx, coly = st.columns([2, 1])

        with colx:
            target_col = st.selectbox(
                "請選擇包含「交易項目」的欄位（例如：game）",
                all_columns
            )

        with coly:
            separator = st.selectbox(
                "請選擇分隔符號",
                [",", ";", "|", "/"],
                index=0
            )

        st.markdown('</div>', unsafe_allow_html=True)

        transactions = df[target_col].apply(lambda x: parse_transaction_cell(x, separator))
        transactions = transactions[transactions.apply(len) > 0]

        if len(transactions) == 0:
            st.warning("⚠️ 選定欄位中沒有可用的交易資料，請確認欄位內容與分隔符號。")
        else:
            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_array, columns=te.columns_)

            tab1, tab2 = st.tabs(["🎯 共玩排行榜", "📊 關聯規則分析"])

            # =========================
            # 共玩排行榜
            # =========================
            with tab1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="small-title">🎯 指定 A 遊戲，查看共玩排行榜</div>', unsafe_allow_html=True)

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

                    if st.button("🚀 產生共玩排行榜"):
                        players_who_play_A = df_encoded[df_encoded[selected_game]]
                        total_A_players = len(players_who_play_A)

                        if total_A_players == 0:
                            st.warning(f"⚠️ 沒有玩家玩過「{selected_game}」。")
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
                                    "同時玩A與B的玩家數": both_count,
                                    "比例": ratio,
                                    "百分比（%）": round(ratio * 100, 2)
                                })

                            ranking_df = pd.DataFrame(result_rows)
                            ranking_df = ranking_df.sort_values(
                                by=["比例", "同時玩A與B的玩家數"],
                                ascending=False
                            ).head(int(top_n))

                            st.success(
                                f"✅ 在玩過「{selected_game}」的 {total_A_players} 位玩家中，以下是最常一起遊玩的遊戲。"
                            )

                            left, right = st.columns([1.25, 1])

                            with left:
                                st.dataframe(
                                    ranking_df[["遊戲A", "遊戲B", "A玩家數", "同時玩A與B的玩家數", "百分比（%）"]],
                                    use_container_width=True,
                                    height=420
                                )

                            with right:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                plot_df = ranking_df.sort_values("百分比（%）", ascending=True)
                                ax.barh(plot_df["遊戲B"], plot_df["百分比（%）"])
                                ax.set_xlabel("百分比（%）")
                                ax.set_ylabel("遊戲 B")
                                ax.set_title(f"玩過 {selected_game} 的玩家之共玩比例")
                                ax.grid(axis="x", linestyle="--", alpha=0.3)

                                for i, v in enumerate(plot_df["百分比（%）"]):
                                    ax.text(v + 0.3, i, f"{v}%", va="center", fontsize=9)

                                plt.tight_layout()
                                st.pyplot(fig)

                            csv_data = convert_df_to_csv(ranking_df)
                            st.download_button(
                                label="📥 下載共玩排行榜 CSV",
                                data=csv_data,
                                file_name=f"{selected_game}_共玩排行榜.csv",
                                mime="text/csv"
                            )

                st.markdown('</div>', unsafe_allow_html=True)

            # =========================
            # 關聯規則分析
            # =========================
            with tab2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="small-title">📊 Apriori 關聯規則分析</div>', unsafe_allow_html=True)

                st.caption(
                    f"目前條件：支持度 ≥ {min_support:.2f}／信賴度 ≥ {min_confidence:.2f}／提升度 ≥ {min_lift:.2f}"
                )

                if st.button("📈 開始執行關聯規則分析"):
                    frequent_itemsets = apriori(
                        df_encoded,
                        min_support=min_support,
                        use_colnames=True
                    )

                    if frequent_itemsets.empty:
                        st.error("❌ 支持度設定過高，找不到任何頻繁項目集。")
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
                                "support": "支持度（%）",
                                "confidence": "信賴度（%）",
                                "lift": "提升度"
                            })

                            result_df = display_df.sort_values("提升度", ascending=False)

                            st.success(f"✅ 分析完成！共找到 {len(result_df)} 條規則。")

                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                height=500
                            )

                            csv_rules = convert_df_to_csv(result_df)
                            st.download_button(
                                label="📥 下載關聯規則 CSV",
                                data=csv_rules,
                                file_name="關聯規則分析結果.csv",
                                mime="text/csv"
                            )

                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"執行時發生錯誤：{e}")

else:
    st.info("💡 請先上傳 Excel 檔案，再開始分析。")
