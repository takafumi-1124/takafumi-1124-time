# ==============================================
# 🌱 ESG投資意思決定アプリ（Streamlit Cloud最終完全版）
# ==============================================

# === seaborn-deepエラー完全回避（Streamlit Cloud安定版） ===
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    plt.style.use("seaborn-deep")
except OSError:
    print("⚠ seaborn-deep style not available on Streamlit Cloud. Using default style instead.")
    plt.style.use("default")
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["axes.unicode_minus"] = False

# === 通常のimport ===
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
import japanize_matplotlib
import matplotlib.style as mstyle
import json
import os

# ==============================================
# 📘 共通関数群
# ==============================================
def get_dynamic_scale_labels(left: str, right: str):
    return [
        f"{left}が非常に重要",
        f"{left}がかなり重要",
        f"{left}が少し重要",
        "同じくらい重要",
        f"{right}が少し重要",
        f"{right}がかなり重要",
        f"{right}が非常に重要"
    ]

def get_dynamic_label_to_value(left: str, right: str):
    values = [7, 5, 3, 1, 1/3, 1/5, 1/7]
    labels = get_dynamic_scale_labels(left, right)
    return dict(zip(labels, values))

def ahp_calculation(pairwise_matrix):
    n = pairwise_matrix.shape[0]
    geo_means = np.prod(pairwise_matrix, axis=1) ** (1/n)
    priorities = geo_means / np.sum(geo_means)
    weighted_sum = np.dot(pairwise_matrix, priorities)
    lamda_max = np.sum(weighted_sum / priorities) / n
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    CR = CI / RI_dict[n]
    return priorities, CR

# ==============================================
# Streamlit 本体
# ==============================================
st.set_page_config(page_title="ESG投資意思決定", layout="centered")
st.title("🌱 ESG投資意思決定サイト")

tabs = st.tabs(["① ユーザー情報", "② ESG優先度測定", "③ 投資提案"])
all_priorities = {}

# --- ① ユーザー情報 ---
with tabs[0]:
    st.header("ユーザー情報入力")
    name = st.text_input("名前", key="name")
    age = st.number_input("年齢", 10, 100, 20, key="age")
    job = st.text_input("あなたの職業を入力してください", placeholder="例：大学生")

# # --- ② Big Five ---
# with tabs[1]:
#     # --- ESG投資とは？説明セクション ---
#     st.header("📘 ESG投資とは？")
#     st.markdown("""
#     ### ESG投資とは？
#     ESG投資とは、企業を「環境（E）・社会（S）・ガバナンス（G）」の3つの視点から評価し、
#     長期的に安定した成長が期待できる企業を選ぶ投資方法です。

#     - **環境（E）**：CO₂削減、再エネ、廃棄物管理  
#     - **社会（S）**：働きやすさ、人権、多様性  
#     - **ガバナンス（G）**：不祥事防止、経営の健全性
#     """)

#     st.markdown("""
#     ### ESGは“社会貢献のための投資”ではありません
#     ESGは「企業のリスク管理能力」や「持続可能性」を測る指標です。  
#     不祥事・規制違反・離職率の高さなどを避け、安定した企業を選ぶための基準となります。
#     """)

#     st.markdown("""
#     ### このアプリで行うこと
#     あなたのESGに対する価値観をAHP法を使って数値化し、  
#     その結果をもとに最適な投資ポートフォリオを提案します。
#     """)

#     # --- Big Five 診断 ---
#     st.header("Big Five 診断")
#     st.markdown("以下の項目について、「非常に当てはまる（5）」〜「まったく当てはまらない（1）」で評価してください。")

#     bigfive_items = [
#         ("無口な", "外向性", True), ("社交的", "外向性", False), ("話好き", "外向性", False),
#         ("いい加減な", "誠実性", True), ("計画性のある", "誠実性", False),
#         ("心配性", "情緒不安定性", False), ("憂鬱な", "情緒不安定性", False),
#         ("好奇心が強い", "開放性", False), ("独創的な", "開放性", True),
#         ("短気", "調和性", True), ("温和な", "調和性", False)
#     ]

#     scores = {}
#     for idx, (item, trait, reverse) in enumerate(bigfive_items, start=1):
#         label = f"**{idx}. {item}**"
#         val = st.slider(label, 1, 5, 3, key=f"bf_{idx}_{item}")
#         if reverse:
#             val = 6 - val
#         scores.setdefault(trait, []).append(val)

#     trait_scores = {k: np.mean(v) for k, v in scores.items()}
#     st.subheader("Big Five スコア")
#     st.dataframe(pd.DataFrame(trait_scores.items(), columns=["性格特性", "スコア"]))

# --- ③ AHP ---
with tabs[2]:
    st.header("ESG優先度測定")
    st.markdown("""
    以下では2つの項目を比較し、どちらをどの程度重視するかを選んでください。  
    """)

    st.markdown("""
    **整合性比率（CR）とは？**  
    比較の一貫性を示す指標です。CRが **0.15以下** なら信頼できる判断とされます。
    """)

    with st.expander("🔍 ESGカテゴリーの補足説明", expanded=True):
        st.markdown("""
        **環境（Environment）**  
        企業が環境への負荷を減らす取り組みをしているか  

        **社会（Social）**  
        企業が人や社会に対して配慮しているか  

        **ガバナンス（Governance）**  
        企業の経営の仕組みや透明性が適切に整えられているか
        """)

    # --- E/S/G 比較 ---
    labels_main = ['環境', '社会', 'ガバナンス']
    matrix_main = np.ones((3, 3))
    for i, row in enumerate(labels_main):
        for j, col in enumerate(labels_main):
            if i < j:
                labels = get_dynamic_scale_labels(row, col)
                mapping = get_dynamic_label_to_value(row, col)
                selected = st.select_slider(f"{row} vs {col}", options=labels, value="同じくらい重要")
                matrix_main[i][j] = mapping[selected]
                matrix_main[j][i] = 1 / mapping[selected]

    priorities_main, cr_main = ahp_calculation(matrix_main)
    st.dataframe(pd.DataFrame({"項目": labels_main, "優先度（%）": (priorities_main * 100).round(1)}))
    st.write(f"整合性比率 (CR): {cr_main:.3f}")

    if cr_main > 0.15:
        st.error("⚠ 判断に矛盾がある可能性があります（CR > 0.15）")
    elif cr_main > 0.10:
        st.warning("⚠ やや不安定です（0.10 < CR ≤ 0.15）")
    else:
        st.success("✅ 一貫した判断です（CR ≤ 0.10）")

    # --- 各カテゴリ ---
    for group_name, group_items in {
        "環境": ['気候変動', '資源循環・循環経済', '生物多様性', '自然資源'],
        "社会": ['人権・インクルージョン', '雇用・労働慣行', '多様性・公平性'],
        "ガバナンス": ['取締役会構成・少数株主保護', '統治とリスク管理']
    }.items():
        st.subheader(f"{group_name}の優先度測定")
        with st.expander("📝 補助説明（クリックで展開）"):
            if group_name == "環境":
                st.markdown("""
                **項目の説明**

                **気候変動**：企業が事業活動の中で温室効果ガス（CO₂など）をどれだけ減らそうとしているか。  

                **資源循環・循環経済**：企業が廃棄物の量を減らしたり、リサイクルや再利用を進めたりしているか。  

                **生物多様性**：企業が絶滅危惧種の保護や森林保全など、自然環境の維持にどれだけ力を入れているか。  

                **自然資源**：水や森林といった自然資源をどれだけ持続可能に使っているか。
                """)
            elif group_name == "社会":
                st.markdown("""
                **項目の説明**

                **人権・インクルージョン**：企業が人権侵害を防ぐために、事前にリスクを把握して対策を取っているか。  

                **雇用・労働慣行**：社員が有給休暇をしっかり取れているか、働きやすい職場環境が整っているか。  

                **多様性・公平性**：女性が管理職についている割合など、性別に関わらず平等に昇進の機会があるか。
                """)
            elif group_name == "ガバナンス":
                st.markdown("""
                **項目の説明**

                **取締役会構成・少数株主保護**：社外取締役が経営者をきちんとチェックする仕組みがあるか。  

                **統治とリスク管理**：企業が法令やルールを守っているか、内部統制がしっかりしているか。
                """)

        size = len(group_items)
        matrix = np.ones((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                labels = get_dynamic_scale_labels(group_items[i], group_items[j])
                mapping = get_dynamic_label_to_value(group_items[i], group_items[j])
                selected = st.select_slider(f"{group_items[i]} vs {group_items[j]}",
                                            options=labels, value="同じくらい重要")
                matrix[i][j] = mapping[selected]
                matrix[j][i] = 1 / mapping[selected]

        priorities, cr = ahp_calculation(matrix)
        st.dataframe(pd.DataFrame({"項目": group_items, "優先度（%）": (priorities * 100).round(1)}))
        all_priorities[group_name] = dict(zip(group_items, priorities))

# --- ④ 投資提案 ---
with tabs[3]:
    st.header("投資先提案")

    df = pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="Sheet1")
    all_labels = list(all_priorities["環境"].keys()) + list(all_priorities["社会"].keys()) + list(all_priorities["ガバナンス"].keys())

    weights = []
    for i, group in enumerate(all_priorities.keys()):
        for label, sub_weight in all_priorities[group].items():
            total_weight = priorities_main[i] * sub_weight
            weights.append(total_weight)

    dummy_csr = pd.DataFrame({
        "企業名": df["社名"],
        "気候変動": df["CO₂スコア"],
        "資源循環・循環経済": df["廃棄物スコア"],
        "生物多様性": df["生物多様性スコア"],
        "自然資源": df.get("自然資源スコア", 0),
        "人権・インクルージョン": df["人権DDスコア"],
        "雇用・労働慣行": df["有休スコア"],
        "多様性・公平性": df["女性比率スコア"],
        "取締役会構成・少数株主保護": df["取締役評価スコア"],
        "統治とリスク管理": df["内部通報スコア"]
    }).fillna(0)

    dummy_csr["スコア"] = dummy_csr[all_labels].dot(weights)
    result = dummy_csr.sort_values("スコア", ascending=False).head(3)

    st.subheader("上位3社（AHPによるスコア内訳）")
    st.caption("各項目は、あなたのESG重視度を反映した寄与スコア（点）です。")

    st.dataframe(result[["企業名", "スコア"]].style.format({"スコア": "{:.2f}"}))

    # --- 株価データとフロンティア ---
    df_price = pd.read_csv("CSR企業_株価データ_UTF-8（週次）.csv", index_col=0, parse_dates=True)
    selected_companies = result["企業名"].tolist()
    df_price = df_price[selected_companies].dropna()

    mu = expected_returns.mean_historical_return(df_price, frequency=52)
    S = risk_models.sample_cov(df_price, frequency=52)

    st.subheader("効率的フロンティア")
    ef_sharpe = EfficientFrontier(mu, S)
    ef_sharpe.max_sharpe()
    cleaned_weights = ef_sharpe.clean_weights()

    st.markdown("**最適ポートフォリオ（シャープレシオ最大）**")
    st.caption("""
    ※ AHPの結果から上位企業をピックアップし、株価データを用いて効率的フロンティアを作成しています。  
    リスクとリターンを最適化した結果、一部の企業は比率0（＝採用されない）になることがあります。
    """)

    portfolio_df = pd.DataFrame([{"企業名": k, "構成比（%）": round(v * 100, 2)} for k, v in cleaned_weights.items() if v > 0])
    st.dataframe(portfolio_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    plotting.plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=False)

    # ランダムポートフォリオ追加
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mu)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        results[0, i], results[1, i] = portfolio_return, portfolio_stddev

    ax.scatter(results[1, :], results[0, :], c="lightblue", alpha=0.3, s=10)
    risk_free_rate = 0.02
    ef_tangent = EfficientFrontier(mu, S)
    ef_tangent.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangent, std_tangent, _ = ef_tangent.portfolio_performance()

    x = np.linspace(0, std_tangent * 1.5, 100)
    y = risk_free_rate + (ret_tangent - risk_free_rate) / std_tangent * x
    ax.plot(x, y, "b-", label="資本市場線")
    ax.scatter(0, risk_free_rate, c="g", s=100, label="無リスク資産（点A）")
    ax.scatter(std_tangent, ret_tangent, c="r", s=200, marker="*", label="最大シャープレシオ点（点B）")

    ax.legend(
        [
            "効率的フロンティア（最適ポートフォリオの集合）",
            "ランダムポートフォリオ",
            "資本市場線",
            "無リスク資産（点A）",
            "最大シャープレシオ点（点B）"
        ],
        loc="upper left",
        bbox_to_anchor=(0.03, 0.97),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=9,
        handlelength=1.5
    )

    ax.set_title("効率的フロンティア（リスクとリターンの関係）", fontsize=13)
    ax.set_xlabel("リスク（標準偏差）")
    ax.set_ylabel("期待リターン")
    st.pyplot(fig)

