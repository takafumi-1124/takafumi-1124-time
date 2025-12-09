import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    plt.style.use("seaborn-deep")
except OSError:
    print("⚠ seaborn-deep style not available on Streamlit Cloud. Using default style instead.")
    plt.style.use("default")
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["axes.unicode_minus"] = False

# === 日本語フォントを明示的に設定（Linux環境用） ===
mpl.rcParams['font.family'] = ['IPAexGothic', 'IPAPGothic', 'TakaoPGothic', 'Noto Sans CJK JP']


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

# --- スライダーの間隔をCSSで調整 ---
st.markdown(
    """
    <style>
    .stSlider {
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["① ユーザー情報", "② ESGについて","③ ESG優先度測定", "④ 投資提案"])
all_priorities = {}

# --- ① ユーザー情報 ---
with tabs[0]:
    st.header("ユーザー情報入力")
    name = st.text_input("名前", key="name")
    age = st.number_input("年齢", 10, 100, 20, key="age")
    job = st.text_input("あなたの職業を入力してください", placeholder="例：大学生")

# # --- ② Big Five ---
with tabs[1]:
    # --- ESG投資とは？説明セクション ---
    st.markdown("""
    ### 📘 ESG投資とは？
    ESG投資とは、会社を「**環境（E）・社会（S）・ガバナンス（G）**」という3つの視点から見て、  
    **安心して長く応援できる企業**を選ぶ投資のことです。
    
    ---
    
    #### 🌿 3つの視点とは？
    - **環境（E）**：気候変動、生物多様性 など  
    - **社会（S）**：人権、公平性 など  
    - **ガバナンス（G）**：リスク管理、株主保護 など  
    
    ---
    
    #### 💡 「社会のため」だけではないESG
    ESGは「良いことをしている会社を応援する」というだけではありません。  
    実は、**会社の“リスクを減らして長く生き残る力”を見極める考え方**でもあります。  
    
    たとえば、環境汚染や労働問題などのトラブルを防ぐ会社は、  
    将来の経営リスクが少なく、**安定した成長が期待できる**のです。
    
    ---
    
    ### 🌍 ESG投資は本当に安定している？
    日本の年金を運用している **GPIF（年金積立金管理運用独立行政法人）** では、  
    2017年からESGの考え方を取り入れた投資を続けています。  
    
    その結果、**ESGを重視した企業のグループ（ESG指数）は、  
    同じ期間の市場平均（TOPIX）よりも安定した成果**を出しました。  
    （2017〜2023年のリスク調整後リターン：ESG指数 0.39／TOPIX 0.37）
    
    ただし、GPIFは  
    > 「短い期間での結果に一喜一憂せず、長い目で見た検証が必要」  
    と述べており、  
    **短期的に儲かる投資ではなく、長期的に安定を目指す方法**です。
    
    ---
    
    #### 💬 まとめ
    - ESG投資は「**安心して長く成長を見守れる企業を選ぶ考え方**」  
    - 世界中の投資家が、「**社会と企業の両方が続く仕組み**」として注目しています。  
    
    
    ---
    
    """)



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
        企業が環境への負荷を減らす取り組みをしているか。  

        **社会（Social）**  
        企業が人や社会に対して配慮しているか。  

        **ガバナンス（Governance）**  
        企業の経営の仕組みや透明性が適切に整えられているか。
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

    st.dataframe(
        pd.DataFrame({
            "項目": labels_main,
            "優先度（%）": (priorities_main * 100).round(1)
        }),
        use_container_width=True,
        hide_index=True
    )

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

        # === 補助説明 ===
        with st.expander("📝 補助説明（クリックで展開）"):
            if group_name == "環境":
                st.markdown("""
                **気候変動**：温室効果ガス（CO₂など）の削減にどれだけ取り組んでいるか。  
                **資源循環・循環経済**：リサイクルや再利用など、廃棄物削減への努力。  
                **生物多様性**：森林保全や絶滅危惧種保護などへの配慮。  
                **自然資源**：水や森林などを持続的に使っているか。
                """)
            elif group_name == "社会":
                st.markdown("""
                **人権・インクルージョン**：人権侵害防止・差別のない職場作り。  
                **雇用・労働慣行**：働きやすい職場環境や有給取得のしやすさ。  
                **多様性・公平性**：性別・国籍に関係なく平等な昇進機会。
                """)
            elif group_name == "ガバナンス":
                st.markdown("""
                **取締役会構成・少数株主保護**：経営のチェック体制が整っているか。  
                **統治とリスク管理**：法令遵守・リスク管理・内部統制ができているか。
                """)

        # === AHP比較 ===
        size = len(group_items)
        matrix = np.ones((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                labels = get_dynamic_scale_labels(group_items[i], group_items[j])
                mapping = get_dynamic_label_to_value(group_items[i], group_items[j])
                selected = st.select_slider(
                    f"{group_items[i]} vs {group_items[j]}",
                    options=labels,
                    value="同じくらい重要"
                )
                matrix[i][j] = mapping[selected]
                matrix[j][i] = 1 / mapping[selected]

        # === 結果計算 ===
        priorities, cr = ahp_calculation(matrix)

        # === 表示 ===
        st.dataframe(
            pd.DataFrame({
                "項目": group_items,
                "優先度（%）": (priorities * 100).round(1)
            }),
            use_container_width=True,
            hide_index=True
        )

        # === 整合性比率の出力 ===
        st.write(f"整合性比率 (CR): {cr:.3f}")
        if cr > 0.15:
            st.error("⚠ 判断に矛盾がある可能性があります（CR > 0.15）")
        elif cr > 0.10:
            st.warning("⚠ やや不安定です（0.10 < CR ≤ 0.15）")
        else:
            st.success("✅ 一貫した判断です（CR ≤ 0.10）")

        # === 結果を保存 ===
        all_priorities[group_name] = dict(zip(group_items, priorities))

    st.divider()
    st.subheader("ESG優先度の結果まとめ")
    top_category = labels_main[np.argmax(priorities_main)]
    if top_category in all_priorities:
        top_sub = max(all_priorities[top_category].items(), key=lambda x: x[1])[0]
        st.markdown(f"""
        あなたが最も重視しているのは **「{top_category}」** です。  
        その中でも特に **「{top_sub}」** を重視している傾向が見られます。
        """)



# --- ④ 投資提案 ---
with tabs[3]:

    # --- CSSは必ず最初 ---
    st.markdown("""
<style>
.esg-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
}
.esg-table th {
    background: #f5f7fa;
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid #ccc;
}
.esg-table td {
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid #eee;
}
.esg-table th:first-child, .esg-table td:first-child {
    min-width: 300px;
    max-width: 450px;
    white-space: nowrap;
    text-align: left;
}
a {
    color: #1a73e8;
    font-weight: bold;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

    # --- データ処理 ---
    df = pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="Sheet1")
    df_url = pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="URL")
    df = pd.merge(df, df_url, on="社名", how="left")

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
        "統治とリスク管理": df["内部通報スコア"],
        "URL": df["URL"]
    }).fillna(0)

    dummy_csr["環境スコア"] = dummy_csr[["気候変動","資源循環・循環経済","生物多様性","自然資源"]].mean(axis=1) * priorities_main[0]
    dummy_csr["社会スコア"] = dummy_csr[["人権・インクルージョン","雇用・労働慣行","多様性・公平性"]].mean(axis=1) * priorities_main[1]
    dummy_csr["ガバナンススコア"] = dummy_csr[["取締役会構成・少数株主保護","統治とリスク管理"]].mean(axis=1) * priorities_main[2]
    dummy_csr["合計スコア"] = dummy_csr["環境スコア"] + dummy_csr["社会スコア"] + dummy_csr["ガバナンススコア"]

    result = dummy_csr.sort_values("合計スコア", ascending=False).head(3)

    result["企業名リンク"] = result.apply(
        lambda x: f'<a href="{x["URL"]}" target="_blank">{x["企業名"]}</a>',
        axis=1
    )

    df_show = result[["企業名リンク","環境スコア","社会スコア","ガバナンススコア","合計スコア"]].round(2)

    html_table = df_show.to_html(index=False, escape=False, classes="esg-table")

    st.markdown(html_table, unsafe_allow_html=True)


















    # --- 株価データとフロンティア ---
    # --- 株価データ読み込み ---
    df_price = pd.read_csv("CSR企業_株価データ_UTF-8（週次）.csv", index_col=0, parse_dates=True)

    # --- 上位企業の抽出 ---
    selected_companies = result["企業名"].tolist()
    df_price = df_price[selected_companies].dropna()

    # --- リターンと分散共分散行列の計算 ---
    mu = expected_returns.mean_historical_return(df_price, frequency=52)
    S = risk_models.sample_cov(df_price, frequency=52)

    # --- 効率的フロンティア計算 ---
    st.subheader("効率的フロンティア")

    ef_sharpe = EfficientFrontier(mu, S)
    ef_sharpe.max_sharpe()
    cleaned_weights = ef_sharpe.clean_weights()

    # --- 投資比率をDataFrameに整形 ---
    portfolio_df = (
        pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["投資配分（%）"])
        .reset_index()
        .rename(columns={"index": "企業名"})
    )
    portfolio_df["投資配分（%）"] = (portfolio_df["投資配分（%）"] * 100).round(2)
    portfolio_df = portfolio_df[portfolio_df["投資配分（%）"] > 0]  # 比率0は除外

    # --- 表示 ---
    st.markdown("**最適ポートフォリオ（シャープレシオ最大）**")
    st.caption("""
    ※ ESG優先度測定の結果から上位企業をピックアップし、株価データを用いて効率的フロンティアを作成しています。  
    リスクとリターンを最適化した結果、一部の企業は比率0（＝採用されない）になることがあります。
    """)

    st.dataframe(
        portfolio_df.style.format({"投資配分（%）": "{:.2f}"})
        .set_properties(subset=["企業名"], **{"font-weight": "bold", "text-align": "left", "width": "200px"})
        .set_table_styles([
            {"selector": "thead th", "props": [("font-size", "14px"), ("text-align", "center")]},
            {"selector": "td", "props": [("font-size", "13px")]}
        ]),
        use_container_width=True,
        hide_index=True
    )

    # --- 効率的フロンティア図の描画 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    plotting.plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=False)

    # --- ランダムポートフォリオ追加 ---
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mu)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        results[0, i], results[1, i] = portfolio_return, portfolio_stddev

    ax.scatter(results[1, :], results[0, :], c="lightblue", alpha=0.3, s=10)

    # --- 資本市場線を追加 ---
    risk_free_rate = 0.02
    ef_tangent = EfficientFrontier(mu, S)
    ef_tangent.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangent, std_tangent, _ = ef_tangent.portfolio_performance()

    x = np.linspace(0, std_tangent * 1.5, 100)
    y = risk_free_rate + (ret_tangent - risk_free_rate) / std_tangent * x
    ax.plot(x, y, "b-", label="資本市場線")
    ax.scatter(0, risk_free_rate, c="g", s=100, label="無リスク資産（点A）")
    ax.scatter(std_tangent, ret_tangent, c="r", s=200, marker="*", label="最大シャープレシオ点（点B）")

    # --- 凡例など整形 ---
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



























