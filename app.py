import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting

# ============================================
# 🔧 日本語フォント設定（Streamlit Cloud 用）
# ============================================
mpl.rcParams['font.family'] = ['IPAexGothic', 'IPAPGothic', 'TakaoPGothic', 'Noto Sans CJK JP']

# ============================================
# 🔧 ページ設定
# ============================================
st.set_page_config(page_title="ESG投資意思決定", layout="centered")

# ============================================
# 🔧 セッション状態（ステップ番号）
# ============================================
if "step" not in st.session_state:
    st.session_state.step = 0   # 0:ユーザー情報 → 3:投資提案

# ============================================
# 🔧 ステップのラベル
# ============================================
steps = ["① ユーザー情報", "② ESGについて", "③ ESG優先度測定", "④ 投資提案"]

# ============================================
# 🔧 インジケータHTML（●○○○）
# ============================================
def render_indicator():
    html = "<div style='display:flex;justify-content:center;gap:8px;margin:10px 0;'>"
    for i in range(4):
        if i == st.session_state.step:
            html += "<div style='width:14px;height:14px;border-radius:50%;background:#ff6b6b;'></div>"
        else:
            html += "<div style='width:14px;height:14px;border-radius:50%;background:#ddd;'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ============================================
# 🔧 ステップ遷移ボタン
# ============================================
def nav_buttons(back=True, next=True):
    cols = st.columns([1,1])
    with cols[0]:
        if back and st.button("◀ 戻る"):
            st.session_state.step -= 1
            st.rerun()
    with cols[1]:
        if next and st.button("次へ ▶"):
            st.session_state.step += 1
            st.rerun()

# ============================================
# 🔧 Step 1：ユーザー情報入力
# ============================================
def step1():
    st.header("① ユーザー情報入力")

    st.write("以下の情報をご入力ください：")

    st.text_input("名前", key="username")
    st.number_input("年齢", min_value=10, max_value=100, key="age", value=20)
    st.text_input("職業（例：大学生）", key="job")

    st.info("入力した情報は分析結果の表示に使用されます。")

    nav_buttons(back=False, next=True)

# ============================================
# 🔧 メイン処理：ステップ切替
# ============================================
render_indicator()

if st.session_state.step == 0:
    step1()



# ============================================
# 🔧 Step 2：ESGとは？
# ============================================
def step2():
    st.header("② ESGとは？")

    st.markdown("""
    ### 🌱 ESG投資とは？
    企業を **環境（E）・社会（S）・ガバナンス（G）** の3つの視点で評価し、  
    **長期的に安心して応援できる企業を選ぶ投資方法** です。

    ---
    ### 🌿 3つの視点
    #### **🔹 環境（Environment）**
    - 気候変動への取り組み  
    - 温室効果ガス削減  
    - 廃棄物削減、自然資源の保全  

    #### **🔹 社会（Social）**
    - 人権や労働環境
    - 雇用制度・ダイバーシティ  
    - 従業員への健康・安全管理  

    #### **🔹 ガバナンス（Governance）**
    - 経営の透明性
    - 役員の監督体制  
    - リスク管理や内部統制  

    ---
    ### 💡 ESG＝「社会に良いこと」＋「企業リスクの低減」
    ESGは「良い企業を応援する」というだけではありません。

    実は、  
    **トラブルを避けて長期的に安定して成長できる企業を選ぶ基準**  
    として世界中の投資家が使用しています。

    ---
    ### 📈 GPIF（日本の年金基金）も採用
    日本最大の投資家である **GPIF（年金積立金管理運用独立行政法人）** は  
    2017年からESG投資を導入しています。

    2017〜2023年では以下の結果が出ています：

    - ESG指数のリスク調整後リターン：**0.39**
    - TOPIX（通常の株価指数）：**0.37**

    👉 短期で必ず儲かるわけではないが、  
    **長期的な安定が期待できる投資方法** と言われています。

    ---
    """)

    nav_buttons(back=True, next=True)

# ---------------------------
# AHP（幾何平均法）
# ---------------------------
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
    # AHPの1〜9スケール（簡略版）
    values = [7,5,3,1,1/3,1/5,1/7]
    labels = get_dynamic_scale_labels(left, right)
    return dict(zip(labels, values))

def ahp_calculation(pairwise_matrix):
    n = pairwise_matrix.shape[0]
    geo_means = np.prod(pairwise_matrix, axis=1) ** (1/n)
    priorities = geo_means / np.sum(geo_means)
    weighted_sum = np.dot(pairwise_matrix, priorities)
    lamda_max = np.sum(weighted_sum / priorities) / n
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    CR = CI / RI_dict[n]
    return priorities, CR


# ============================================
# 🔧 Step 3：ESG優先度測定（AHP）
# ============================================
def step3():

    st.header("③ ESG優先度測定（AHP）")

    st.markdown("""
    2つの項目を比較して、どちらをどの程度重視するかを選んでください。
    あなたの価値観に合わせて **ESG優先度（重み）** を算出します。
    """)

    # ---------------------------
    # メイン（環境・社会・ガバナンス）
    # ---------------------------
    st.subheader("■ 環境・社会・ガバナンスの比較")

    main_labels = ["環境", "社会", "ガバナンス"]
    mat_main = np.ones((3,3))

    for i in range(3):
        for j in range(i+1, 3):
            labels = get_dynamic_scale_labels(main_labels[i], main_labels[j])
            mapping = get_dynamic_label_to_value(main_labels[i], main_labels[j])

            selected = st.select_slider(
                f"{main_labels[i]} vs {main_labels[j]}",
                options=labels,
                value="同じくらい重要"
            )
            mat_main[i][j] = mapping[selected]
            mat_main[j][i] = 1 / mapping[selected]

    priorities_main, cr_main = ahp_calculation(mat_main)

    df_main = pd.DataFrame({
        "項目": main_labels,
        "優先度（%）": (priorities_main * 100).round(1)
    })

    st.dataframe(df_main, hide_index=True, use_container_width=True)
    st.write(f"整合性比率（CR）: **{cr_main:.3f}**")

    if cr_main > 0.15:
        st.error("⚠ 一貫性が低く、判断に矛盾がある可能性があります。")
    elif cr_main > 0.10:
        st.warning("⚠ やや不安定（0.10〜0.15）")
    else:
        st.success("✅ 一貫した判断です！")

    # ---------------------------
    # 個別カテゴリの比較
    # ---------------------------
    category_items = {
        "環境": ['気候変動', '資源循環・循環経済', '生物多様性', '自然資源'],
        "社会": ['人権・インクルージョン', '雇用・労働慣行', '多様性・公平性'],
        "ガバナンス": ['取締役会構成・少数株主保護', '統治とリスク管理']
    }

    # 保存先
    st.session_state.category_priorities = {}

    st.markdown("---")
    st.subheader("■ 各カテゴリ内の重要度比較")

    for cat, items in category_items.items():

        st.markdown(f"### 🔹 {cat}")

        n = len(items)
        matrix = np.ones((n,n))

        for i in range(n):
            for j in range(i+1, n):
                labels = get_dynamic_scale_labels(items[i], items[j])
                mapping = get_dynamic_label_to_value(items[i], items[j])

                selected = st.select_slider(
                    f"{items[i]} vs {items[j]}",
                    options=labels,
                    value="同じくらい重要"
                )

                matrix[i][j] = mapping[selected]
                matrix[j][i] = 1 / mapping[selected]

        pri, cr = ahp_calculation(matrix)

        df_cat = pd.DataFrame({
            "項目": items,
            "優先度（%）": (pri * 100).round(1)
        })

        st.dataframe(df_cat, hide_index=True, use_container_width=True)
        st.write(f"CR：**{cr:.3f}**")

        # 保存
        st.session_state.category_priorities[cat] = pri

        if cr > 0.15:
            st.error("⚠ 矛盾が大きい可能性があります。やり直すと改善します。")
        elif cr > 0.10:
            st.warning("⚠ やや不安定です。")
        else:
            st.success("✅ OK！")

    # ---------------------------
    # まとめ表示
    # ---------------------------
    st.markdown("---")
    st.subheader("■ あなたのESG優先度まとめ")

    top_cat = main_labels[np.argmax(priorities_main)]

    st.success(f"あなたが最も重視するのは **「{top_cat}」** です！")

    if top_cat in st.session_state.category_priorities:
        pri_cat = st.session_state.category_priorities[top_cat]
        items = category_items[top_cat]
        top_item = items[np.argmax(pri_cat)]

        st.info(f"その中でも特に **「{top_item}」** を重視しています。")

    st.markdown("---")

    nav_buttons(back=True, next=True)




# ============================================
# 🔥 Step 4：投資提案（ESG × 株価 × フロンティア）
# ============================================
def step4():

    st.header("④ 投資提案")
    st.markdown("""
    ここでは、あなたの **ESG優先度（AHP）** をもとに企業をスコア化し、
    さらに **株価データ（週次）** を使って効率的フロンティアから
    **最適ポートフォリオ** を算出します。
    """)

    # -------------------------------------------
    # 1️⃣ データ読み込み
    # -------------------------------------------
    df = pd.read_excel("スコア付きESGデータ.xlsx", sheet_name="Sheet1")
    df_url = pd.read_excel("スコア付きESGデータ.xlsx", sheet_name="URL")

    # URL を企業名に結合
    df = pd.merge(df, df_url, on="社名", how="left")

    # -------------------------------------------
    # 2️⃣ ESG寄与スコアを計算する
    # -------------------------------------------

    # AHP の結果を取得
    priorities_main = st.session_state.priorities_main   # [環境, 社会, ガバナンス]
    cat_pri = st.session_state.category_priorities       # 個別項目の重み

    # ダミーデータフレーム（ESG項目）
    dummy = pd.DataFrame({
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

    # 重み（AHP 結果）
    w_E = priorities_main[0]
    w_S = priorities_main[1]
    w_G = priorities_main[2]

    # カテゴリ内の重みを取得
    w_e = cat_pri["環境"]
    w_s = cat_pri["社会"]
    w_g = cat_pri["ガバナンス"]

    # スコア計算
    dummy["環境スコア"] = (
        dummy[["気候変動", "資源循環・循環経済", "生物多様性", "自然資源"]]
        .dot(w_e)
    ) * w_E

    dummy["社会スコア"] = (
        dummy[["人権・インクルージョン", "雇用・労働慣行", "多様性・公平性"]]
        .dot(w_s)
    ) * w_S

    dummy["ガバナンススコア"] = (
        dummy[["取締役会構成・少数株主保護", "統治とリスク管理"]]
        .dot(w_g)
    ) * w_G

    dummy["合計スコア"] = (
        dummy["環境スコア"] +
        dummy["社会スコア"] +
        dummy["ガバナンススコア"]
    )

    # -------------------------------------------
    # 3️⃣ 上位3社を表示（URLリンク付き）
    # -------------------------------------------
    st.subheader("上位3社（ESG × あなたの価値観）")

    st.caption("""
    ※ ESG項目スコアを AHP の重みに合わせて集計した結果です。
    ※ 企業名をクリックすると会社ページに飛びます。
    """)

    top3 = dummy.sort_values("合計スコア", ascending=False).head(3)

    # HTMLリンク化
    top3["企業リンク"] = top3.apply(
        lambda x: f'<a href="{x["URL"]}" target="_blank">{x["企業名"]}</a>'
        if pd.notna(x["URL"]) else x["企業名"],
        axis=1
    )

    show_df = top3[[
        "企業リンク", "環境スコア", "社会スコア", "ガバナンススコア", "合計スコア"
    ]].round(2)

    st.markdown(
        show_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    # -------------------------------------------
    # 4️⃣ 株価データの読み込み
    # -------------------------------------------
    st.subheader("効率的フロンティアによる最適ポートフォリオ")

    df_price = pd.read_csv("CSR企業_株価データ_UTF-8（週次）.csv", index_col=0, parse_dates=True)

    # Top3 の企業のみ
    selected = top3["企業名"].tolist()
    df_price = df_price[selected].dropna()

    # -------------------------------------------
    # 5️⃣ 平均リターン & 分散共分散
    # -------------------------------------------
    mu = expected_returns.mean_historical_return(df_price, frequency=52)
    Sigma = risk_models.sample_cov(df_price, frequency=52)

    # -------------------------------------------
    # 6️⃣ 最大シャープレシオポートフォリオ
    # -------------------------------------------
    ef = EfficientFrontier(mu, Sigma)
    ef.max_sharpe()
    weights = ef.clean_weights()

    pf_df = pd.DataFrame.from_dict(weights, orient="index", columns=["比率"])
    pf_df = pf_df.reset_index().rename(columns={"index": "企業名"})
    pf_df["比率"] = (pf_df["比率"] * 100).round(2)
    pf_df = pf_df[pf_df["比率"] > 0]

    st.dataframe(pf_df, use_container_width=True, hide_index=True)

    # -------------------------------------------
    # 7️⃣ 効率的フロンティア描画
    # -------------------------------------------
    st.subheader("効率的フロンティア（リスク×リターン）")

    fig, ax = plt.subplots(figsize=(7, 5))
    plotting.plot_efficient_frontier(EfficientFrontier(mu, Sigma), ax=ax, show_assets=False)

    # ランダムポートフォリオ追加
    num = 2000
    rand = np.zeros((2, num))
    for i in range(num):
        w = np.random.random(len(mu))
        w /= w.sum()
        rand[0, i] = np.dot(w, mu)
        rand[1, i] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))

    ax.scatter(rand[1], rand[0], c="lightblue", alpha=0.3, s=10)

    # 資本市場線
    rf = 0.02
    ef_tan = EfficientFrontier(mu, Sigma)
    ef_tan.max_sharpe(risk_free_rate=rf)
    ret_tan, std_tan, _ = ef_tan.portfolio_performance()

    x = np.linspace(0, std_tan * 1.3, 200)
    y = rf + (ret_tan - rf) / std_tan * x
    ax.plot(x, y, "r-", label="資本市場線")

    ax.scatter(std_tan, ret_tan, c="red", s=120, marker="*", label="最大シャープ点")

    ax.set_xlabel("リスク（標準偏差）")
    ax.set_ylabel("期待リターン")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------------------
    # 次へ・戻るボタン
    # -------------------------------------------
    nav_buttons(back=True, next=False)
