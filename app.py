# ============================================
# ESG投資意思決定アプリ（Streamlit Cloud対応版）
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- seaborn関連エラー対策 ---
mpl.rcParams.update(mpl.rcParamsDefault)
try:
    plt.style.use("default")
except OSError:
    pass

# ✅ ここでPyPortfolioOptを後からimport（順番が重要）
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
import japanize_matplotlib
import gspread
from google.oauth2 import service_account



# ============================================
# 関数定義
# ============================================

def get_dynamic_scale_labels(left: str, right: str):
    return [
        f"{left}が圧倒的に重要", f"{left}が非常に重要", f"{left}がかなり重要", f"{left}が少し重要",
        "同じくらい重要",
        f"{right}が少し重要", f"{right}がかなり重要", f"{right}が非常に重要", f"{right}が圧倒的に重要"
    ]

def get_dynamic_label_to_value(left: str, right: str):
    values = [9, 7, 5, 3, 1, 1/3, 1/5, 1/7, 1/9]
    labels = get_dynamic_scale_labels(left, right)
    return dict(zip(labels, values))

def ahp_calculation(pairwise_matrix):
    n = pairwise_matrix.shape[0]
    geo_means = np.prod(pairwise_matrix, axis=1) ** (1/n)
    priorities = geo_means / np.sum(geo_means)
    weighted_sum = np.dot(pairwise_matrix, priorities)
    lamda_max = np.sum(weighted_sum / priorities) / n
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
        5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45
    }
    CR = CI / RI_dict[n]
    return priorities, CR


# ============================================
# Streamlit ページ設定
# ============================================

st.set_page_config(page_title="ESG投資意思決定サイト", layout="centered")
st.title("🌱 ESG投資意思決定サイト")

tabs = st.tabs(["① ユーザー情報", "② Big Five 診断", "③ ESG優先度測定", "④ 投資提案"])
all_priorities = {}


# ============================================
# ① ユーザー情報入力
# ============================================

with tabs[0]:
    st.header("ユーザー情報入力")
    name = st.text_input("名前", key="name")
    age = st.number_input("年齢", 10, 100, 20, key="age")
    job = st.text_input("職業", placeholder="例：大学生")


# ============================================
# ② Big Five診断
# ============================================

with tabs[1]:
    st.header("Big Five 診断")
    st.markdown("各項目について、1〜5で回答してください。")

    bigfive_items = [
        ("無口な", "外向性", True), ("社交的", "外向性", False), ("話好き", "外向性", False),
        ("外向的", "外向性", False), ("陽気な", "外向性", False),
        ("いい加減な", "誠実性", True), ("几帳面", "誠実性", False),
        ("不安になりやすい", "情緒不安定性", False), ("心配性", "情緒不安定性", False),
        ("多才の", "開放性", False), ("独創的な", "開放性", True),
        ("短気", "調和性", True), ("温和な", "調和性", False),
    ]

    scores = {}
    for item, trait, reverse in bigfive_items:
        val = st.slider(f"{item}", 1, 5, 3, key=f"bf_{item}")
        if reverse:
            val = 6 - val
        scores.setdefault(trait, []).append(val)

    trait_scores = {k: np.mean(v) for k, v in scores.items()}
    st.dataframe(pd.DataFrame(trait_scores.items(), columns=["性格特性", "スコア"]))


# ============================================
# ③ AHP：ESG優先度測定
# ============================================

with tabs[2]:
    st.header("ESG優先度測定")

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
    st.dataframe(pd.DataFrame({"項目": labels_main, "優先度": priorities_main}))
    st.write(f"整合性比率 (CR): {cr_main:.3f}")

    for group_name, group_items in {
        "環境": ['気候変動', '資源循環・循環経済', '生物多様性', '自然資源'],
        "社会": ['人権・インクルージョン', '雇用・労働慣行', '多様性・公平性'],
        "ガバナンス": ['取締役会構成・少数株主保護', '統治とリスク管理']
    }.items():
        st.subheader(f"{group_name}の優先度測定")
        size = len(group_items)
        matrix = np.ones((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                labels = get_dynamic_scale_labels(group_items[i], group_items[j])
                mapping = get_dynamic_label_to_value(group_items[i], group_items[j])
                selected = st.select_slider(f"{group_items[i]} vs {group_items[j]}", options=labels, value="同じくらい重要")
                matrix[i][j] = mapping[selected]
                matrix[j][i] = 1 / mapping[selected]

        priorities, cr = ahp_calculation(matrix)
        st.dataframe(pd.DataFrame({"項目": group_items, "優先度": priorities}))
        st.write(f"整合性比率 (CR): {cr:.3f}")
        all_priorities[group_name] = dict(zip(group_items, priorities))

    st.divider()
    st.subheader("AHP結果まとめ")
    top_category = labels_main[np.argmax(priorities_main)]
    if top_category in all_priorities:
        top_sub = max(all_priorities[top_category].items(), key=lambda x: x[1])[0]
        st.markdown(f"あなたが最も重視しているのは **{top_category}** で、その中でも **{top_sub}** です。")


# ============================================
# ④ 投資提案 & スプレッドシート保存
# ============================================

@st.cache_data
def load_data():
    return pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="Sheet1")

df = load_data()

# --- ✅ Google Sheets認証（Secrets使用） ---
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope
)
gc = gspread.authorize(creds)
st.write("✅ Google認証に成功しました！")

spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1WVOWgp5Q4TlQu_HhX3TJ5F0beSw1zkcZfcxiiVhn3Yk/edit?gid=0#gid=0")
worksheet = spreadsheet.sheet1


# --- 投資提案結果 ---
with tabs[3]:
    st.header("投資先提案")
    st.subheader("投資先の企業")

    dummy_csr = df.copy()
    dummy_csr["スコア"] = np.random.rand(len(df))
    result = dummy_csr.sort_values("スコア", ascending=False).head(3)
    st.dataframe(result[["社名", "スコア"]])

    if st.button("結果をスプレッドシートに保存"):
        row_data = [
            name,
            age,
            job,
            trait_scores.get("外向性", 0),
            trait_scores.get("誠実性", 0),
            trait_scores.get("情緒不安定性", 0),
            trait_scores.get("開放性", 0),
            trait_scores.get("調和性", 0),
            ", ".join(result["社名"].tolist())
        ]
        worksheet.append_row(row_data, value_input_option="USER_ENTERED")
        st.success("✅ 結果をスプレッドシートに保存しました！")

