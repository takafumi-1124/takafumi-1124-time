import streamlit as st
import pandas as pd
import numpy as np
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import itertools
# 🔽 ここに追加してください
# === seaborn-deepエラー完全対策 ===
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mstyle
_original_style_use = plt.style.use

def safe_style_use(style_name):
    if style_name == "seaborn-deep":
        print("⚠ seaborn-deep style skipped (not available on Streamlit Cloud).")
        return
    return _original_style_use(style_name)

plt.style.use = safe_style_use
mpl.style.use = safe_style_use

# 🔼 ここまでを追加

from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
import japanize_matplotlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns  # ← seabornを先に読み込む

# --- seaborn-deep エラー対策 ---
try:
    plt.style.use("default")  # seaborn系スタイルを解除
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.set_style("whitegrid")  # seaborn側も安全なスタイルに固定
except OSError:
    pass

# --- PyPortfolioOpt 読み込み前に対策を適用 ---
import matplotlib.style as mstyle
mstyle.available = ["default", "classic"]  # seaborn-deepを無効化リストにする

import pypfopt.plotting as pplot  # ← これが問題のモジュール
pplot.plt = plt  # pltを上書き

# --- その他のインポート ---
from pypfopt import expected_returns, risk_models, EfficientFrontier
import japanize_matplotlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# ✅ seabornを強制的にインポート（PyPortfolioOptが暗黙的に呼ぶため）
import seaborn as sns
sns.set_style("whitegrid")



# スケール生成
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


st.set_page_config(page_title="ESG投資意思決定", layout="centered")
# st.markdown(
#     """
#     <meta http-equiv="Content-Language" content="ja">
#     <meta name="google" content="notranslate">
#     <style>
#     body, div, span, table {
#         class: notranslate;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.title("\U0001f331 ESG投資意思決定サイト")

tabs = st.tabs(["① ユーザー情報", "② Big Five 診断", "③ ESG優先度測定", "④ 投資提案"])
all_priorities = {}

# ① ユーザー情報
with tabs[0]:
    st.header("ユーザー情報入力")
    name = st.text_input("名前", key="name")
    age = st.number_input("年齢", 10, 100, 20, key="age")
    job = st.text_input("あなたの職業を入力してください", placeholder="例：大学生")

# ② Big Five
with tabs[1]:
    st.header("Big Five 診断")
    st.markdown("以下の各項目について、「非常に当てはまる（5）」から「まったく当てはまらない（1）」までの5段階で評価してください")

    bigfive_items = [
        ("無口な", "外向性", True), ("社交的", "外向性", False), ("話好き", "外向性", False),
        ("外向的", "外向性", False), ("陽気な", "外向性", False),
        ("いい加減な", "誠実性", True), ("ルーズな", "誠実性", True), ("成り行きまかせ", "誠実性", True),
        ("怠惰な", "誠実性", False), ("計画性のある", "誠実性", False), ("軽率な", "誠実性", False),
        ("几帳面", "誠実性", False),
        ("不安になりやすい", "情緒不安定性", False), ("心配性", "情緒不安定性", False),
        ("弱気になる", "情緒不安定性", False), ("緊張しやすい", "情緒不安定性", False), ("憂鬱な", "情緒不安定性", False),
        ("多才の", "開放性", False), ("進歩的", "開放性", False), ("独創的な", "開放性", True),
        ("頭の回転の速い", "開放性", False), ("興味の広い", "開放性", False), ("好奇心が強い", "開放性", False),
        ("短気", "調和性", True), ("怒りっぽい", "調和性", True), ("温和な", "調和性", False),
        ("寛大な", "調和性", False), ("自己中心的", "調和性", True), ("親切な", "調和性", False),
    ]

    scores = {}
    for item, trait, reverse in bigfive_items:
        val = st.slider(f"{item}", 1, 5, 3, key=f"bf_{item}")
        if reverse:
            val = 6 - val
        scores.setdefault(trait, []).append(val)

    trait_scores = {k: np.mean(v) for k, v in scores.items()}
    st.subheader("Big Five スコア")
    st.dataframe(pd.DataFrame(trait_scores.items(), columns=["性格特性", "スコア"]))

# ③ AHP
with tabs[2]:
    st.header("ESG優先度測定")
    st.markdown("""
    以下の項目では、2つの要素が並んで表示されます。  
    バーの位置を左右に動かして、どちらをどの程度優先するかを選んでください。  

    - 左に動かす → 左側の要素をより重視  
    - 右に動かす → 右側の要素をより重視  
    - 真ん中 → 「同じくらい重要」
    """)


    labels_main = ['環境', '社会', 'ガバナンス']
    matrix_main = np.ones((3, 3))

    for i, row in enumerate(labels_main):
        for j, col in enumerate(labels_main):
            if i < j:
                labels = get_dynamic_scale_labels(row, col)
                mapping = get_dynamic_label_to_value(row, col)
                selected = st.select_slider(
                    f"{row} vs {col}", options=labels,
                    key=f"main_{row}_{col}", value="同じくらい重要"
                )
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
                selected = st.select_slider(
                    f"{group_items[i]} vs {group_items[j]}",
                    options=labels,
                    key=f"{group_name}_{i}_{j}",
                    value="同じくらい重要"
                )
                matrix[i][j] = mapping[selected]
                matrix[j][i] = 1 / mapping[selected]

        priorities, cr = ahp_calculation(matrix)
        st.dataframe(pd.DataFrame({"項目": group_items, "優先度": priorities}))
        st.write(f"整合性比率 (CR): {cr:.3f}")
        all_priorities[group_name] = dict(zip(group_items, priorities))


    st.divider()  # 区切り線

    st.subheader("AHP結果のまとめ")

    # 大分類で最も重視されているものを特定
    top_category = labels_main[np.argmax(priorities_main)]

    # 小分類の中で最も重視されているものを特定
    if top_category in all_priorities:
        top_sub = max(all_priorities[top_category].items(), key=lambda x: x[1])[0]
        st.markdown(f"""
        あなたが最も重視しているのは **「{top_category}」** です。  
        その中でも特に **「{top_sub}」** を重視している傾向が見られます。
        """)
    else:
        st.info("各カテゴリーの入力が完了すると、結果が表示されます。")


# ④ 投資提案
@st.cache_data
def load_data():
    return pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="Sheet1")


import os
import json
import streamlit as st
import pandas as pd
from google.oauth2 import service_account
import gspread

# --- Excelファイル読み込み ---
@st.cache_data
def load_data():
    return pd.read_excel("スコア付きESGデータ - コピー.xlsx", sheet_name="Sheet1")

df = load_data()

# --- Google Sheets 認証 ---
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

try:
    # ✅ Streamlit Cloud または secrets.toml に認証情報がある場合
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scope
    )
    st.write("✅ 認証: Streamlit Secrets 経由で成功しました。")
# except Exception as e:
#     # ✅ ローカル開発用のフォールバック
#     # st.warning("⚠️ Streamlit Secretsが見つからないため、ローカルのcredentials.jsonを使用します。")
#     with open("credentials.json", "r") as f:
#         creds_json = json.load(f)
#     creds = service_account.Credentials.from_service_account_info(
#         creds_json,
#         scopes=scope
#     )

# --- gspreadクライアント作成 ---
gc = gspread.authorize(creds)

# --- スプレッドシート接続 ---
spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1WVOWgp5Q4TlQu_HhX3TJ5F0beSw1zkcZfcxiiVhn3Yk/edit?gid=0#gid=0")
worksheet = spreadsheet.sheet1  # 最初のシートを選択

# st.write("✅ Googleスプレッドシートに接続しました。")



with tabs[3]:
    st.header("投資先提案")
    st.subheader("投資先の企業")


    all_labels = (
        list(all_priorities["環境"].keys())
        + list(all_priorities["社会"].keys())
        + list(all_priorities["ガバナンス"].keys())
    )

    # AHPで決定した重み（大分類 × 小分類）
    weights = []
    for i, group in enumerate(all_priorities.keys()):
        for label, sub_weight in all_priorities[group].items():
            total_weight = priorities_main[i] * sub_weight
            weights.append(total_weight)

    # 企業データの準備
    dummy_csr = pd.DataFrame({
        "企業名": df["社名"],
        "気候変動": df["CO₂スコア"],
        "資源循環・循環経済": df["廃棄物スコア"],
        "生物多様性": df["生物多様性スコア"],
        "自然資源": df["自然資源スコア"] if "自然資源スコア" in df.columns else 0,
        "人権・インクルージョン": df["人権DDスコア"],
        "雇用・労働慣行": df["有休スコア"],
        "多様性・公平性": df["女性比率スコア"],
        "取締役会構成・少数株主保護": df["取締役評価スコア"],
        "統治とリスク管理": df["内部通報スコア"]
    })

    # 欠損値を0で補完
    dummy_csr = dummy_csr.fillna(0)

    # スコア計算（大分類×小分類の重みを反映）
    dummy_csr["スコア"] = dummy_csr[all_labels].dot(weights)

    # 上位10社表示
    result = dummy_csr.sort_values("スコア", ascending=False).head(3)
    st.dataframe(result[["企業名", "スコア"]])

    # ==========================
    # 効率的フロンティア（CSVの株価データ利用）
    # ==========================
    st.subheader("効率的フロンティア")

    try:
        df_price = pd.read_csv("CSR企業_株価データ（週次）.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("週次株価データファイルが見つかりません")
    else:
        try:
            # ★ CSVの列名（企業名）が AHP の結果と一致している前提
            selected_companies = result["企業名"].tolist()
            df_price = df_price[selected_companies].dropna()
        except KeyError as e:
            st.error(f"株価データに一致しない企業名があります: {e}")
        else:
            # リターンと分散共分散を計算（週次なので frequency=52）
            returns = df_price.pct_change().dropna()
            mu = expected_returns.mean_historical_return(df_price, frequency=52)
            S = risk_models.sample_cov(df_price, frequency=52)
            

            # 最適ポートフォリオ（シャープレシオ最大）
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            st.subheader("最適ポートフォリオ（シャープレシオ最大）")
            for stock, weight in cleaned_weights.items():
                if weight > 0:
                    st.write(f"{stock}: {weight:.2%}")

            # 効率的フロンティアを描画
            ef_plot = EfficientFrontier(mu, S)
            fig, ax = plt.subplots()
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)

            # ===== ランダムポートフォリオの追加 =====
            n_samples = 1000
            n_assets = len(mu)
            port_returns = []
            port_volatility = []

            # for _ in range(n_samples):
            #     weights = np.random.random(n_assets)
            #     weights /= np.sum(weights)
            #     ret = np.dot(weights, mu)
            #     vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
            #     port_returns.append(ret)
            #     port_volatility.append(vol)

            # for _ in range(n_samples):
            #     # まず全てゼロにする
            #     weights = np.zeros(n_assets)

            #     # ランダムに使う銘柄を選ぶ（1社だけでもOK、最大で全社）
            #     selected = np.random.choice(n_assets, size=np.random.randint(1, n_assets+1), replace=False)

            #     # 選んだ銘柄にだけ重みを割り当てる
            #     random_vals = np.random.random(len(selected))
            #     random_vals /= np.sum(random_vals)
            #     weights[selected] = random_vals

            #     # リターンとリスクを計算
            #     ret = np.dot(weights, mu)
            #     vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

            #     port_returns.append(ret)
            #     port_volatility.append(vol)

            # ax.scatter(port_volatility, port_returns, c="lightblue", alpha=0.3, s=10, label="ランダムポートフォリオ")

            # ===== ランダムポートフォリオの追加 =====
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))  # 0行=リターン, 1行=リスク, 2行=シャープレシオ

            for i in range(num_portfolios):
                weights = np.random.random(len(mu))       # 資産ごとに乱数を割り当て
                weights /= np.sum(weights)                # 合計を1に正規化

                # リターンとリスクを計算
                portfolio_return = np.dot(weights, mu)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

                # 無リスク利子率を仮定してシャープレシオを計算（例：2%）
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_stddev

                # 結果を格納
                results[0, i] = portfolio_return
                results[1, i] = portfolio_stddev
                results[2, i] = sharpe_ratio

            # 散布図をプロット
            ax.scatter(results[1, :], results[0, :], c="lightblue", alpha=0.3, s=10, label="ランダムポートフォリオ")
            # =========================================

            # =========================================

            # 無リスク金利を仮定（例：2%）
            risk_free_rate = 0.02
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            ret_tangent, std_tangent, _ = ef.portfolio_performance()

            # 資本市場線
            x = np.linspace(0, std_tangent * 1.5, 100)
            y = risk_free_rate + (ret_tangent - risk_free_rate) / std_tangent * x
            ax.plot(x, y, "b-", label="資本市場線")

            # 無リスク資産（点A）
            ax.scatter(0, risk_free_rate, marker="o", s=100, c="g", label="無リスク資産（点A）")

            # 最大シャープレシオ点（点B）
            ax.scatter(std_tangent, ret_tangent, marker="*", s=200, c="r", label="最大シャープレシオ点（点B）")

            # 凡例を日本語に
            handles, labels = ax.get_legend_handles_labels()
            label_map = {"Efficient frontier": "効率的フロンティア", "assets": "資産"}
            labels = [label_map.get(l, l) for l in labels]
            ax.legend(handles, labels, loc="best")

            # タイトルと軸ラベル
            ax.set_title("効率的フロンティアと資本市場線")
            ax.set_xlabel("リスク（ボラティリティ）")
            ax.set_ylabel("リターン")

            # ✅ 最後に1回だけ描画
            st.pyplot(fig)

            # =====================
            # Google Sheetsに保存（ボタンで実行）
            # =====================
            if st.button("保存"):
                # Big Five（平均値を使う）
                bigfive_values = [trait_scores.get(trait, 0) for trait in ["外向性", "誠実性", "情緒不安定性", "開放性", "調和性"]]

                # 大分類AHP（環境・社会・ガバナンスの重み）
                main_priorities_values = list(priorities_main)

                # 小分類AHP（all_prioritiesの値を順番に展開）
                sub_priorities_values = []
                for group in ["環境", "社会", "ガバナンス"]:
                    sub_priorities_values.extend(all_priorities[group].values())

                # 上位3社（企業名をカンマ区切りで連結）
                top3_companies = ", ".join(result["企業名"].tolist())

                # 最適ポートフォリオ（銘柄:比率 を文字列化）
                portfolio_str = ", ".join([f"{stock}:{weight:.2%}" for stock, weight in cleaned_weights.items() if weight > 0])

                # 1行にまとめる
                row_data = [
                    name,
                    age,
                    job,
                    *bigfive_values,
                    *main_priorities_values,
                    *sub_priorities_values,
                    top3_companies,
                    portfolio_str
                ]

                # シートに1行追加
                worksheet.append_row(row_data, value_input_option="USER_ENTERED")

                st.success("保存しました！")










