import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import matplotlib.style as mstyle
import seaborn as sns # <-- sns.set_styleを使用するため必須
import io # <-- モックデータ作成用

# Google Sheets関連のimportは、今回は使用しないためそのまま残しますが、
# Streamlit Secretsで認証情報を管理する際に使用してください。
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
import json
import os
import japanize_matplotlib # 日本語表示のため

# =======================================================
# seaborn-deepエラー完全対策（起動時の競合を避ける）
# =======================================================
plt.style.use("default")
mpl.rcParams.update(mpl.rcParamsDefault)

_original_style_use = plt.style.use # 組み込みの plt.style.use を一度だけ保存
def safe_style_use(style_name):
    # 'seaborn-deep'はStreamlit Cloudでしばしば問題を起こすためスキップ
    if style_name == "seaborn-deep":
        return
    return _original_style_use(style_name)

# Matplotlibのスタイル設定関数をパッチする
plt.style.use = safe_style_use
mpl.style.use = safe_style_use

# pypfoptが内部で存在しないスタイルを呼び出してもクラッシュしないように、利用可能なスタイルを上書き
mstyle.available = ["default", "classic", "whitegrid"] 

# seabornの設定
try:
    sns.set_style("whitegrid")
except Exception:
    # seabornがない環境でも動くようにエラーを無視
    pass


# 🔤 日本語フォント設定
mpl.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'IPAexGothic', 'Hiragino Sans']
mpl.rcParams['axes.unicode_minus'] = False  # マイナス符号の文字化け防止

# ==============================
# データファイルがない場合のモックデータ作成関数
# ==============================

@st.cache_data
def load_mock_data():
    """モックデータを生成し、DataFrameとして返却する"""
    
    # 1. スコア付きESGデータ のモック
    csr_data = {
        "社名": ["企業A", "企業B", "企業C", "企業D", "企業E"],
        "CO₂スコア": [80, 50, 90, 70, 60],
        "廃棄物スコア": [75, 65, 85, 55, 70],
        "生物多様性スコア": [60, 80, 50, 90, 70],
        "人権DDスコア": [90, 70, 80, 60, 75],
        "有休スコア": [85, 95, 75, 80, 65],
        "女性比率スコア": [70, 80, 90, 60, 50],
        "取締役評価スコア": [80, 70, 60, 90, 50],
        "内部通報スコア": [75, 65, 85, 70, 90],
    }
    df_csr = pd.DataFrame(csr_data)
    
    # 2. CSR企業_株価データ（週次）.csv のモック
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='W')
    price_data = {}
    for company in df_csr["社名"]:
        # 株価データのランダム生成（乗数系列）
        base_price = np.random.randint(100, 500)
        returns = np.random.normal(loc=0.001, scale=0.01, size=100)
        prices = base_price * (1 + returns).cumprod()
        price_data[company] = prices
        
    df_price = pd.DataFrame(price_data, index=dates)
    
    return df_csr, df_price

# アプリの開始時にデータを読み込む
df_csr, df_price = load_mock_data()


# ==============================
# 共通関数 (AHPスケールと計算)
# ==============================

def get_dynamic_scale_labels(left: str, right: str):
    """AHPの7段階スケールラベルを生成"""
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
    """AHPのラベルと対応する数値（逆数含む）をマッピング"""
    values = [7, 5, 3, 1, 1/3, 1/5, 1/7]
    labels = get_dynamic_scale_labels(left, right)
    return dict(zip(labels, values))

def ahp_calculation(pairwise_matrix):
    """AHPの重みと整合性比率（CR）を計算"""
    n = pairwise_matrix.shape[0]
    geo_means = np.prod(pairwise_matrix, axis=1) ** (1/n)
    priorities = geo_means / np.sum(geo_means)
    weighted_sum = np.dot(pairwise_matrix, priorities)
    lamda_max = np.sum(weighted_sum / priorities) / n
    CI = (lamda_max - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    
    if n not in RI_dict or RI_dict[n] == 0:
        CR = 0.0
    else:
        CR = CI / RI_dict[n]
        
    return priorities, CR

# ==============================
# Streamlit本体
# ==============================
st.set_page_config(page_title="ESG投資意思決定", layout="centered")
st.title("🌱 ESG投資意思決定サイト")

tabs = st.tabs(["① ユーザー情報", "② Big Five 診断", "③ ESG優先度測定", "④ 投資提案"])
all_priorities = {}

# --- ① ユーザー情報 ---
with tabs[0]:
    st.header("ユーザー情報入力")
    name = st.text_input("名前", key="name")
    age = st.number_input("年齢", 10, 100, 20, key="age")
    job = st.text_input("あなたの職業を入力してください", placeholder="例：大学生")

# --- ② Big Five ---
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
    for idx, (item, trait, reverse) in enumerate(bigfive_items, start=1):
        label = f"**{idx}. {item}**"
        val = st.slider(label, 1, 5, 3, key=f"bf_{idx}_{item}")
        if reverse:
            val = 6 - val
        scores.setdefault(trait, []).append(val)

    trait_scores = {k: np.mean(v) for k, v in scores.items()}
    st.subheader("Big Five スコア")
    st.dataframe(pd.DataFrame(trait_scores.items(), columns=["性格特性", "スコア"]))

# --- ③ AHP ---
with tabs[2]:
    st.header("ESG優先度測定")
    st.markdown("""
    以下の項目では、2つの要素が並んで表示されます。  
    バーの位置を左右に動かして、どちらをどの程度優先するかを選んでください。
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
    st.subheader("主要3要素の優先度")
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

    st.divider()
    st.subheader("AHP結果のまとめ")
    top_category = labels_main[np.argmax(priorities_main)]
    if top_category in all_priorities:
        top_sub = max(all_priorities[top_category].items(), key=lambda x: x[1])[0]
        st.markdown(f"""
        あなたが最も重視しているのは **「{top_category}」** です。  
        その中でも特に **「{top_sub}」** を重視している傾向が見られます。
        """)

# --- ④ 投資提案 ---
with tabs[3]:
    st.header("投資先提案")

    # 【pypfoptの遅延インポート】
    # Streamlit Cloudでのseaborn-deep/pypfopt競合エラーを避けるため、
    # styleパッチが完全に適用された後にインポートする
    try:
        from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
        import pypfopt.plotting as pplot
        pplot.plt = plt
    except Exception as e:
        st.error(f"ポートフォリオ最適化ライブラリのロード中にエラーが発生しました。Streamlit環境の問題です: {e}")
        st.warning("このセクションをスキップします。")
        st.stop()


    # データはload_mock_dataで取得
    df = df_csr 

    # --- ESGスコアの計算とランキング ---
    all_labels = (
        list(all_priorities["環境"].keys())
        + list(all_priorities["社会"].keys())
        + list(all_priorities["ガバナンス"].keys())
    )

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
        # .get()でキーが存在しない場合に0を返す処理を再現
        "自然資源": df.get("自然資源スコア", [0] * len(df["社名"])), 
        "人権・インクルージョン": df["人権DDスコア"],
        "雇用・労働慣行": df["有休スコア"],
        "多様性・公平性": df["女性比率スコア"],
        "取締役会構成・少数株主保護": df["取締役評価スコア"],
        "統治とリスク管理": df["内部通報スコア"]
    })

    dummy_csr["スコア"] = dummy_csr[all_labels].dot(weights)
    result = dummy_csr.sort_values("スコア", ascending=False).head(3)
    
    st.subheader("優先度の高いESG特性に基づく推奨企業")
    st.dataframe(result[["企業名", "スコア"]])
    
    if len(result) == 0:
        st.warning("推奨企業を特定できませんでした。")
        st.stop()
        
    # === 株価データ利用 ===
    df_price_clean = df_price[result["企業名"].tolist()].dropna()

    if df_price_clean.empty:
        st.warning("選択された企業の株価データがありません。モックデータでは企業A/B/C/D/Eが使われています。")
        st.stop()

    # === 平均リターンと共分散 ===
    mu = expected_returns.mean_historical_return(df_price_clean, frequency=52)
    S = risk_models.sample_cov(df_price_clean, frequency=52)

    # ===== 最適ポートフォリオ（シャープレシオ最大） =====
    ef_sharpe = EfficientFrontier(mu, S)
    ef_sharpe.max_sharpe(risk_free_rate=0.02)
    cleaned_weights = ef_sharpe.clean_weights()

    st.subheader("最適ポートフォリオ（シャープレシオ最大）")
    weights_df = pd.DataFrame(cleaned_weights.items(), columns=["銘柄", "投資比率"])
    weights_df["投資比率"] = (weights_df["投資比率"] * 100).map('{:.2f}%'.format)
    weights_df = weights_df[weights_df["投資比率"] != '0.00%']
    st.dataframe(weights_df, hide_index=True)


    # ===== 効率的フロンティアの描画 =====
    st.subheader("効率的フロンティア")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 日本語フォント設定（図にも反映させる）
    mpl.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'IPAexGothic', 'Hiragino Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    # --- 効率的フロンティア（線）を描画 ---
    ef_plot = EfficientFrontier(mu, S)
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False, lw=2, color='darkgreen')

    # ===== ランダムポートフォリオの追加 =====
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))  # [リターン, リスク, シャープレシオ]

    for i in range(num_portfolios):
        weights_rand = np.random.random(len(mu))
        weights_rand /= np.sum(weights_rand)  # 合計が1になるよう正規化
        
        # 各ポートフォリオのリターン・リスクを計算
        portfolio_return = np.dot(weights_rand, mu)
        portfolio_stddev = np.sqrt(np.dot(weights_rand.T, np.dot(S, weights_rand)))
        
        # 無リスク金利を考慮したシャープレシオ（例：2%）
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_stddev
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio

    # === 散布図として描画 ===
    ax.scatter(results[1, :], results[0, :],
               c="lightblue", alpha=0.3, s=10,
               label="ランダムポートフォリオ")


    # ===== 資本市場線などを追加 =====
    risk_free_rate = 0.02
    ef_tangent = EfficientFrontier(mu, S) 
    ef_tangent.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangent, std_tangent, _ = ef_tangent.portfolio_performance()

    # 資本市場線 (CML) の描画
    x = np.linspace(0, std_tangent * 1.5, 100)
    y = risk_free_rate + (ret_tangent - risk_free_rate) / std_tangent * x
    ax.plot(x, y, "b--", label="資本市場線", alpha=0.7)

    # 特殊な点の描画
    ax.scatter(0, risk_free_rate, c="g", s=100, label="無リスク資産（点A）", zorder=3)
    ax.scatter(std_tangent, ret_tangent, c="r", s=200, marker="*", label="最大シャープレシオ点（点B）", zorder=3)
    
    # 軸ラベルとタイトル
    ax.set_title("効率的フロンティアと資本市場線")
    ax.set_xlabel("リスク（ボラティリティ）")
    ax.set_ylabel("リターン")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)
    plt.close(fig) # メモリリーク防止のため

    st.markdown("""
    ---
    **補足:** このアプリは、外部ファイル（`スコア付きESGデータ - コピー.xlsx` と `CSR企業_株価データ（週次）.csv`）が存在しない場合、テスト用の**モックデータ**を使用して実行されています。
    実際に利用する際は、適切なデータをStreamlit Cloudの環境にアップロードするか、Google Sheets API経由で読み込むようコードを調整してください。
    """)
