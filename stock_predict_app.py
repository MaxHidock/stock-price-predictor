import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error

st.title("📈 株価予測アプリ")

# 銘柄コード入力（デフォルト：9023.T）
symbol = st.text_input("銘柄コードを入力（例：AAPL、GOOG、7203.T）", "9023.T")

if symbol:
    try:
        # 銘柄情報と会社名取得
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get("longName") or info.get("shortName", "企業名が取得できませんでした")
        st.write(f"**銘柄名：{company_name}**")

        # ⏱ 過去30日分を日付指定で取得
        end_date = datetime.datetime.today().date()
        start_date = end_date - datetime.timedelta(days=45)  # 土日なども含めて45日さかのぼる
        data = ticker.history(start=start_date, end=end_date)

        # 欠損値除去
        data = data.dropna()

        if data.empty:
            raise ValueError("株価データが空です。")

        # XGBoostによる予測
        # 翌営業日を取得（タイムゾーン付きで）
        last_date = data.index[-1]
        next_day = last_date + pd.tseries.offsets.BDay(1)
        next_day_str = next_day.strftime("%m月%d日")  # 修正された部分

        # 特徴量作成用の関数
        def create_features(data, window_size=5):
            features = []
            labels = []
            for i in range(window_size, len(data)-1):
                features.append(data['Close'].values[i-window_size:i])
                labels.append(data['Close'].values[i+1])  # 翌営業日の終値
            return np.array(features), np.array(labels)

        # 特徴量を作成
        X_xgb, y_xgb = create_features(data, window_size=5)

        # 学習データと予測用データに分割
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X_xgb, y_xgb)

        # 予測（最新の5日分を使って翌日を予測）
        last_features = data['Close'].values[-5:].reshape(1, -1)
        xgb_pred = xgb_model.predict(last_features)[0]

        # XGBoost予測結果を最初に表示
        st.subheader(f"📊 {next_day_str}の＜XGBoost＞終値予測")
        st.write(f"{xgb_pred:.2f} 円")

        # 線形回帰による予測
        data["Day"] = range(len(data))
        X = data[["Day"]]
        y = data["Close"]

        model = LinearRegression()
        model.fit(X, y)

        next_day = [[len(data)]]
        predicted_price = model.predict(next_day)
        predicted_price_value = predicted_price.item()

        # 線形回帰予測結果をその後に表示
        st.subheader("📊 翌営業日の＜線形回帰＞終値予測")
        st.write(f"{predicted_price_value:.2f} 円")

        # データのプレビューを表示
        st.write("取得したデータのプレビュー")
        st.dataframe(data)

        st.write(f"取得開始日: {data.index.min().date()}")
        st.write(f"取得終了日: {data.index.max().date()}")
        st.write(f"データ件数: {len(data)}")

        st.subheader("📉 過去の株価データ")
        chart_data = data[["Close"]].copy()
        chart_data.index = pd.to_datetime(chart_data.index)
        st.line_chart(chart_data)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
