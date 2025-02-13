import numpy as np
import pandas as pd
import mysql.connector
import os
import torch
from dotenv import load_dotenv
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

# 환경 변수 로드
load_dotenv()

# MySQL 연결 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def create_connection():
    """ MySQL 연결 생성 """
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def fetch_latest_ais_data():
    """ 각 선박(MMSI)별 최신 2개 데이터 가져오기 """
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        WITH RankedAIS AS (
            SELECT *, 
                ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY created_at DESC) AS rn
            FROM aisdatareal
        )
        SELECT * FROM RankedAIS WHERE rn <= 2;
        """
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        if not results:
            return None  # 데이터 없음

        df = pd.DataFrame(results)
        print("✅ 데이터베이스에서 가져온 MMSI 목록:", df["mmsi"].unique())  

        if df["mmsi"].nunique() == 0 or df.shape[0] < 2:
            return None  # MMSI별 최신 2개 데이터가 없으면 거리 계산 불가

        return df

    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        return None

def fetch_latest_ocean_data():
    """ MySQL에서 최신 해양 데이터 가져오기 """
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT datetime, wind_speed, air_temperature, humidity, 
               air_pressure, water_temperature, salinity
        FROM oceandata
        ORDER BY datetime DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        if not results:
            return None  # 데이터 없음

        return pd.DataFrame(results)  # 데이터프레임으로 변환

    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        return None

def preprocess_ais_data():
    """ 데이터베이스에서 불러온 AIS 데이터를 전처리 """
    df_ais = fetch_latest_ais_data()
    df_ocean = fetch_latest_ocean_data()

    if df_ais is None or df_ocean is None or df_ocean.empty:
        print("No valid ocean data available. Skipping processing.")
        return None

    # 시간 데이터 변환
    df_ais["created_at"] = pd.to_datetime(df_ais["created_at"])
    df_ocean["datetime"] = pd.to_datetime(df_ocean["datetime"])

    # ✅ 대한민국 경계 내 데이터 필터링
    lat_min, lat_max = 34.5, 35.5
    lon_min, lon_max = 128.5, 130.0
    df_ais = df_ais[(df_ais['lat'] >= lat_min) & (df_ais['lat'] <= lat_max) &
                    (df_ais['lon'] >= lon_min) & (df_ais['lon'] <= lon_max)]
    print(f"🔹 대한민국 경계 내 데이터 크기: {df_ais.shape}")

    # ✅ 해양 데이터 병합
    df_ocean["key"] = 1  
    df_ais["key"] = 1
    merged_df = df_ais.merge(df_ocean, on="key", how="left").drop(columns=["key"])
    merged_df["ocean_time_diff"] = abs(merged_df["created_at"] - merged_df["datetime"])
    merged_df = merged_df.sort_values(["mmsi", "ocean_time_diff"]).drop_duplicates(subset=["mmsi"], keep="first")
    merged_df = merged_df.drop(columns=["datetime", "ocean_time_diff"])
    df_ais = merged_df.copy()

    # ✅ `salinity` 값 처리 (문자 → 숫자로 변환 및 결측치 처리)
    df_ais["salinity"] = df_ais["salinity"].astype(str)  # 문자열 변환
    df_ais["salinity"] = df_ais["salinity"].replace("-", np.nan)  # '-' 값을 NaN으로 변환
    df_ais["salinity"] = df_ais["salinity"].apply(lambda x: x if x.replace('.', '', 1).isdigit() else np.nan)  # 숫자가 아닌 값 NaN 처리
    df_ais["salinity"] = df_ais["salinity"].astype(float)  # float 타입으로 변환
    df_ais["salinity"].fillna(method="ffill", inplace=True)  # NaN 값 앞의 값으로 채움
    df_ais["salinity"].fillna(method="bfill", inplace=True)  # 그래도 NaN이면 뒤의 값으로 채움

    # ✅ 로그 스케일링 적용 (lat, lon)
    scaler = MinMaxScaler(feature_range=(1, 10))
    df_ais[["lat", "lon"]] = scaler.fit_transform(df_ais[["lat", "lon"]])
    df_ais[["lat", "lon"]] = np.log1p(df_ais[["lat", "lon"]])

    # ✅ `time_diff` 계산 및 NaN 처리
    df_ais["time_diff"] = df_ais.groupby("mmsi")["created_at"].diff().dt.total_seconds()
    df_ais["time_diff"].fillna(0, inplace=True)

    # ✅ 신호 소실 여부 추가
    df_ais["signal_loss"] = df_ais["time_diff"] > 300

    # ✅ `turn`, `speed`, `course`, `heading` 결측치 보간
    for col in ["turn", "speed", "course", "heading"]:
        df_ais[col] = df_ais[col].fillna(method="ffill").fillna(method="bfill")

    # ✅ Feature Engineering (추가 변수 생성)
    df_ais["acceleration"] = (df_ais["speed"] - df_ais.groupby("mmsi")["speed"].shift(1)) / df_ais["time_diff"]
    df_ais["heading_diff"] = (df_ais["heading"] - df_ais.groupby("mmsi")["heading"].shift(1) + 180) % 360 - 180

    for window in [5, 10, 30]:  
        df_ais[f"avg_speed_{window}steps"] = df_ais.groupby("mmsi")["speed"].rolling(window).mean().reset_index(level=0, drop=True)
        df_ais[f"avg_heading_{window}steps"] = df_ais.groupby("mmsi")["heading"].rolling(window).mean().reset_index(level=0, drop=True)

    # ✅ 신호 소실 및 거리 이상치 제거
    df_ais["prev_lat"] = df_ais.groupby("mmsi")["lat"].shift(1)
    df_ais["prev_lon"] = df_ais.groupby("mmsi")["lon"].shift(1)
    df_ais["distance_km"] = df_ais.apply(lambda row: geodesic((row["prev_lat"], row["prev_lon"]), (row["lat"], row["lon"])).km if pd.notnull(row["prev_lat"]) else 0, axis=1)
    df_ais = df_ais[~((df_ais["distance_km"] > 5) & (~df_ais["signal_loss"]))]
 # ✅ 스케일링 적용
    scale_columns = [
        'turn', 'speed', 'accuracy', 'course', 'heading', 
        'wind_direct', 'wind_speed', 'surface_curr_drc', 'surface_curr_speed', 
        'air_temperature', 'water_temperature', 'air_pressure', 'humidity', 'salinity',
        'acceleration', 'heading_diff', 'distance_km', 'avg_speed_5steps', 'avg_heading_5steps',
        'position_change_5steps', 'avg_speed_10steps', 'avg_heading_10steps', 'position_change_10steps', 
        'avg_speed_30steps', 'avg_heading_30steps', 'position_change_30steps'
    ]

    # RobustScaler & MinMaxScaler 정의
    robust_columns = ['turn', 'acceleration', 'heading_diff']
    minmax_columns = [col for col in scale_columns if col not in robust_columns]

    # NaN 값 처리 및 이상치 변환
    df_ais[robust_columns] = df_ais[robust_columns].replace([np.inf, -np.inf], np.nan)
    for col in robust_columns:
        df_ais[col].fillna(method='ffill', inplace=True)
        df_ais[col].fillna(method='bfill', inplace=True)
        df_ais[col].fillna(0, inplace=True)

    # ColumnTransformer 적용
    scaler = ColumnTransformer([
        ("robust", RobustScaler(), robust_columns),
        ("minmax", MinMaxScaler(), minmax_columns)
    ], remainder="passthrough")

    # 데이터 스케일링
    scaled_data = scaler.fit_transform(df_ais)
    # ✅ PyTorch Tensor 변환
    tensors = {mmsi: torch.tensor(row, dtype=torch.float32) for mmsi, row in zip(df_ais["mmsi"], df_ais.values)}

    print("🚀 전처리 완료!")
    return tensors

if __name__ == "__main__":
    processed_data = preprocess_ais_data()
    print("🚀 전처리된 데이터:", processed_data)
