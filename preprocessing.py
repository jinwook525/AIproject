import numpy as np
import pandas as pd
import torch
import mysql.connector
import os
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# 환경 변수 로드
load_dotenv()

# MySQL 연결 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def create_connection():
    """ MySQL 연결을 생성 """
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def fetch_latest_ocean_data():
    """ MySQL에서 최신 ocean 데이터 가져오기 """
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT wind_speed, air_temperature, humidity, air_pressure, water_temperature, salinity
        FROM oceandata
        ORDER BY datetime DESC
        LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return list(result.values())  # 딕셔너리를 리스트로 변환
        else:
            return None
    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        return None


def fetch_latest_ais_data():
    """ 각 선박(MMSI)별 최신 2개 데이터 가져오기 (현재 위치 + 이전 위치) """
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        # 각 선박(MMSI)별 최신 2개 데이터 가져오기
        query = """
        WITH RankedAIS AS (
            SELECT *, 
                ROW_NUMBER() OVER (PARTITION BY mmsi ORDER BY timestamp DESC) AS rn
            FROM aisdata
        )
        SELECT * FROM RankedAIS WHERE rn <= 2;
        """
        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        if not results:
            return None  # 데이터 없음

        # 데이터를 데이터프레임으로 변환
        df = pd.DataFrame(results)

        # MMSI별로 최신 데이터 2개 존재하는지 확인
        if df["mmsi"].nunique() == 0 or df.shape[0] < 2:
            return None  # 각 선박별로 2개 데이터가 존재하지 않으면 거리 계산 불가

        return df

    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        return None


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """ 벡터 연산으로 거리 계산 """
    R = 6371  # 지구 반경 (km)
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c  # km 단위 거리 반환

def preprocess():
    """ 각 선박(MMSI)별로 이동 거리, 신호 소실을 계산하고 전처리 """
    df = fetch_latest_ais_data()

    if df is None:
        print("No valid data available. Skipping processing.")
        return None

    # 시간 차이 계산
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["mmsi", "timestamp"])  # MMSI별 정렬
    df["time_diff"] = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds()

    # 신호 소실 여부 (5분(300초) 이상 신호 없음)
    df["signal_loss"] = df["time_diff"] > 300

    # 이동 거리 계산
    df["prev_lat"] = df.groupby("mmsi")["lat"].shift(1)
    df["prev_lon"] = df.groupby("mmsi")["lon"].shift(1)

    df["distance"] = df.apply(lambda row: haversine_vectorized(row["prev_lat"], row["prev_lon"], row["lat"], row["lon"]) 
                              if not pd.isna(row["prev_lat"]) else np.nan, axis=1)

    # 이동 거리 이상치 필터링 (5km 이상 이동한 경우 + 신호 소실 없으면 제거)
    df = df[~((df["distance"] > 5) & (~df["signal_loss"]))]

    # 필요없는 컬럼 삭제
    df = df.drop(columns=["prev_lat", "prev_lon", "signal_loss", "time_diff"])

    # 선형 보간 적용 (speed, turn, course, heading)
    cols_to_interpolate = ["speed", "turn", "course", "heading"]
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method="linear", limit_direction="both")

    # 스케일링 적용
    scale_columns = ["turn", "speed", "accuracy", "course", "heading", "distance"]
    scaler = ColumnTransformer([
        ("robust", RobustScaler(), ["turn"]),
        ("minmax", MinMaxScaler(), scale_columns)
    ])
    normalized_data = scaler.fit_transform(df[scale_columns])

    # PyTorch Tensor 변환 (각 MMSI별 데이터 반환)
    tensors = {mmsi: torch.tensor(normalized_data[idx], dtype=torch.float32)
               for idx, mmsi in enumerate(df["mmsi"].values)}

    return tensors  # MMSI별 텐서 반환

if __name__ == "__main__":
    # 🔥 테스트 실행
    processed_data = preprocess()
    print("실시간 전처리된 데이터:", processed_data)


