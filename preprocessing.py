import numpy as np
import pandas as pd
import torch
import mysql.connector
import os
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
    """ MySQL에서 해양 데이터 가져오기 (여러 개) """
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

        return pd.DataFrame(results)  # 데이터프레임으로 반환

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

        # 데이터를 데이터프레임으로 변환
        df = pd.DataFrame(results)
        # ✅ 데이터베이스에서 가져온 MMSI 확인
        print("✅ 데이터베이스에서 가져온 MMSI 목록:", df["mmsi"].unique()) 
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

# ✅ StandardScaler 저장 (lat, lon 역스케일링용)
lat_lon_scaler = StandardScaler()

def fit_scalers(original_data):
    """ ✅ 원본 데이터(`original_data`)를 기반으로 StandardScaler 학습 """
    global lat_lon_scaler
    lat_lon_scaler.fit(original_data[["lat", "lon"]])

def get_lat_lon_scaler():
    """ ✅ lat, lon 역스케일링을 위한 StandardScaler 반환 """
    return lat_lon_scaler

def get_original_lat(mmsi):
    """ 특정 MMSI의 원래 위도(lat)를 반환 """
    df_original = fetch_latest_ais_data()  # ✅ 원래 데이터를 가져오기
    original_row = df_original[df_original["mmsi"] == mmsi]
    return original_row["lat"].values[0] if not original_row.empty else None

def get_original_lon(mmsi):
    """ 특정 MMSI의 원래 경도(lon)를 반환 """
    df_original = fetch_latest_ais_data()  # ✅ 원래 데이터를 가져오기
    original_row = df_original[df_original["mmsi"] == mmsi]
    return original_row["lon"].values[0] if not original_row.empty else None

def preprocess():
    """AIS 데이터와 해양 데이터를 시간 기준으로 병합하여 전처리"""
    df_ais = fetch_latest_ais_data()  # AIS 데이터 가져오기
    df_ocean = fetch_latest_ocean_data()  # 해양 데이터 가져오기

    if df_ais is None or df_ocean is None or df_ocean.empty:
        print("No valid data available. Skipping processing.")
        return None

    # `created_at`을 datetime 형식으로 변환
    df_ais["created_at"] = pd.to_datetime(df_ais["created_at"])
    df_ocean["datetime"] = pd.to_datetime(df_ocean["datetime"])

    # `created_at`을 기준으로 가장 가까운 `oceandata` 매칭
    df_ocean["key"] = 1  # 병합을 위한 가상 키 생성
    df_ais["key"] = 1

    merged_df = df_ais.merge(df_ocean, on="key", how="left").drop(columns=["key"])

    # 새로운 `ocean_time_diff` 변수 생성하여 가장 가까운 데이터만 유지
    merged_df["ocean_time_diff"] = abs(merged_df["created_at"] - merged_df["datetime"])
    merged_df = merged_df.sort_values(["mmsi", "ocean_time_diff"]).drop_duplicates(subset=["mmsi"], keep="first")

    # `datetime` 컬럼 삭제 (더 이상 필요 없음)
    merged_df = merged_df.drop(columns=["datetime", "ocean_time_diff"])

    # 병합된 데이터 사용
    df_ais = merged_df.copy()  # 이후 모든 처리를 `df_ais`에서 수행
    
    # ✅ `lat_lon_scaler` 학습 (lat, lon 변환을 위해 필수)
    fit_scalers(df_ais)
    
    #  created_at이 NaT인지 확인
    if df_ais["created_at"].isna().sum() > 0:
        print("⚠️ Warning: NaT values detected in created_at. Filling with previous values.")
        df_ais["created_at"].fillna(method="ffill", inplace=True)  # 이전 값으로 채우기
    
    #  `lat`, `lon`이 NaN인 행을 삭제
    df_ais = df_ais.dropna(subset=["lat", "lon"])
    
    #  `turn`, `speed`, `course`, `heading` NaN 값 보간 (앞/뒤 값으로 채움)
    df_ais["turn"] = df_ais["turn"].fillna(method="ffill").fillna(method="bfill")
    df_ais["speed"] = df_ais["speed"].fillna(method="ffill").fillna(method="bfill")
    df_ais["course"] = df_ais["course"].fillna(method="ffill").fillna(method="bfill")
    df_ais["heading"] = df_ais["heading"].fillna(method="ffill").fillna(method="bfill")
    #  시간 차이 계산
    df_ais = df_ais.sort_values(["mmsi", "created_at"])  # MMSI별 정렬
    df_ais["time_diff"] = df_ais.groupby("mmsi")["created_at"].diff().dt.total_seconds()

    #  NaN을 0으로 변환
    df_ais["time_diff"].fillna(0, inplace=True)

    #  최종 NaN 개수 확인
    print("NaN values in time_diff after fillna:", df_ais["time_diff"].isna().sum())

    # 신호 소실 여부 (5분(300초) 이상 신호 없음)
    df_ais["signal_loss"] = df_ais["time_diff"] > 300

    # 이동 거리 계산
    df_ais["prev_lat"] = df_ais.groupby("mmsi")["lat"].shift(1)
    df_ais["prev_lon"] = df_ais.groupby("mmsi")["lon"].shift(1)

    df_ais["distance"] = df_ais.apply(lambda row: haversine_vectorized(row["prev_lat"], row["prev_lon"], row["lat"], row["lon"]) 
                                      if not pd.isna(row["prev_lat"]) else np.nan, axis=1)

    # 이동 거리 이상치 필터링 (5km 이상 이동한 경우 + 신호 소실 없으면 제거)
    df_ais = df_ais[~((df_ais["distance"] > 5) & (~df_ais["signal_loss"]))]

    # 위도(lat) 또는 경도(lon)가 0.1 이상 차이나는 이상치 데이터 제거
    df_ais["lat_diff"] = abs(df_ais["lat"] - df_ais["prev_lat"])
    df_ais["lon_diff"] = abs(df_ais["lon"] - df_ais["prev_lon"])

    # 이상치 조건: (신호 소실 없음) + (lat/lon 변화량 0.1 이상) + (이동 거리 5km 미만)
    df_ais = df_ais[~(((df_ais["lat_diff"] > 0.1) | (df_ais["lon_diff"] > 0.1)) & (df_ais["distance"] < 5) & (~df_ais["signal_loss"]))]

    # 필요없는 컬럼 삭제
    df_ais = df_ais.drop(columns=["prev_lat", "prev_lon", "signal_loss", "lat_diff", "lon_diff","distance"])

    # status 원-핫 인코딩 수행
    df_ais["status"] = df_ais["status"].fillna("nan")  # NaN 값을 'nan' 문자열로 변환
    status_encoded = pd.get_dummies(df_ais["status"], prefix="status")  # 원-핫 인코딩
    status_encoded = status_encoded.astype(int)  # True/False를 0/1로 변환

    # 모든 가능한 status 컬럼을 강제로 포함 (없으면 0으로 추가)
    expected_status_columns = [
        "status_0.0", "status_1.0", "status_2.0", "status_3.0", "status_5.0",
        "status_6.0", "status_7.0", "status_8.0", "status_9.0", "status_10.0",
        "status_11.0", "status_12.0", "status_15.0"
    ]
    for col in expected_status_columns:
        if col not in status_encoded:
            status_encoded[col] = 0  # 없는 컬럼은 0으로 채우기

    # 원본 데이터와 원-핫 인코딩된 데이터 병합
    df_ais = pd.concat([df_ais, status_encoded], axis=1)
    df_ais = df_ais.drop(columns=["status"])  # 원본 'status' 컬럼 삭제

    # 모델이 요구하는 30개 입력 변수만 선택
    selected_columns = [
        "turn", "speed", "accuracy", "course", "heading", "lat", "lon",
        "time_diff", "wind_direct","wind_speed", "surface_curr_drc","surface_curr_speed","air_temperature", 
        "air_pressure", "water_temperature","humidity", "salinity"
    ] + expected_status_columns  # 강제로 포함된 status 컬럼 추가
    # 🚀 `df_ais`에 NaN 값이 있는지 확인
    print("NaN values per column before scaling:\n", df_ais.isna().sum())

    # 🚀 NaN 값이 포함된 경우 0으로 변환
    if df_ais.isna().sum().sum() > 0:
        print("⚠️ Warning: NaN values detected in df_ais before scaling. Replacing NaN with 0.")
        df_ais.fillna(0, inplace=True)

    # 🚀 df_ais에서 누락된 컬럼 확인
    print("Available columns in df_ais:", df_ais.columns.tolist())

    missing_columns = [col for col in selected_columns if col not in df_ais.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")

    # 🚀 선택된 컬럼이 누락된 경우 0으로 채움
    for col in selected_columns:
        if col not in df_ais.columns:
            df_ais[col] = 0  # 누락된 컬럼을 0으로 추가

    # 🚀 이제 KeyError 발생 없이 선택된 컬럼만 유지
    df_ais_mmsi = df_ais[["mmsi"]]
    df_ais = df_ais[selected_columns]

    # 🚀 데이터 크기 확인
    print(f"Final df_ais shape: {df_ais.shape}")
    
    

    # ✅ 스케일링이 필요한 컬럼만 분류
    robust_columns = ["turn"]  # 이상치에 강한 RobustScaler 적용
    standard_columns = ["lat", "lon"]  # StandardScaler 적용
    minmax_columns = [
        "speed", "accuracy", "course", "heading", "time_diff",
        "wind_speed", "air_temperature", "humidity", "air_pressure",
        "water_temperature", "salinity"
    ]  # MinMaxScaler 적용

    # ✅ status_* 컬럼을 제외 (원-핫 인코딩된 값이므로 스케일링 불필요)
    status_columns = [col for col in df_ais.columns if col.startswith("status_")]

    # ✅ ColumnTransformer 구성 (status_* 컬럼 제외)
    scaler = ColumnTransformer([
        ("robust", RobustScaler(), robust_columns),
        ("standard", StandardScaler(), standard_columns),
        ("minmax", MinMaxScaler(), minmax_columns)
    ], remainder="passthrough")  # ❗ status_* 컬럼을 변경 없이 유지

    # ✅ 데이터 스케일링
    normalized_data = scaler.fit_transform(df_ais)

    # ✅ 데이터 크기 확인
    print("Shape of scaled data:", normalized_data.shape)   


    # PyTorch Tensor 변환 (각 MMSI별 데이터 반환)
    tensors = {mmsi: torch.tensor(row, dtype=torch.float32)
              for mmsi, row in zip(df_ais_mmsi["mmsi"], normalized_data)}

    print('preprocessing end')
    return tensors  # MMSI별 텐서 반환




    

if __name__ == "__main__":
    # 🔥 테스트 실행
    processed_data = preprocess()
    print("실시간 전처리된 데이터:", processed_data)

