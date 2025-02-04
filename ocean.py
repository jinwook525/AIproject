import time
import mysql.connector
import requests
from datetime import datetime
import schedule
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 및 DB 설정
SERVICE_KEY = os.getenv("SERVICE_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# MySQL 연결 객체
conn = None

def create_connection():
    """ MySQL 연결 생성 """
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            pool_name="mypool",
            pool_size=5
        )
    except mysql.connector.Error as e:
        print(f"MySQL connection error: {e}")
        return None

def ensure_connection():
    """ 연결이 없거나 끊어졌다면 재연결 """
    global conn
    if conn is None or not conn.is_connected():
        print("Database connection is not active. Attempting to reconnect...")
        conn = create_connection()
        if conn and conn.is_connected():
            print("Reconnected to the database.")
        else:
            print("Failed to reconnect to the database.")

def fetch_api_data():
    """ API에서 데이터를 가져옴 """
    url = "http://marineweather.nmpnt.go.kr:8001/openWeatherNow.do"
    params = {
        "serviceKey": SERVICE_KEY,
        "resultType": "json",
        "mmaf": "101",
        "mmsi": "994401597",
        "dataType": "2"
    }

    try:
        response = requests.get(url, params=params, timeout=10)  # 타임아웃 추가
        response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
        return response.json()  # JSON 데이터 반환
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None

def upsert_data(data):
    """ MySQL에 데이터 삽입 또는 업데이트 """
    if data is None:
        print("No data received from API. Skipping database update.")
        return

    ensure_connection()  # DB 연결 확인 및 재연결
    if conn is None or not conn.is_connected():
        print("Database connection is unavailable. Data not saved.")
        return

    try:
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO oceandata (
            datetime, mmaf_code, mmaf_name, mmsi_code, mmsi_name, wind_direct, wind_speed,
            surface_curr_drc, surface_curr_speed, air_temperature, humidity, air_pressure,
            water_temperature, salinity, latitude, longitude
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            wind_direct = VALUES(wind_direct),
            wind_speed = VALUES(wind_speed),
            surface_curr_drc = VALUES(surface_curr_drc),
            surface_curr_speed = VALUES(surface_curr_speed),
            air_temperature = VALUES(air_temperature),
            humidity = VALUES(humidity),
            air_pressure = VALUES(air_pressure),
            water_temperature = VALUES(water_temperature),
            salinity = VALUES(salinity);
        """

        if "result" in data and "recordset" in data["result"]:
            for entry in data["result"]["recordset"]:
                try:
                    cursor.execute(insert_query, (
                        datetime.strptime(entry["DATETIME"], "%Y%m%d%H%M%S"),
                        entry["MMAF_CODE"],
                        entry["MMAF_NM"],
                        entry["MMSI_CODE"],
                        entry["MMSI_NM"],
                        float(entry["WIND_DIRECT"]) if entry["WIND_DIRECT"] != "미제공" else None,
                        float(entry["WIND_SPEED"]) if entry["WIND_SPEED"] != "미제공" else None,
                        float(entry["SURFACE_CURR_DRC"]) if entry["SURFACE_CURR_DRC"] != "미제공" else None,
                        float(entry["SURFACE_CURR_SPEED"]) if entry["SURFACE_CURR_SPEED"] != "미제공" else None,
                        float(entry["AIR_TEMPERATURE"]) if entry["AIR_TEMPERATURE"] != "미제공" else None,
                        float(entry["HUMIDITY"]) if entry["HUMIDITY"] != "미제공" else None,
                        float(entry["AIR_PRESSURE"]) if entry["AIR_PRESSURE"] != "미제공" else None,
                        float(entry["WATER_TEMPER"]) if entry["WATER_TEMPER"] != "미제공" else None,
                        float(entry["SALINITY"]) if entry["SALINITY"] != "미제공" else None,
                        round(float(entry["LATITUDE"]), 5),
                        round(float(entry["LONGITUDE"]), 5)
                    ))
                except Exception as e:
                    print(f"Error inserting data: {e}")

            conn.commit()
            print("Data successfully inserted or updated.")

        cursor.close()
    except mysql.connector.Error as e:
        print(f"MySQL error: {e}")
        ensure_connection()  # 연결이 끊어졌다면 재연결 시도
    except Exception as e:
        print(f"Unexpected error: {e}")

# 초기에 API 호출 및 DB 업데이트
api_data = fetch_api_data()
upsert_data(api_data)

# 스케줄링: 10분마다 데이터 업데이트
schedule.every(10).minutes.do(lambda: upsert_data(fetch_api_data()))

try:
    print("Scheduler is running... Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("Scheduler stopped. Closing database connection...")
    if conn and conn.is_connected():
        conn.close()
