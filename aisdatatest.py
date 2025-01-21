import os
import asyncio
import websockets
import logging
from dotenv import load_dotenv
import signal
import json
import mysql.connector
from mysql.connector import Error

# .env 파일 로드
load_dotenv()

# 디버깅 로깅 활성화
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# DB 연결 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
SERVICE_KEY = os.getenv("SERVICE_KEY")

conn = None  # DB 연결 객체
stop_event = asyncio.Event()  # 종료 플래그

def create_connection():
    """MySQL 연결 생성."""
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None

def ensure_connection():
    """DB 연결 상태 확인 및 재연결."""
    global conn
    if conn is None or not conn.is_connected():
        logging.info("Reconnecting to the database...")
        conn = create_connection()
        if conn:
            logging.info("Database reconnected.")
        else:
            logging.error("Failed to reconnect to the database.")

def save_to_database(table, data, sql):
    """데이터를 DB에 저장."""
    ensure_connection()
    if conn is None:
        logging.error("Database connection not available. Data not saved.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(sql, data)
        conn.commit()
        logging.info(f"Data saved to {table}.")
    except Error as e:
        logging.error(f"Error saving data to {table}: {e}")
    finally:
        cursor.close()

def process_type8_message(parsed_data):
    """Type 8 메시지 처리 및 저장."""
    sql = '''
        INSERT INTO type8data (mmsi, dac, fid, data)
        VALUES (%s, %s, %s, %s)
    '''
    data = (
        parsed_data.get("mmsi"),
        parsed_data.get("dac"),  # Application Identifier (DAC)
        parsed_data.get("fid"),  # Function Identifier (FID)
        parsed_data.get("data")  # Additional data
    )
    save_to_database("type8data", data, sql)

def process_position_message(parsed_data):
    """Type 1, 2, 3 메시지 처리 및 저장."""
    sql = '''
        INSERT INTO aisdatatest (msg_type, mmsi, status, turn, speed, accuracy, lon, lat, course, heading)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    data = (
        parsed_data["msg_type"],
        parsed_data["mmsi"],
        parsed_data.get("status"),
        parsed_data.get("turn"),
        parsed_data.get("speed"),
        parsed_data.get("accuracy"),
        parsed_data.get("lon"),
        parsed_data.get("lat"),
        parsed_data.get("course"),
        parsed_data.get("heading")
    )
    save_to_database("aisdata", data, sql)

async def connect_to_ais_server():
    """AIS 서버 연결 및 메시지 수신."""
    uri = os.getenv("AIS_SERVER_URI")
    if not uri:
        logging.error("환경 변수 'AIS_SERVER_URI'가 설정되지 않았습니다.")
        return

    try:
        logging.info(f"Connecting to AIS server at {uri}...")
        async with websockets.connect(uri) as websocket:
            logging.info("WebSocket connection established.")
            await websocket.send("INITIAL_MESSAGE")
            logging.info("Sent initial message.")

            while not stop_event.is_set():
                try:
                    data = await websocket.recv()
                    parsed_data = json.loads(data)
                    logging.debug(f"Received data: {parsed_data}")

                    msg_type = parsed_data.get("msg_type")
                    if msg_type == 8:  # Type 8 메시지 처리
                        process_type8_message(parsed_data)
                    elif msg_type in {1, 2, 3}:  # Position Report Class A
                        process_position_message(parsed_data)
                    else:
                        logging.warning(f"Unhandled message type: {msg_type}")

                except json.JSONDecodeError:
                    logging.error("Failed to parse JSON.")
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed.")
                    break

    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
    finally:
        logging.info("WebSocket connection closed.")

def signal_handler():
    """종료 신호 처리."""
    logging.info("Received termination signal. Shutting down.")
    stop_event.set()

def main():
    signal.signal(signal.SIGINT, lambda *_: signal_handler())
    signal.signal(signal.SIGTERM, lambda *_: signal_handler())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(connect_to_ais_server())
    finally:
        loop.close()
        logging.info("Event loop closed.")

if __name__ == "__main__":
    main()
