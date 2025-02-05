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
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
SERVICE_KEY = os.getenv("SERVICE_KEY")

# 연결 객체
conn = None

def create_connection():
    """MySQL 연결을 생성하고 반환."""
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
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def ensure_connection():
    """연결이 없거나 끊어진 경우 재연결."""
    global conn
    if conn is None or not conn.is_connected():
        print("Database connection is not active. Attempting to reconnect...")
        conn = create_connection()
        if conn and conn.is_connected():
            print("Reconnected to the database.")
        else:
            print("Failed to reconnect to the database.")
def save_to_database(data):
    """수신된 데이터를 데이터베이스에 저장."""
    ensure_connection()  # 연결 확인 및 재연결 시도
    if conn is None or not conn.is_connected():
        logging.error("Database connection is not available. Data not saved.")
        return

    try:
        cursor = conn.cursor()
        # 데이터 삽입 SQL
        sql = '''
        INSERT INTO aisdatareal (msg_type, mmsi, status, turn, speed, accuracy, lon, lat, course, heading)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        logging.debug(f"삽입할 데이터: {data}")
        logging.debug(f"실행할 SQL: {sql}")
        cursor.execute(sql, (
            data["msg_type"],
            data["mmsi"],
            data["status"],
            data["turn"],
            data["speed"],
            data["accuracy"],
            data["lon"],
            data["lat"],
            data["course"],
            data["heading"]
        ))
        conn.commit()
        logging.info("Data saved successfully.")
    except Error as e:
        logging.error(f"Error saving data to MySQL: {e}")
    finally:
        cursor.close()

# 종료 신호 플래그
stop_event = asyncio.Event()

async def connect_to_ais_server():
    uri = os.getenv("AIS_SERVER_URI")  # 환경 변수에서 URI 읽기
    if not uri:
        logging.error("환경 변수 'AIS_SERVER_URI'가 설정되지 않았습니다.")
        return

    try:
        logging.info(f"웹소켓 서버에 연결 시도: {uri}")
        async with websockets.connect(uri) as websocket:
            logging.info("웹소켓 연결 성공!")

            # 초기 메시지 전송
            initial_message = "INITIAL_MESSAGE"
            logging.info(f"서버로 초기 메시지 전송: {initial_message}")
            await websocket.send(initial_message)
            logging.info("초기 메시지를 전송했습니다.")

            # 메시지 수신 루프
            while not stop_event.is_set():
                try:
                    logging.debug("서버로부터 메시지 수신 대기 중...")
                    data = await websocket.recv()
                    logging.info(f"서버로부터 받은 데이터: {data}")

                    # JSON 데이터 파싱
                    try:
                        parsed_data = json.loads(data)
                        logging.debug(f"파싱된 데이터: {parsed_data}")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON 파싱 실패: {e}")
                        continue

                    # 데이터베이스에 저장
                    save_to_database(parsed_data)

                except websockets.exceptions.ConnectionClosedOK:
                    logging.info("서버가 연결을 정상적으로 종료했습니다.")
                    break
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"서버와의 연결이 종료되었습니다: {e}")
    except Exception as e:
        logging.error(f"웹소켓 연결 실패: {e}")
    finally:
        logging.info("웹소켓 연결 종료.")


def signal_handler(signum, frame):
    """Ctrl+C 신호를 처리하여 프로그램 종료."""
    logging.info("종료 신호를 받았습니다. 프로그램을 종료합니다.")
    stop_event.set()  # 종료 신호 설정

def main():
    # 종료 신호 핸들러 설정
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 메인 이벤트 루프 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(connect_to_ais_server())
    finally:
        logging.info("이벤트 루프 종료.")
        loop.close()

if __name__ == "__main__":
    main()
