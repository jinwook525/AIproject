from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import torch
import preprocessing  # 전처리 모듈
import model  # AI 모델 모듈
import logging
import numpy as np
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running!"})

logging.basicConfig(filename="debug_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ StandardScaler 불러오기
lat_lon_scaler = preprocessing.get_lat_lon_scaler()

@app.route("/predict", methods=["GET"])
def predict():
    """ 최신 데이터 불러와 실시간 예측 수행 """
    processed_data = preprocessing.preprocess()  # 전처리 수행

    if processed_data is None:
        return jsonify({"error": "No valid data available"}), 400

    predictions = {}
    for mmsi, tensor in processed_data.items():
        #print(f"Processing MMSI {mmsi}, tensor shape: {tensor.shape}")

        # 모델 예측 수행
        pred = model.predict(tensor)

        # ✅ pred가 리스트일 경우 numpy 배열로 변환
        if isinstance(pred, list):
            pred = np.array(pred)

        # ✅ pred가 torch.Tensor이면 변환
        elif isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        # 🚀 lat, lon을 원래 값으로 변환 (역스케일링)
        if pred.ndim == 1:  # ✅ 1차원 배열인 경우 reshape 필요
            pred = pred.reshape(1, -1)

        pred[:, 0:2] = lat_lon_scaler.inverse_transform(pred[:, 0:2])  # ✅ 첫 2개 컬럼(lat, lon)만 변환

        predictions[mmsi] = {
            "prediction": pred.tolist(),
        }

    return jsonify({"predictions": predictions})




@socketio.on("real_time_request")
def handle_realtime_request():
    """ WebSocket을 통해 실시간 요청 처리 """
    processed_data = preprocessing.preprocess()  # 전처리 수행

    if processed_data is None:
        emit("real_time_response", {"error": "No valid data available"})
        return

    # 각 MMSI별 AI 모델 예측 수행
    predictions = {mmsi: model.predict(tensor) for mmsi, tensor in processed_data.items()}

    emit("real_time_response", {"predictions": predictions})

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
