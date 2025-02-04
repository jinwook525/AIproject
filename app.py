from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import preprocessing  # 전처리 모듈
import model  # AI 모델 모듈 (모델.py가 준비되면 불러오기)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/predict", methods=["GET"])
def predict():
    """ 최신 데이터 불러와 실시간 예측 수행 """
    processed_data = preprocessing.preprocess()  # 전처리 수행

    if processed_data is None:
        return jsonify({"error": "No valid data available"}), 400

    # 각 MMSI별 AI 모델 예측 수행
    predictions = {}
    for mmsi, tensor in processed_data.items():
        pred = model.predict(tensor)  # AI 예측 수행
        lat, lon = preprocessing.get_latest_lat_lon(mmsi)  # 최신 좌표 가져오기
        predictions[mmsi] = {
            "prediction": pred.tolist(),
            "lat": lat,
            "lon": lon
        }

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
