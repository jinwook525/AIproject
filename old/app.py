from flask import Flask, jsonify
import torch
import preprocessing  # 전처리 모듈
import model  # AI 모델 모듈
import logging
import numpy as nppy

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running!"})

logging.basicConfig(filename="debug_log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ StandardScaler 불러오기
lat_lon_scaler = preprocessing.get_lat_lon_scaler()


@app.route("/predict", methods=["GET"])
def predict():
    """ 최신 데이터 불러와 실시간 예측 수행 """
    try:
        processed_data = preprocessing.preprocess()  # 전처리 수행

        if processed_data is None or not processed_data:
            logging.error("❌ No valid data available for prediction.")
            return jsonify({"error": "No valid data available"}), 400

        print("✅ Flask에서 받은 processed_data의 mmsi 목록:", list(processed_data.keys()))
        logging.info(f"✅ MMSI 목록: {list(processed_data.keys())}")

        predictions = {}

        for mmsi, tensor in processed_data.items():
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32).to(model.device)

            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)  # (1, sequence_length, features)

            # ✅ 입력 데이터 차원 확인
            logging.info(f"📌 MMSI {mmsi} 예측 입력 shape: {tensor.shape}")

            try:
                pred = model.predict(tensor)
            except Exception as e:
                logging.error(f"❌ Prediction error for MMSI {mmsi}: {e}")
                return jsonify({"error": f"Prediction failed for MMSI {mmsi}", "details": str(e)}), 500

            if isinstance(pred, list):
                pred = np.array(pred)
            elif isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()

            if pred.ndim == 1:
                pred = pred.reshape(1, -1)

            try:
                pred[:, 0:2] = lat_lon_scaler.inverse_transform(pred[:, 0:2])
            except Exception as e:
                logging.error(f"⚠️ Scaling error for MMSI {mmsi}: {e}")
                continue  # 오류 발생 시 해당 MMSI 데이터 건너뛰기

            predictions[mmsi] = {
                "mmsi": mmsi,
                "prediction": pred.tolist(),
            }

        return jsonify({"predictions": predictions})

    except Exception as e:
        logging.error(f"🚨 Internal Server Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
