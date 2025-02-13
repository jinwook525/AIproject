import torch
import torch.nn as nn
import logging
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝 출력 사용

# ✅ 저장된 모델과 동일한 설정 사용
INPUT_SIZE = 30  # 🚨 저장할 때 사용한 X_train.shape[2] 값과 동일해야 함
HIDDEN_SIZE = 128  # 🚨 저장할 때 사용한 hidden_size 값과 동일해야 함
NUM_LAYERS = 2  # 동일해야 함
OUTPUT_SIZE = 2  # 동일해야 함
DROPOUT = 0.0  # 동일해야 함

# ✅ 모델 초기화 후 가중치 로드
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)

try:
    state_dict = torch.load("lstm_model_v1.pth", map_location=torch.device("cpu"))  # ✅ 가중치만 로드
    model.load_state_dict(state_dict)  # ✅ 가중치를 모델에 적용
    model.eval()  # ✅ 평가 모드로 전환
    print("✅ 모델 가중치를 성공적으로 로드했습니다!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

def predict(input_data):
    """입력 데이터를 받아 LSTM 모델로 예측 수행"""
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)  # NumPy 배열을 Tensor로 변환
    
    logging.info(f"📌 원본 입력 데이터 shape: {input_data.shape}")  # ✅ 변환 전 shape 확인

    # ✅ 차원 변환 (LSTM이 요구하는 3D 입력 형태로 변경)
    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
    elif input_data.dim() == 2:
        input_data = input_data.unsqueeze(0)  # (1, sequence_length, input_size)

    logging.info(f"📌 변환 후 입력 데이터 shape: {input_data.shape}")  # ✅ 변환 후 shape 확인

    with torch.no_grad():
        output = model(input_data)
    
    return output.squeeze().tolist()  # 예측값 반환

