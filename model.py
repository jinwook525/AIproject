import torch
import torch.nn as nn

# 모델 클래스 정의 (Jupyter Notebook과 동일해야 함)
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# 모델 로드
MODEL_PATH = "model.pth"
INPUT_SIZE = 30  # Train 데이터와 동일한 크기
OUTPUT_SIZE = 2  # Train 데이터의 출력 크기

model = MyModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()  # 예측 모드로 변경

def predict(input_data):
    """ 입력 데이터를 받아 AI 예측 수행 """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)  # NumPy 배열을 Tensor로 변환
    
    with torch.no_grad():
        output = model(input_data)
    return output.squeeze().tolist()  # 예측값 반환
