#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv("../document/merged_data.csv")
data.head()


# In[2]:


data = data.drop(columns=['timestamp'])  # 'timestamp '을 삭제

# 결과 저장
data.to_csv('../document/drop_merged_data.csv', index=False)  # 수정된 데이터를 저장


# In[3]:


# NaN 값 확인
print(data.isna().sum())


# In[4]:


# NaN 값이 있는 행의 인덱스 찾기
first_nan_index = data['datetime'].isna().idxmax()  # NaN 값이 처음 나타나는 인덱스
print(f"'datetime' 열에서 NaN 값이 처음 나타나는 행의 인덱스: {first_nan_index}")

# NaN 값이 시작하는 행 데이터 확인
print(data.iloc[first_nan_index])


# In[5]:


# 특정 행(1172700~1172800)의 'created_at'과 'datetime' 열만 선택
subset = data.iloc[1791300:1797959][['created_at', 'datetime']]
print(subset)


# In[6]:


# 특정 행 번호 이후 데이터 삭제
row_to_drop_from = 1791300  # 삭제를 시작할 행 번호 (0부터 시작)
data_dropped = data.iloc[:row_to_drop_from]

# mmsi 값이 0인 데이터 제거
data_cleaned = data_dropped[data_dropped['mmsi'] != 0]

# 결과 확인
print(f"최종 데이터 크기: {data_cleaned.shape}")


# 결과 저장
data_cleaned.to_csv('../document/data_after_dropping.csv', index=False, encoding='utf-8-sig')

print("최종 클린 데이터가 '../document/data_after_dropping.csv'로 저장되었습니다.")


# In[8]:


df = pd.read_csv("../document/data_after_dropping.csv")
df.head()


# In[9]:


print(df.isna().sum())


# In[10]:


import pandas as pd

# 원본 데이터 로드
file_path = '../document/data_after_dropping.csv'
data = pd.read_csv(file_path)

# 시간 데이터 변환
data['created_at'] = pd.to_datetime(data['created_at'])

# 시간 순 정렬
data = data.sort_values(['mmsi', 'created_at'])

# MMSI별 선형 보간 적용
data['lat'] = data.groupby('mmsi')['lat'].transform(lambda x: x.interpolate(method='linear'))
data['lon'] = data.groupby('mmsi')['lon'].transform(lambda x: x.interpolate(method='linear'))

# 최종 데이터 저장
data.to_csv('../document/final_cleaned_datalinear.csv', index=False, encoding='utf-8-sig')

print(f"최종 데이터가 저장되었습니다. 최종 데이터 행 수: {len(data)}")


# In[11]:


import pandas as pd

# 경로 설정 (원본 데이터와 최종 데이터)
original_file_path = '../document/data_after_dropping.csv'
final_cleaned_file_path = '../document//final_cleaned_datalinear.csv'

# 데이터 로드
original_data = pd.read_csv(original_file_path)
final_cleaned_data = pd.read_csv(final_cleaned_file_path)

# 데이터 크기 비교
original_count = len(original_data)
final_cleaned_count = len(final_cleaned_data)

# 열 이름 비교
original_columns = original_data.columns.tolist()
final_cleaned_columns = final_cleaned_data.columns.tolist()

# 결과 출력
print(f"원본 데이터 행 수: {original_count}")
print(f"최종 클린 데이터 행 수: {final_cleaned_count}")
print(f"원본 데이터 열 이름: {original_columns}")
print(f"최종 클린 데이터 열 이름: {final_cleaned_columns}")

# 최종 클린 데이터에 없는 행 확인
removed_rows = original_data[~original_data.set_index(['mmsi', 'created_at']).index.isin(final_cleaned_data.set_index(['mmsi', 'created_at']).index)]
print(f"제거된 데이터 행 수: {len(removed_rows)}")

# 제거된 데이터 확인
print("제거된 데이터 예시:")
print(removed_rows
      )


# In[13]:


# # 총 lat과 lon이 결측치인 행의 갯수
# total_nan = result_df[(result_df['lat_is_nan'] & result_df['lon_is_nan'])]

# # 지금, 앞뒤로 다 없는 행의 갯수
# all_nan = result_df[
#     (result_df['lat_is_nan'] & result_df['lon_is_nan']) &
#     (~result_df['has_previous_lat'] & ~result_df['has_previous_lon']) &
#     (~result_df['has_next_lat'] & ~result_df['has_next_lon'])
# ]

# # 지금은 없지만 앞뒤로는 다 있는 행의 갯수
# surrounded_by_data = result_df[
#     (result_df['lat_is_nan'] & result_df['lon_is_nan']) &
#     (result_df['has_previous_lat'] & result_df['has_previous_lon']) &
#     (result_df['has_next_lat'] & result_df['has_next_lon'])
# ]

# # 앞이 없고 뒤만 데이터가 있는 경우
# only_next_data = result_df[
#     (result_df['lat_is_nan'] & result_df['lon_is_nan']) &
#     (~result_df['has_previous_lat'] & ~result_df['has_previous_lon']) &
#     (result_df['has_next_lat'] & result_df['has_next_lon'])
# ]

# # 앞은 있고 뒤는 없는 경우
# only_previous_data = result_df[
#     (result_df['lat_is_nan'] & result_df['lon_is_nan']) &
#     (result_df['has_previous_lat'] & result_df['has_previous_lon']) &
#     (~result_df['has_next_lat'] & ~result_df['has_next_lon'])
# ]

# # 지금은 없지만 앞 혹은 뒤에 데이터가 있는 갯수
# has_either_data = result_df[
#     (result_df['lat_is_nan'] & result_df['lon_is_nan']) &
#     ((result_df['has_previous_lat'] & result_df['has_previous_lon']) |
#      (result_df['has_next_lat'] & result_df['has_next_lon']))
# ]

# # 결과 출력
# print(f"총 lat과 lon이 결측치인 행의 갯수: {len(total_nan)}")
# print(f"지금, 앞뒤로 다 없는 행의 갯수: {len(all_nan)}")
# print(f"지금은 없지만 앞뒤로는 다 있는 행의 갯수: {len(surrounded_by_data)}")
# print(f"앞이 없고 뒤만 데이터가 있는 행의 갯수: {len(only_next_data)}")
# print(f"앞은 있고 뒤는 없는 행의 갯수: {len(only_previous_data)}")
# print(f"지금 없지만 앞 혹은 뒤에 데이터가 있는 갯수: {len(has_either_data)}")


# In[12]:


import pandas as pd

# 데이터 로드
data = pd.read_csv('../document/final_cleaned_datalinear.csv')

# NaN을 새로운 범주로 추가
data['status'] = data['status'].fillna('nan')  # NaN 값을 'nan' 문자열로 변환

# 원-핫 인코딩 수행
status_encoded = pd.get_dummies(data['status'], prefix='status')

# True/False를 0/1로 변환
status_encoded = status_encoded.astype(int)

# 원본 데이터와 원-핫 인코딩된 데이터 병합
data = pd.concat([data, status_encoded], axis=1)

# 결과 확인
print(data.head())

# 결과 저장
data.to_csv('../document/final_cleaned_data_with_nan_encodedlinear.csv', index=False, encoding='utf-8-sig')

print("결과 데이터가 'final_cleaned_data_with_nan_encoded.csv'로 저장되었습니다.")


# In[13]:


# 대한민국 위도 및 경도 범위
lat_min, lat_max = 34.5, 35.5
lon_min, lon_max = 128.5, 130.0

# 대한민국 범위 내 데이터 필터링
data = data[(data['lat'] >= lat_min) & (data['lat'] <= lat_max) &
                    (data['lon'] >= lon_min) & (data['lon'] <= lon_max)]

# 이상치 데이터 확인 (대한민국 범위를 벗어나는 데이터)
outliers = data[~((data['lat'] >= lat_min) & (data['lat'] <= lat_max) &
                  (data['lon'] >= lon_min) & (data['lon'] <= lon_max))]
print("이상치 데이터:\n", outliers)

# 결과 확인
print(f"대한민국 범위 내 데이터 크기: {data.shape}")

data.to_csv('../document/cleaned_data_korealinear.csv', index=False, encoding='utf-8-sig')
print("클린 데이터가 'cleaned_data_korea.csv'로 저장되었습니다.")


# In[17]:


# status 값의 고유 값 및 분포 확인
print("status 값 분포:")
print(data['status'].value_counts(dropna=False))


# In[18]:


# 숫자로 변환할 수 없는 값 확인
for column in data.columns:
    try:
        data[column].astype(float)
    except ValueError as e:
        print(f"열 '{column}'에 변환 불가능한 값이 있습니다.")


# In[19]:


# 문제 열에서 숫자가 아닌 값 필터링
print(data['salinity'][~data['salinity'].str.replace('.', '', 1).str.isdigit()])


# In[20]:


data['salinity'] = data['salinity'].fillna(method='ffill')  # 앞의 값으로 채움
data['salinity'] = data['salinity'].fillna(method='bfill')  # 뒤의 값으로 채움


# In[21]:


print(data['salinity'].unique())


# In[22]:


import numpy as np

# '-' 값을 NaN으로 변환
data['salinity'] = data['salinity'].replace('-', np.nan).astype(float)
# 앞의 값으로 대체
data['salinity'] = data['salinity'].fillna(method='ffill')

# 뒤의 값으로 대체
data['salinity'] = data['salinity'].fillna(method='bfill')


# In[23]:


# NaN 값 확인
print(data['salinity'].isna().sum())

# salinity 고유 값 확인
print(data['salinity'].unique())


# In[30]:


data['turn'] = data['turn'].interpolate(method='linear', limit_direction='both')
data['speed'] = data['speed'].interpolate(method='linear', limit_direction='both')
data['course'] = data['course'].interpolate(method='linear', limit_direction='both')
data['heading'] = data['heading'].interpolate(method='linear', limit_direction='both')
data['accuracy'] = data['accuracy'].interpolate(method='linear', limit_direction='both')


# In[31]:


print(data.isna().sum())


# In[46]:


import pandas as pd
import numpy as np

# Haversine 공식 사용하여 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    
    # 위도 및 경도를 라디안으로 변환
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # 차이 계산
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine 공식 적용
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # km 단위 거리 반환

    return distance

# 시간 데이터 변환 및 정렬
data['created_at'] = pd.to_datetime(data['created_at'])
data = data.sort_values(['mmsi', 'created_at'])

# Feature 리스트
features = []

# 가속도 계산
data['prev_speed'] = data.groupby('mmsi')['speed'].shift(1)
data['time_diff'] = data.groupby('mmsi')['created_at'].diff().dt.total_seconds()
data['acceleration'] = (data['speed'] - data['prev_speed']) / data['time_diff']
data['acceleration'].fillna(0, inplace=True)
features.append('acceleration')

# 회전율 계산
data['prev_heading'] = data.groupby('mmsi')['heading'].shift(1)
data['heading_diff'] = (data['heading'] - data['prev_heading'] + 180) % 360 - 180
data['heading_diff'].fillna(0, inplace=True)
features.append('heading_diff')

# 거리 계산
data['prev_lat'] = data.groupby('mmsi')['lat'].shift(1)
data['prev_lon'] = data.groupby('mmsi')['lon'].shift(1)
data['distance_km'] = data.apply(lambda row: haversine(row['prev_lat'], row['prev_lon'], row['lat'], row['lon']) if pd.notnull(row['prev_lat']) else 0, axis=1)
features.append('distance_km')

# 이동 평균 및 위치 변화량 계산
for window in [5, 10, 30]:  
    speed_feature = f'avg_speed_{window}steps'
    heading_feature = f'avg_heading_{window}steps'
    position_feature = f'position_change_{window}steps'

    data[speed_feature] = data.groupby('mmsi')['speed'].rolling(window).mean().reset_index(level=0, drop=True)
    data[heading_feature] = data.groupby('mmsi')['heading'].rolling(window).mean().reset_index(level=0, drop=True)

    data[f'prev_lat_{window}'] = data.groupby('mmsi')['lat'].shift(window)
    data[f'prev_lon_{window}'] = data.groupby('mmsi')['lon'].shift(window)
    data[position_feature] = data.apply(lambda row: haversine(row['lat'], row['lon'], row[f'prev_lat_{window}'], row[f'prev_lon_{window}']) if pd.notnull(row[f'prev_lat_{window}']) else 0, axis=1)

    features.extend([speed_feature, heading_feature, position_feature])

# 결측값 처리
for feature in features:
    data[feature].fillna(method='ffill', inplace=True)
    data[feature].fillna(method='bfill', inplace=True)
    data[feature].fillna(0, inplace=True)

# 파일 저장 경로
save_path = "../document/feature_extracted_data.csv"

# CSV로 저장
data.to_csv(save_path, index=False, encoding="utf-8-sig")



# In[47]:





# In[34]:


print("추출된 Features:", features)


# In[35]:


# 음수 값 확인 대상 열
columns_to_check = [
    'turn', 'speed', 'accuracy', 'course', 'heading', 
    'wind_direct', 'wind_speed', 'surface_curr_drc', 'surface_curr_speed', 
    'air_temperature', 'water_temperature', 'air_pressure', 'humidity', 'salinity',
    'acceleration', 'heading_diff', 'distance_km', 'avg_speed_5steps', 'avg_heading_5steps', 'position_change_5steps', 'avg_speed_10steps', 'avg_heading_10steps', 'position_change_10steps', 'avg_speed_30steps', 'avg_heading_30steps', 'position_change_30steps'
]

# 선택한 열에서 음수 값 확인
negative_values = (data[columns_to_check] < 0).sum()

# 음수 값이 있는 열 필터링
negative_columns = negative_values[negative_values > 0]

# 결과 출력
if len(negative_columns) > 0:
    print("음수 값이 포함된 열과 그 개수:")
    print(negative_columns)
else:
    print("선택한 열에 음수 값이 없습니다.")


# In[39]:


from sklearn.preprocessing import MinMaxScaler, RobustScaler

# 스케일링 대상 열
scale_columns = [
    'turn', 'speed', 'accuracy', 'course', 'heading', 
    'wind_direct', 'wind_speed', 'surface_curr_drc', 'surface_curr_speed', 
    'air_temperature', 'water_temperature', 'air_pressure', 'humidity', 'salinity',
    'acceleration', 'heading_diff', 'distance_km', 'avg_speed_5steps', 'avg_heading_5steps', 'position_change_5steps', 'avg_speed_10steps', 'avg_heading_10steps', 'position_change_10steps', 'avg_speed_30steps', 'avg_heading_30steps', 'position_change_30steps'
]

# MinMaxScaler와 RobustScaler 초기화
scaler_minmax = MinMaxScaler()
scaler_robust = RobustScaler()

## 원본 데이터 유지
data_scaled = data.copy()

# RobustScaler 적용할 컬럼 지정
robust_columns = ['turn', 'acceleration', 'heading_diff']

# inf 값을 NaN으로 변환
data[robust_columns] = data[robust_columns].replace([np.inf, -np.inf], np.nan)

# NaN을 이전 값으로 채우고, 그래도 NaN이면 0으로 대체
for col in robust_columns:
    data[col].fillna(method='ffill', inplace=True)  # 이전 값으로 채우기
    data[col].fillna(method='bfill', inplace=True)  # 이후 값으로 채우기
    data[col].fillna(0, inplace=True)  # 최종적으로 0으로 대체

print("Infinite values handled successfully! 🚀")

# RobustScaler & MinMaxScaler 정의
scaler_robust = RobustScaler()
scaler_minmax = MinMaxScaler()

# RobustScaler 적용
data_scaled[robust_columns] = scaler_robust.fit_transform(data[robust_columns])

# 나머지 변수에는 MinMaxScaler 적용
columns_except_robust = [col for col in scale_columns if col not in robust_columns]
data_scaled[columns_except_robust] = scaler_minmax.fit_transform(data[columns_except_robust])
# 결과 저장
data_scaled.to_csv('../document/scaled_datalinear.csv', index=False, encoding='utf-8-sig')
print("스케일링된 데이터가 'scaled_data.csv'로 저장되었습니다.")


# In[37]:


print("Checking for infinite values...")
print(data[robust_columns].replace([np.inf, -np.inf], np.nan).isnull().sum())


# In[28]:


# 전처리 완료 데이터 저장
data.to_csv('../document/processed_data.csv', index=False, encoding='utf-8-sig')

print("전처리 완료된 데이터가 'processed_data.csv'로 저장되었습니다.")


# In[41]:


import pandas as pd

# CSV 파일 로드
df = pd.read_csv("../document/scaled_datalinear.csv")  # 파일 경로에 맞게 수정

# status가 NaN인 행 제거
df_cleaned = df.dropna(subset=["status"])

# 변경된 데이터 확인
print(df_cleaned.head())

# 정제된 데이터 저장 (선택)
df_cleaned.to_csv("../document/cleaned_datalinear.csv", index=False, encoding='utf-8-sig')


# In[49]:


import pandas as pd

def read_csv_features(csv_file):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # 모든 feature(컬럼) 출력
    print("CSV 파일에 포함된 feature 목록:")
    print(df.columns.tolist())

    # 데이터 샘플 출력
    print("\n데이터 샘플:")
    print(df.head())

    return df.columns.tolist(), df

# 사용 예시
csv_file_path = "../document/cleaned_datalinear.csv"  # 여기에 CSV 파일 경로를 입력하세요.
features, df = read_csv_features(csv_file_path)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic  # 두 좌표 사이 거리 계산

def calculate_haversine(lat1, lon1, lat2, lon2):
    """ 위도(lat)와 경도(lon)를 이용해 두 지점 사이의 거리를 계산 (단위: km) """
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
        return np.nan  # NaN이 있는 경우 거리를 NaN으로 설정
    return geodesic((lat1, lon1), (lat2, lon2)).km

# 데이터 불러오기
data = pd.read_csv("../document/cleaned_datalinear.csv")

# datetime 변환 및 시간 간격 계산
data["created_at"] = pd.to_datetime(data["created_at"])
data["time_diff"] = data["created_at"].diff().dt.total_seconds()  # 시간 간격 (초 단위)

# 이전 위도, 경도 추가 (이동 거리 계산용)
data["prev_lat"] = data["lat"].shift(1)
data["prev_lon"] = data["lon"].shift(1)

# 이동 거리 계산 (NaN 처리는 유지)
data["distance"] = data.apply(lambda row: calculate_haversine(row["prev_lat"], row["prev_lon"], row["lat"], row["lon"]), axis=1)

# 신호 소실 여부 탐지 (5분 = 300초 이상 데이터가 끊겼으면 신호 소실)
data["signal_loss"] = data["time_diff"] > 300

# 이상치 탐지 기준 (예: 5km 이상 이동하면 이상치)
threshold = 5  # km

# 🚨 신호 소실(5분) 후 첫 번째 점프는 이상치에서 제외
data["outlier"] = (data["distance"] > threshold) & (~data["signal_loss"].shift(1).fillna(False))

# plt.figure(figsize=(10, 6))

# # 정상 데이터 (파란색)
# plt.scatter(data[data["outlier"] == False]["lon"], 
#             data[data["outlier"] == False]["lat"], 
#             label="Normal Data", alpha=0.5, c="blue", s=10)

# # 이상치 (빨간색)
# plt.scatter(data[data["outlier"] == True]["lon"], 
#             data[data["outlier"] == True]["lat"], 
#             label="Outliers", alpha=0.8, c="red", s=30)

# plt.xlabel("Longitude (경도)")
# plt.ylabel("Latitude (위도)")
# plt.title("Scatter Plot of GPS Data with Outliers (5-min Signal Loss Considered)")
# plt.legend()
# plt.show()





# In[44]:


# 이상치 제거 (보간 없이 NaN 유지)
data_cleaned = data[data["outlier"] == False].drop(columns=["prev_lat", "prev_lon", "distance", "outlier", "signal_loss", "time_diff"])

# 전처리된 데이터 저장
data_cleaned.to_csv("../document/cleaned_data_no_outlierslinear.csv", index=False, encoding='utf-8-sig')

print(f"이상치 제거 전 데이터 크기: {len(data)}")
print(f"이상치 제거 후 데이터 크기: {len(data_cleaned)}")


# In[45]:


import numpy as np
import pandas as pd

# 데이터 로드
file_path = "../document/cleaned_data_no_outlierslinear.csv"  # 파일 경로를 올바르게 설정하세요
data = pd.read_csv(file_path)

# datetime 변환
data["created_at"] = pd.to_datetime(data["created_at"])

# 이전 위도, 경도 추가 (이전 좌표와 비교)
data["prev_lat"] = data["lat"].shift(1)
data["prev_lon"] = data["lon"].shift(1)

# 경도 및 위도의 변화량 계산
data["lat_diff"] = abs(data["lat"] - data["prev_lat"])
data["lon_diff"] = abs(data["lon"] - data["prev_lon"])

# 신호 소실 여부 탐지 (5분 = 300초 이상 데이터가 끊겼으면 신호 소실)
data["time_diff"] = data["created_at"].diff().dt.total_seconds()
data["signal_loss"] = data["time_diff"] > 300

# 신호 소실이 아닌 경우, 경도 또는 위도가 0.1 이상 차이나는 데이터 찾기 (이상치)
outliers = data[(data["signal_loss"] == False) & ((data["lat_diff"] > 0.1) | (data["lon_diff"] > 0.1))]

# 이상치를 제거한 데이터 생성
filtered_data = data.drop(outliers.index)

# 정렬 (MMSI 및 시간 순으로 정렬)
filtered_data = filtered_data.sort_values(by=["mmsi", "created_at"])

# 이상치 제거 후 데이터 저장
output_file = "../document/cleaned_data_without_lat_lon_outlierslinear.csv"
filtered_data.to_csv(output_file, index=False, encoding="utf-8-sig")

# 결과 출력
print(f"이상치 제거된 데이터 개수: {len(filtered_data)}")
print(f"이상치 제거 후 데이터가 '{output_file}'로 저장되었습니다.")


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
data = pd.read_csv('../document/processed_data.csv')

# 위도와 경도의 평균 및 표준편차 계산
lat_mean, lat_std = data['lat'].mean(), data['lat'].std()
lon_mean, lon_std = data['lon'].mean(), data['lon'].std()

# 이상치 기준: 평균 ± 2 * 표준편차
lat_lower, lat_upper = lat_mean - 2 * lat_std, lat_mean + 2 * lat_std
lon_lower, lon_upper = lon_mean - 2 * lon_std, lon_mean + 2 * lon_std

# 이상치 데이터 탐지
outliers = data[
    (data['lat'] < lat_lower) | (data['lat'] > lat_upper) |
    (data['lon'] < lon_lower) | (data['lon'] > lon_upper)
]

# 이상치 범위 출력
print("위도 이상치 범위:")
print(f"Lower: {lat_lower:.4f}, Upper: {lat_upper:.4f}")
print("경도 이상치 범위:")
print(f"Lower: {lon_lower:.4f}, Upper: {lon_upper:.4f}")

# 이상치와 정상 데이터 분포 시각화
plt.figure(figsize=(14, 6))

# 위도 분포
plt.subplot(1, 2, 1)
sns.histplot(data['lat'], kde=True, color='blue', label='Lat Data', bins=30)
plt.axvline(lat_lower, color='red', linestyle='--', label='Lat Lower Bound')
plt.axvline(lat_upper, color='green', linestyle='--', label='Lat Upper Bound')
plt.title('Latitude Distribution with Outliers')
plt.legend()

# 경도 분포
plt.subplot(1, 2, 2)
sns.histplot(data['lon'], kde=True, color='orange', label='Lon Data', bins=30)
plt.axvline(lon_lower, color='red', linestyle='--', label='Lon Lower Bound')
plt.axvline(lon_upper, color='green', linestyle='--', label='Lon Upper Bound')
plt.title('Longitude Distribution with Outliers')
plt.legend()

plt.tight_layout()
plt.show()

# 이상치의 위경도 산포도
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['lon'], y=data['lat'], label='Normal Data', alpha=0.6)
sns.scatterplot(x=outliers['lon'], y=outliers['lat'], color='red', label='Outliers')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Latitude and Longitude (Outliers Highlighted)')
plt.legend()
plt.show()

