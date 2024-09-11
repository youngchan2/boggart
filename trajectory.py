import cv2
import os
import pandas as pd
from configs import main_dir
video = "auburn_ss21_30_kyc10"
# CSV 파일 읽기
traj_dir = f"{main_dir}/{video}/trajectories/"
# df = pd.read_csv(traj_data)

# 동영상 파일 또는 프레임이 저장된 경로 설정
video_path = f'{main_dir}/{video}/video/{video}_0.mp4'
output_path = f'{main_dir}/{video}/{video}_traj.mp4'

# 동영상 읽기
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 프레임 속도
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 비디오 쓰기
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수

# 10초 비디오면 0~150 / 150~300에만 trajectory 정보 있으므로 앞 2개 csv파일만 확인
csv_files = [f for f in os.listdir(traj_dir) if f.split('_')[0] in ['0', '150']]

# 비디오를 한 프레임씩 처리
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오 끝에 도달하면 루프 종료

    # 현재 프레임 번호에 해당하는 TS 값 확인
    # current_ts_data = df[df['TS'] == frame_idx]
    current_ts_data = pd.DataFrame()
    for csv_file in csv_files:
        traj_path = os.path.join(traj_dir, csv_file)
        df = pd.read_csv(traj_path)
        current_ts_data = pd.concat([current_ts_data, df[df['TS']==frame_idx]])

    # 데이터가 있으면 그 프레임에 맞는 박스를 그림
    for index, row in current_ts_data.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        state = row['bstate']

        # 박스 그리기 (초록색, 두께 2)
        # color = (0, 255, 0) if state == "ObjectState.OBJECT" else (0, 0, 255)
        if state == "ObjectState.OBJECT":
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 상태 텍스트 추가
            cv2.putText(frame, str(row['ObjId']), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 프레임을 출력 파일에 쓰기
    out.write(frame)

    frame_idx += 1
    if frame_idx >= frame_count:  # 총 프레임 수를 초과하면 종료
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
