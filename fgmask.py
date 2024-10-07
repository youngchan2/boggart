import os
import re
import pandas as pd
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from configs import BOGGART_REPO_PATH
from mongoengine import connect, DoesNotExist
from db_model import DetectionResult, Frame

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load("ultralytics/yolov5", "yolov5n").to(device)

video = "auburn_ss16_10_kyc"
ml_model = "yolov5"
min_frame_dir = f"{BOGGART_REPO_PATH}/min_frame/{video}/"
fg_dir = f"{BOGGART_REPO_PATH}/fg/{video}/"
filtered_dir = f"{BOGGART_REPO_PATH}/filtered/{video}/"
detect_dir = f"{BOGGART_REPO_PATH}/detected/{video}/"
new_csv_location = f"{BOGGART_REPO_PATH}/new_gt/{video}/gt_{video}10.csv"
ori_csv_location = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video}/{video}10.csv"

os.makedirs(fg_dir, exist_ok=True)
os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(detect_dir, exist_ok=True)

key_list = []

for filename in os.listdir(min_frame_dir):
    if re.match(r'frame_\d{4}\.png$', filename):
        num = filename.split('_')[1].split('.')[0]
        key_list.append(num)
key_list.sort(key=int)

def filter_frame_with_foreground_mask(frame_list):
    for frame in frame_list:
    # for frame in range(300):
        # original_frame_path = f"{BOGGART_REPO_PATH}/original/{video}/frame_{frame:04d}.png"
        # foreground_mask_path = f"{fg_dir}fg_{frame:04d}.png"        # 이미 흑백인 foreground 마스크 경로
        # filtered_path = f"{filtered_dir}filtered_frame_{frame:04d}.png"          # 결과 파일 경로
        original_frame_path = f"{min_frame_dir}frame_{frame}.png"  # 원래 프레임 경로
        foreground_mask_path = f"{fg_dir}fg_{frame}.png"        # 이미 흑백인 foreground 마스크 경로
        filtered_path = f"{filtered_dir}filtered_frame_{frame}.png"          # 결과 파일 경로
        
        original_frame = cv2.imread(original_frame_path)
        h, w, _ = original_frame.shape

        foreground_mask = cv2.imread(foreground_mask_path, cv2.IMREAD_GRAYSCALE)
        foreground_mask_resized = cv2.resize(foreground_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if foreground_mask_resized.dtype != np.uint8:
            foreground_mask_resized = foreground_mask_resized.astype(np.uint8)

        filtered = cv2.bitwise_and(original_frame, original_frame, mask=foreground_mask_resized)

        cv2.imwrite(filtered_path, filtered)

def run_model(frame_list):
    
    os.makedirs(os.path.dirname(new_csv_location), exist_ok=True)
    if not os.path.exists(new_csv_location):
        pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(new_csv_location, index=False)

    data_frame = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])
    for frame in frame_list:
    # for frame in range(300):
        # original_frame_path = f"{BOGGART_REPO_PATH}/original/{video}/frame_{frame:04d}.png"
        original_frame_path = f"{min_frame_dir}frame_{frame}.png"  # 원래 프레임 경로
        filtered_path = f"{filtered_dir}filtered_frame_{frame}.png"          # 결과 파일 경로
        output_path = f"{detect_dir}filtered_detect_{frame}.png"
        
        img = cv2.imread(filtered_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori = cv2.imread(original_frame_path)
        img_ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)

        result = model(img_rgb)
        result_ori = model(img_ori)

        result_data = result.xyxy[0].cpu().numpy()
        ori_data = result_ori.xyxy[0].cpu().numpy()
        
        # 필터링 frame을 이용해 object detection 결과 확인
        for *box, conf, cls in result_data:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0),2)

            cv2.imwrite(output_path, img)
        # 필터링 결과를 확인하기 위해 original frame의 결과를 표시
        for *box, conf, cls in ori_data:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255),2)

            cv2.imwrite(output_path, img)
        # gt를 db에 올리기 위해 data frame 생성
        frame_result = result.pandas().xyxy[0]
        frame_result = frame_result.rename(columns={"xmin" : "x1",
                                                    "ymin" : "y1",
                                                    "xmax" : "x2",
                                                    "ymax" : "y2",
                                                    "confidence": "conf",
                                                    "name" : "label",
                                                    })
        frame_result["frame"] = int(frame)
        frame_result["label"] = frame_result["label"].replace("truck", "car")
        frame_result = frame_result[["frame", "x1", "y1", "x2", "y2", "label", "conf"]]
        data_frame = pd.concat([data_frame, frame_result], ignore_index=True)
    data_frame.to_csv(new_csv_location, mode="a", header=False, index=False)

def load_data(key_list):
    db = connect(db=f"gt_{video}",
                 username='root',
                 password='root',
                 host='mango4.kaist.ac.kr',
                 authentication_source='admin',
                 port=27017,
                 maxPoolSize=10000)
    
    df = pd.read_csv(new_csv_location, skiprows=1, names=["frame", "x1", "y1", "x2", "y2", "label", "conf"], dtype=str)
    df['frame'] = df['frame'].astype(float).astype(int)
    df['conf'] = df['conf'].astype(float)

    # for i in key_list:
    for f in range(300):
        # f = int(i)
        frame = None
        try:
            frame = Frame.objects.get(frame_no = f, hour=10)
            if ml_model in frame.inferenceResults:
                continue
        except DoesNotExist:
            pass

        curr_data = df[df['frame']==f]

        # save det
        det = DetectionResult()
        det.model = ml_model
        det.detection_boxes = curr_data[['x1', 'y1', 'x2', 'y2']].values.tolist()
        det.detection_classes = curr_data['label'].values.tolist()
        det.detection_scores = curr_data['conf'].round(3).values.tolist()
        det.num_detections = len(det.detection_boxes)
        det.save()

        # save frame
        if not frame:
            frame = Frame(frame_no=f, hour=10)
        frame.inferenceResults[ml_model] = det
        frame.save()



filter_frame_with_foreground_mask(key_list)
run_model(key_list)
# load_data(key_list)