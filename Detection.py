from itertools import product
from VideoData import VideoData, NoMoreVideo
from tqdm import tqdm, trange
from configs import BOGGART_REPO_PATH
import numpy as np
import pandas as pd
import torch
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(device)

ml_model = "yolov5"
video_name = "auburn_first_angle_kyc"
hour = 10
csv_location = f'{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}/{video_name}{hour}.csv'

def execute(ingest_combos, vid, chunk_size):
    os.makedirs(os.path.dirname(csv_location), exist_ok=True)
    if not os.path.exists(csv_location):
        pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(csv_location, index=False)

    for i in range(len(ingest_combos)):
        data_frame = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])
        params = ingest_combos[i]
        chunk_start = params[1][0]
        frame_generator = vid.get_frames_by_bounds(chunk_start, chunk_start+chunk_size, int(1))
        for i in trange(chunk_start, chunk_start+chunk_size, int(1), leave=False, desc=f"{chunk_start}_{chunk_size}"):
            f = next(frame_generator)
            if f is None:
                print(f"skipping frame {i}")
                continue

            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

            h,w, = f.shape
            h /= 2
            w /= 2

            f = cv2.resize(f, (int(w),int(h)))

            results = model(f)
            frame_result = results.pandas().xyxy[0]
            frame_result = frame_result.rename(columns={"xmin":"x1", "ymin":"y1", "xmax":"x2", "ymax":"y2", "confidence":"conf", "name":"label"})
            frame_result['frame'] = i
            frame_result['label'] = frame_result['label'].replace('truck', 'car')
            frame_result = frame_result[['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'conf']]
            data_frame = pd.concat([data_frame, frame_result], ignore_index=True)
        data_frame.to_csv(csv_location, mode='a', header=False, index=False)

if __name__=="__main__":
    vid = "auburn_first_angle_kyc"
    hour = 10
    chunk_size = 180
    query_seg_size = 180

    video = VideoData(vid, hour)
    minutes = list(range(0, 1800, 180))
    param_sweeps = {
        "diff_thresh" : [16],
        "peak_thresh" : [0.1],
        "fps": [30]
    }

    sweep_param_keys = list(param_sweeps.keys())[::-1]
    _combos = list(product(*[param_sweeps[k] for k in sweep_param_keys]))

    segment_combos = []
    for minute in minutes:
        chunk_starts = list(range(minute, minute+180, chunk_size))
        segment_combos.append(chunk_starts)
    ingest_combos = list(product(_combos, segment_combos))
    execute(ingest_combos, video, chunk_size)