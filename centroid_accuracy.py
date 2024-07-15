from ModelProcessor import ModelProcessor
from VideoData import VideoData
from utils import (calculate_bbox_accuracy, parallelize_update_dictionary)
from create_mfs import (get_network_bytes, create_centroid_video)
import numpy as np
import pandas as pd
import time
import cv2
import os

query_class = 2
query_conf = 0.7
fps = 30

def get_gt(video_name, hour, model, query_segment_start, query_segment_size = 150, extract_frame = False):
    video_data = VideoData(video_name, hour)
    modelProcessor = ModelProcessor(model, video_data, query_class, query_conf, fps)

    if extract_frame:
        frame_generator = video_data.get_frames_by_bounds(query_segment_start, query_segment_start+query_segment_size)

        output_dir = f'./centroid_frame_png/{video_name}/{query_segment_start}'
        os.makedirs(output_dir, exist_ok=True)

        for idx, frame in enumerate(frame_generator):
            frame_filename = os.path.join(output_dir, f'frame_{idx+query_segment_start:04d}.png')
            cv2.imwrite(frame_filename, frame)

    gt_bboxes, _ = modelProcessor.get_ground_truth(query_segment_start, query_segment_start+query_segment_size)

    return gt_bboxes

def accuracy(chunk_start):
    # print(f"chunk start {chunk_start}")
    scores = []
    rates = []
    gt_bboxes = get_gt("auburn_ss2_kyc", 60, "yolov5", chunk_start, extract_frame=False)
    det_bboxes = get_gt("auburn_ss2_crf23_kyc", 60, "yolov5", chunk_start)

    # video_bitrate = create_centroid_video("auburn_ss2_kyc", 60, chunk_start)

    for bbox_gt, sr in zip(gt_bboxes, det_bboxes):
        scores.append(calculate_bbox_accuracy(bbox_gt, sr))

    # rates.append(video_bitrate)

    # return {"scores": scores}, {"rates": rates}
    return {"scores": scores}

total_scores = []
total_rates = []
vid_name = "auburn_ss2_kyc"
centroid_reslut = f'./centroid_result/centroids_{vid_name}.csv'
df = pd.read_csv(centroid_reslut)
seg_start_values = df['seg_start']
seg_start_list = seg_start_values.tolist()
print(seg_start_list)
start_bytes = get_network_bytes()
start_time = time.time()

scores_dict = parallelize_update_dictionary(accuracy, seg_start_list, max_workers=1, total_cpus=4)

end_bytes = get_network_bytes()
end_time = time.time()

bitrate = ((end_bytes - start_bytes))

# for ts, (score,rate) in scores_dict.items():
#      total_scores.extend(score["scores"])
    #  total_rates.extend(rate["rates"])

for ts, score in scores_dict.items():
     total_scores.extend(score["scores"])

# print(f"score: {round(np.mean(np.array(total_scores)), 4)}, bitrate: {round(np.mean(np.array(total_rates)), 4)}")
print(f"score: {round(np.mean(np.array(total_scores)), 4)}")
