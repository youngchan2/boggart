import os
import pandas as pd
import json
from mongoengine import connect, disconnect
from ModelProcessor import ModelProcessor
from VideoData import VideoData
from utils import (
    calculate_bbox_accuracy,
    calculate_count_accuracy,
    get_ioda_matrix,
    calculate_binary_accuracy,
    parallelize_update_dictionary,
)
from bitrate_measure import (get_network_bytes, get_video_bitrate)
import numpy as np
import time
import cv2
import subprocess
import shutil

# from evaluator import Evaluator

query_class = 2
query_conf = 0.7
fps = 30


def get_gt(video_name, hour, model, query_segment_start, query_segment_size=150):
    video_data = VideoData(video_name, hour)
    modelProcessor = ModelProcessor(model, video_data, query_class, query_conf, fps)

    gt_bboxes, gt_counts = modelProcessor.get_ground_truth(
        query_segment_start, query_segment_start + query_segment_size
    )

    # disconnect(alias='my')
    # print(gt_bboxes)
    return gt_bboxes


def accuracy(chunk_start):
    # print(f"chunk start {chunk_start}")
    scores = []
    rates = []
    gt_bboxes = get_gt("auburn_first_angle_kyc", 10, "yolov5", chunk_start)

    det_bboxes = get_gt("auburn_first_angle60_crf23_kyc", 60, "yolov5", chunk_start)
    # print(f"gt:{gt_bboxes}")
    # print(f"dt:{det_bboxes}")
    video_bitrate = encoding("auburn_first_angle_kyc", 10, chunk_start)

    # bounding box accuracy
    for bbox_gt, sr in zip(gt_bboxes, det_bboxes):
        scores.append(calculate_bbox_accuracy(bbox_gt, sr))
        # print(scores)
    rates.append(video_bitrate)

    return {"scores": scores}, {"rates": rates}

def encoding(video_name, hour, query_start, query_size = 150, fps = 30):
    vd = VideoData(video_name, hour)
    frame_generator = vd.get_frames_by_bounds(query_start, query_start+query_size)

    output_video_dir = f'./baseline_encoding/'
    os.makedirs(output_video_dir, exist_ok=True)

    output_video_path = os.path.join(output_video_dir, f"baseline_{video_name}{query_start}.mp4")

    temp_dir = './temp_frames/'
    os.makedirs(temp_dir, exist_ok=True)

    for idx, frame in enumerate(frame_generator):
        frame_file = os.path.join(temp_dir, f'frame_{idx:04d}.png')
        cv2.imwrite(frame_file, frame)

    # command = ['ffmpeg',
    #            '-framerate', str(fps),
    #            '-i', frame_file,
    #            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    #            output_video_path
    #            ]
    
    # subprocess.run(command, check=True)
    # shutil.rmtree(temp_dir)

    bitrate = get_video_bitrate(output_video_path)
    return bitrate


total_scores = []
total_rates = []
# 모든 frame에 대해 query 진행
start_bytes = get_network_bytes()
start_time = time.time()

scores_dict = parallelize_update_dictionary(accuracy, range(0, 1800, 150), max_workers=1, total_cpus=4)

end_bytes = get_network_bytes()
end_time = time.time()

bitrate = ((end_bytes - start_bytes))
for ts, (score, rate) in scores_dict.items():
    total_scores.extend(score["scores"])
    total_rates.extend(rate["rates"])

print(f"score: {round(np.mean(np.array(total_scores)), 4)}, bitrate: {round(np.mean(np.array(total_rates)), 4)}")


# gt_bboxess = get_gt("auburn_first_angle", 10, "yolov5", 0)
# query_results = get_gt("auburn_first_angle", 10, "yolov5", 0)
# for model_a_dets, model_b_dets in zip(gt_bboxess, query_results):

#     # check both empty
#     if len(model_a_dets) == len(model_b_dets) == 0:
#         print(1)
#     # check one is empty
#     if len(model_a_dets) == 0 or len(model_b_dets) == 0:
#         print(0)

#     if len(model_a_dets) == 0:
#         model_a_dets = np.empty(shape=[0, 4], dtype=np.float32)
#     if len(model_b_dets) == 0:
#         model_b_dets = np.empty(shape=[0, 4], dtype=np.float32)
#     # print(model_b_dets)
#     # for det in model_b_dets:
#     #     print(len(det), det)
#     det_dict = {
#         'detection_boxes': np.array(model_b_dets, dtype=np.float32),
#         'detection_scores': np.array([1 for _ in range(len(model_b_dets))], dtype=np.float32),
#         'detection_classes': np.array([0 for _ in range(len(model_b_dets))], dtype=np.uint8)
#     }
#     # print(det_dict)
#     gt_dict = {
#         "groundtruth_boxes" : np.array(model_a_dets, dtype=np.float32),
#         "groundtruth_classes" : np.array([0 for _ in range(len(model_a_dets))], dtype=np.uint8)
#     }
#     # x1,y1,x2,y2 -> x1,y1,w,h
#     det_dict['detection_boxes'] = np.hstack((det_dict['detection_boxes'][:, 0:2],
#                                             (det_dict['detection_boxes'][:, 2] - det_dict['detection_boxes'][:, 0])[:,np.newaxis],
#                                             (det_dict['detection_boxes'][:, 3] - det_dict['detection_boxes'][:, 1])[:,np.newaxis]))

#     gt_dict['groundtruth_boxes'] = np.hstack((gt_dict['groundtruth_boxes'][:, 0:2],
#                                             (gt_dict['groundtruth_boxes'][:, 2] - gt_dict['groundtruth_boxes'][:, 0])[:,np.newaxis],
#                                             (gt_dict['groundtruth_boxes'][:, 3] - gt_dict['groundtruth_boxes'][:, 1])[:,np.newaxis]))

#     det_combined = np.hstack((det_dict['detection_boxes'], det_dict['detection_scores'][:, np.newaxis], det_dict['detection_classes'][:, np.newaxis]))
#     gt_combined = np.hstack((gt_dict['groundtruth_boxes'], gt_dict['groundtruth_classes'][:, np.newaxis]))

#     # print(det_combined)

#     coco_eval = Evaluator()
#     coco_eval.add(det_combined, gt_combined)
#     coco_eval.accumulate()
#     a = coco_eval.summarize()
#     if a == -1:
#         print(gt_combined)
#         print("sdf")
#         print(det_combined)
