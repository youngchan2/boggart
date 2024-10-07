# %%
from ClusteringPipelineEngine import ClusteringPipelineEngine
from Experiment import Experiment
from VideoData import VideoData
from create_mfs import create_mfs_video
import cv2
import pandas as pd
import csv

# vid = "lausanne_pont_bassieres"
vid = "auburn_ss16_10_kyc"
hours = list(range(10, 11))
chunk_size = 150
query_seg_size = 150

### AHEAD-OF-TIME PROCESSING ###
# This is done once per video
# for hr in hours:
#     VideoData(vid, hr).check_vids()
#     Experiment(vid = vid, hour = hr, chunk_size = chunk_size, query_seg_size = query_seg_size).run_ingest()

for hr in hours:
    Experiment(vid = vid, hour = hr, chunk_size = chunk_size, query_seg_size = query_seg_size).run_ingest()

# QUERY-TIME PROCESSING ###
# This is done once per query

query_class = "car"
model = "yolov5"
acc_target = 0.9
query_conf = 0.7
qtype = "bbox"
pc = 0.1

# # convert query_class name to the corresponding index
qclass_label = {"car" : 2, "person": 0}[query_class] # coco

cpe = ClusteringPipelineEngine(vid, query_conf=query_conf)
results_df = cpe.execute(hours, qtype, model, qclass_label, acc_target, percent_clusters=pc, ioda=0.1, get_boggart_results=True)
print(results_df)

#################preset optio########################################################
# ultrafast: 가장 빠른 인코딩 속도. 파일 크기는 가장 크며, 압축 효율성이 가장 낮습니다.
# superfast: 매우 빠른 인코딩 속도.
# veryfast: 빠른 인코딩 속도.
# faster: 상대적으로 빠른 인코딩 속도.
# fast: 기본적으로 빠른 인코딩 속도.
# medium: 속도와 압축 효율성의 균형. 기본값입니다.
# slow: 느린 인코딩 속도. 더 좋은 압축 효율성.
# slower: 더 느린 인코딩 속도.
# veryslow: 가장 느린 인코딩 속도. 가장 좋은 압축 효율성.
######################################################################################
# presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
# preset_bitrate = []
# qp_preset_bitrate = []

# for preset in presets:
#     bitrate, time = create_mfs_video(vid, hours[0], fps=30, preset=preset)
#     qp_bitrate, qp_time = create_mfs_video(vid, hours[0], fps=30, preset=preset, qp=36)
#     preset_bitrate.append({'preset': preset, 'bitrate': bitrate, 'time': time})
#     qp_preset_bitrate.append({'preset': preset, 'bitrate': qp_bitrate, 'time': qp_time})

# bitrate_path = f'./bitrate/bitrate_{vid}{hours[0]}.csv'
# qp_bitrate_path = f'./bitrate/btirate_qp_{vid}{hours[0]}.csv'

# df = pd.DataFrame(preset_bitrate)
# df = df.sort_values(by='bitrate', ascending=False)
# qp_df = pd.DataFrame(qp_preset_bitrate)
# qp_df = qp_df.sort_values(by='bitrate', ascending=False)

# df.to_csv(bitrate_path, index=False)
# qp_df.to_csv(qp_bitrate_path, index=False)

# results_df.to_csv(f"results_{hours[0]}.csv", index=False)

# df = pd.read_csv('results_10.csv')
# cap = cv2.VideoCapture('/data/auburn_first_angle_kyc10/video/auburn_first_angle_kyc10_0.mp4')
# # fps = cap.get(cv2.CAP_PROP_FPS)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output_video_path = 'output_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break
#     frame_boxes = df[df['frame_no'] == cap.get(cv2.CAP_PROP_POS_FRAMES)-1]
    
#     for _, row in frame_boxes.iterrows():
#         x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     out.write(frame)

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# Results of boggart are located in results_df

# If bounding box query,
#   columns: hour, frame_no, x1, y1, x2, y2
# If count query,
#   columns: hour, frame_no, count
# If binary query,
#   columns: hour, frame_no, found