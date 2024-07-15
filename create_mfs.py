from VideoData import VideoData
import cv2
import os
import subprocess
import psutil
import time
import pandas as pd

def create_mfs_video(video_name, hour, fps=30, preset='veryslow', qp = None):
    mfs_result_path = f'./mfs_result/{video_name}{hour}.csv'
    min_frame_dir = f'./min_frame/{video_name}'
    output_video_dir = f'./mfs_video/{video_name}/'
    os.makedirs(output_video_dir, exist_ok=True)

    df = pd.read_csv(mfs_result_path, header=None)
    frame_num = df[0].tolist()

    if not frame_num:
        raise ValueError("No frame number")
    if qp:
        output_video_path = os.path.join(output_video_dir, f'mfs_{video_name}_qp{qp}_{preset}.mp4')
        command = ['ffmpeg',
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', os.path.join(min_frame_dir, 'frame_*.png'),
                '-qp', str(qp),
                '-c:v', 'libx264',
                '-preset', preset,
                '-pix_fmt', 'yuv420p',
                output_video_path]
    else:
        output_video_path = os.path.join(output_video_dir, f'mfs_{video_name}_{preset}.mp4')
        command = ['ffmpeg',
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', os.path.join(min_frame_dir, 'frame_*.png'),
                '-c:v', 'libx264',
                '-preset', preset,
                '-pix_fmt', 'yuv420p',
                output_video_path]
    
    start_time = time.time()
    subprocess.run(command, check=True)
    end_time = time.time()
    
    encoding_time = end_time - start_time

    bitrate = get_video_bitrate(output_video_path)

    return bitrate, encoding_time

def get_video_bitrate(video_path):
    command = ['ffprobe',
               '-v', 'error',
               '-select_streams', 'v:0',
               '-show_entries', 'stream=bit_rate',
               '-of', 'default=noprint_wrappers=1:nokey=1',
               video_path
               ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    bitrate = int(result.stdout.strip())
    return bitrate

# vid = "auburn_first_angle_kyc"
# hours = 10

# mfs_video_path = create_mfs_video(vid, hours)
# bitrate = get_video_bitrate(mfs_video_path)
# kbyterate = (bitrate/8)/1024
# print(f'bitrate {vid}:{kbyterate}')

# def create_centroid_video(video_name, hour, query_segment_start, query_segment_size = 150, fps = 30):
#     vd = VideoData(video_name, hour)

#     # frame_generator = vd.get_frames_by_bounds(query_segment_start, query_segment_start+query_segment_size)

#     frame_path = f'./centroid_frame_png/{video_name}/{query_segment_start}'
#     output_video_dir = f'./centroid_video/{video_name}'
#     os.makedirs(output_video_dir, exist_ok=True)
#     output_video_path = os.path.join(output_video_dir, f"centroid_{video_name}{query_segment_start}.mp4")
                 
#     command = ['ffmpeg', 
#                '-framerate', str(fps), 
#                '-start_number', str(query_segment_start),
#                '-i', os.path.join(frame_path, 'frame_%04d.png'), 
#                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
#                output_video_path
#                ]

#     subprocess.run(command, check=True)
    
#     bitrate = get_video_bitrate(output_video_path)
#     return bitrate

# def get_network_bytes():
#     net_io = psutil.net_io_counters()
#     return net_io.bytes_sent + net_io.bytes_recv

# vd = VideoData("auburn_first_angle_kyc", 10)
# frame_generator = vd.get_frames_by_bounds(0, 1800)

# output_dir = f'./frame/'
# os.makedirs(output_dir, exist_ok=True)

# for idx, frame in enumerate(frame_generator):
#     frame_filename = os.path.join(output_dir, f'frame_{idx:04d}.png')
#     cv2.imwrite(frame_filename, frame)