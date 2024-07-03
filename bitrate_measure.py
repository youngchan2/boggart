from VideoData import VideoData
import cv2
import os
import subprocess
import psutil
import time

def create_centroid_video(video_name, hour, query_segment_start, query_segment_size = 150, fps = 30):
    vd = VideoData(video_name, hour)

    # frame_generator = vd.get_frames_by_bounds(query_segment_start, query_segment_start+query_segment_size)

    frame_path = f'./centroid_frame_png/{video_name}/{query_segment_start}'
    output_video_dir = f'./centroid_video/{video_name}'
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"centroid_{video_name}{query_segment_start}.mp4")
                 
    command = ['ffmpeg', 
               '-framerate', str(fps), 
               '-start_number', str(query_segment_start),
               '-i', os.path.join(frame_path, f'frame_%04d.png'), 
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
               output_video_path
               ]

    subprocess.run(command, check=True)
    
    bitrate = get_video_bitrate(output_video_path)
    return bitrate

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

def get_network_bytes():
    net_io = psutil.net_io_counters()
    return net_io.bytes_sent + net_io.bytes_recv

vd = VideoData("auburn_first_angle_kyc", 10)
frame_generator = vd.get_frames_by_bounds(0, 1800)

output_dir = f'./frame/'
os.makedirs(output_dir, exist_ok=True)

for idx, frame in enumerate(frame_generator):
    frame_filename = os.path.join(output_dir, f'frame_{idx:04d}.png')
    cv2.imwrite(frame_filename, frame)