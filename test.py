import cv2
import os
import subprocess
import pandas as pd

def get_frames_by_bounds(self, start, stop, skip=1):
        cap = cv2.VideoCapture(self.vname_chunked(0))
        # if not cap.isOpened():
        #     raise NoMoreVideo

        # 총 프레임 수
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # if start >= total_frames:
        #     raise NoMoreVideo
        # elif start < 0 or stop > total_frames or start > stop:
        #     print("Error: Invalid frame range.")
        #     return

        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        current_frame = start
        while current_frame < stop:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            current_frame += 1

        cap.release()

def create_mfs_video(video_name, hour, fps=30, preset='veryfast'):
    mfs_result_path = f'./mfs_result/{video_name}/{video_name}{hour}.csv'
    min_frame_dir = f'./min_frame/{video_name}'
    output_video_dir = f'./mfs_video/{video_name}'
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f'mfs_{video_name}.mp4')

    df = pd.read_csv(mfs_result_path, header=None)
    frame_num = df[0].tolist()

    if not frame_num:
        raise ValueError("No frame number")

    command = ['ffmpeg',
               '-framerate', str(fps),
               '-pattern_type', 'glob',
               '-i', os.path.join(min_frame_dir, 'frame_*.png'),
               '-c:v', 'libx264',
               '-preset', preset,
               '-pix_fmt', 'yuv420p',
               output_video_path]
    
    subprocess.run(command, check=True)

    return output_video_path

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

vid = "auburn_first_angle_kyc"
hours = 10

mfs_video_path = create_mfs_video(vid, hours)
bitrate = get_video_bitrate(mfs_video_path)
kbyterate = (bitrate/8)/1024
print(f'bitrate {vid}:{kbyterate}')