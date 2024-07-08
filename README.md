# Boggart in Camera

# Work in Nvidia Jetson AGX Xavier
## Environment/Repository Setup Instructions
### Python Environment
```
cd boggart
sudo apt install python3.8
python3.8 -m venv env --system-site-packages
source env/bin/activate
pip install --upgrade pip
pip install tqdm munkres mongoengine
```
In `configs.py`, set `BOGGART_REPO_PATH` to the location of the Boggart repository.

### For mAP Evaluation
```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
cd ..
```
coco_evaluation돌릴때 다음과 같은 애러 뜨면 해결법:
```ImportError: cannot import name 'builder' from 'google.protobuf.internal'```
```
pip install --upgrade protobuf
pip show protobuf를 통해 설치된 경로를 찾아서 site-packages/google/protobuf/internal안에 있는 builder.py를 찾아서 따로 저장
pip install protobuf==3.19.5
builder.py를 다시 site-packages/google/protobuf/internal 저장
export LD_PRELOAD=/home/nvidia/boggart/env/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```
## Data Setup Instructions
### Video Data Setup

Videos (and data generated during execution) will be stored at `<BOGGART_REPO_PATH>/data/`. Make sure to set this path in `configs.py` (see the `main_dir` property).

Videos are expected to be split up by hour and then stored in ten-minute chunks. For example, the first ten minute chunk of hour 10 of the `auburn_first_angle` video dataset would be located at `<BOGGART_REPO_PATH>/data/auburn_first_angle/video/auburn_first_angle10_0.mp4`.

Example file structure for `data/`:
- boggart/
    - data/
        - auburn_first_angle10/
            - video/
                - auburn_first_angle10_0.mp4
                - auburn_first_angle10_1.mp4
                - auburn_first_angle10_2.mp4
                - auburn_first_angle10_3.mp4
                - auburn_first_angle10_4.mp4
                - auburn_first_angle10_5.mp4
        - auburn_first_angle11/
            - ...

Download video from youtube:
1080p, 30fps로 다운, audio도 같이 다운 후에 audio는 제거 필요
```
sudo docker run --rm -it -v $(pwd):/config jauderho/yt-dlp -f "bestvideo[height=1080][fps=30]+bestaudio/best[height=1080]" --merge-output-format mp4 -o "/config/auburn_first_angle.mp4" "https://www.youtube.com/watch?v=5WN2PJ_Qxjs"
```
split video to 10 munute chunks:
```
sudo docker run --rm -v $(pwd):/config linuxserver/ffmpeg -i /config/auburn_first_angle.mp4 -ss 00:10:00 -t 00:00:10 -c copy /config/auburn_first_angle_kyc1_0.mp4
```
video quality down:
using the x264 implementation of the H.264 codec.
```
sudo docker run --rm -v $(pwd):/config linuxserver/ffmpeg -i /config/auburn_first_angle.mp4 -ss 00:10:00 -t 01:00:10 -an -c:v libx264 -r 30 -crf 23 /config/auburn_first_angle_crf23_kyc1_0.mp4
```

frame 분할:
```
ffmpeg -i auburn_first_angle10_kyc1_0.mp4 -vf "fps=30" ./frame/frame_%04d.png
ffmpeg -i auburn_first_angle_kyc10_0.mp4 -ss 00:00:10 -t 00:00:05 -vf "fps=30" ./frame/000/frame_%04d.png

```

frame 인코딩:
```
ffmpeg -framerate 30 -i ./frame/300/frame_%04d.png -c:v libx264 -pix_fmt yuv420p ./output_300.mp4
```

bitrate 확인:
```
ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of default=noprint_wrappers=1:nokey=1 ./output_600.mp4
```

### Model Inference Data Setup

Boggart's current implementation requires that model results are already generated and saved into MongoDB. The repository contains a helper script to load model inference results into MongoDB. `load_inference_results_into_mongodb.py` requires that inference results are stored in per-hour chunks. For example, the inference results for running YOLOv3 (trained on the COCO dataset) for hour 10 of the `auburn_first_angle` video dataset should be located at `<BOGGART_REPO_PATH>/inference_results/yolo3-coco/auburn_first_angle/auburn_first_angle10.csv` .


To set up MongoDB, run:
```
sudo apt install -y mongodb
```
Add 'directoryperdb=True' to `/etc/mongodb.conf`.
```
sudo mongod --config /etc/mongodb.conf
```

Then, update `ml_model`, `video_name` and `hour` in  `load_detections_into_mongodb.py`. Running this script will then load that hour's worth of inference results into the database.

### Run Boggart
Instructions to execute Boggart's ahead-of-time and query-time processing can be found in `run.py`.

### 방식
사용하는 video 달라질 경우  
`Detection.py`의 `vid`, `hour`, `minutes` 변경  
`VideoData.py`의 `stored_dur` 변경  
`Experiment.py`의 `minute` 변경  
`ClusteringPipelineEngine.py`의 `total_frames_per_hour` 변경  
`load_detections_into_mongodb.py`의 `video_name`, `hour`, `range` 변경  
`run.py`의 `vid`, `hours` 변경  

`Boggart`
cluster -> centroid frame에 대해 query 진행 (`ClusteringPipelineEngine.py`의 161번째 줄)

`Reducto`
temporla filtering: frame 간의 featrue difference value 사용 => threshold 넘는 frame만 filtering
query type 별로 best feature, threshold value 존재 (static하게 결정)

### video 길이가 짧은 경우
- Clustering 문제
1분짜리 영상 사용할 때 chunk size도 1800으로 해서 한 번에 처리하는 경우 clustering index 에러 났었음
영상 길이에 따라 chunk size를 줄여 chunk가 여러개 나오도록 해야할듯

- filtering하지 않는 `baseline_accuracy.py` 결과가 제대로 나오지 않음 (frame 수 부족?)
