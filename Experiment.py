from itertools import product

from configs import BackgroundConfig, TrajectoryConfig
from ingest import IngestTimeProcessing
from QueryProcessor import QueryProcessor
from utils import parallelize_update_dictionary
from VideoData import VideoData
from ModelProcessor import ModelProcessor

class Experiment:

    def __init__(self, vid, hour, additional_sweeps=None, skips_qs_starts=[], skip_no_traj=False, chunk_size=180, query_seg_size=180):
        self.video_data = VideoData(
            db_vid = vid,
            hour = hour
        )

        minutes = list(range(0, 1800, chunk_size))
        self.minutes = minutes

        param_sweeps = {
            "diff_thresh" : [16],
            "peak_thresh": [0.1],
            "fps": [30]
        }

        self.chunk_size = chunk_size
        self.query_seg_size = query_seg_size

        if additional_sweeps is not None:
            for k, v in additional_sweeps.items():
                param_sweeps[k] = v

        self.sweep_param_keys = list(param_sweeps.keys())[::-1]
        _combos = list(product(*[param_sweeps[k] for k in self.sweep_param_keys]))
        
        segment_combos = []
        for minute in minutes:
            chunk_starts = list(range(minute, minute+1800, self.chunk_size))
            segment_combos.append(chunk_starts)
        self.ingest_combos = list(product(_combos, segment_combos))

        segment_combos = []
        for minute in minutes:
            chunk_starts = list(range(minute, minute+1800, self.chunk_size))
            for chunk_start in chunk_starts:
                query_seg_starts = list(range(chunk_start, chunk_start+self.chunk_size, self.query_seg_size))
                segment_combos.extend(list(product([chunk_start], query_seg_starts)))
        self.query_combos = list(product(_combos, segment_combos))

        self.skips_qs_starts = skips_qs_starts

        self.skip_no_traj = skip_no_traj

    def run_single_default(self, chunk_start):
        bg_conf = BackgroundConfig(peak_thresh=0.1, bg_dur = self.chunk_size)
        traj_conf = TrajectoryConfig(diff_thresh=16, chunk_size=self.chunk_size, fps=30)
        e = IngestTimeProcessing(self.video_data, bg_conf, traj_conf)
        bg_start = e.bg_config.get_bg_start(chunk_start)
        e.generate_background(bg_start)
        e.run_tracker(chunk_start, True)

    def run_ingest(self):
        try:
            for start in self.minutes:
                self.run_single_default(start)
            return 1
        except Exception as e:
            print("FAILED AT ", e)