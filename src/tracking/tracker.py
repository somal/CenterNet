from typing import Dict, List

import numpy as np

from src.tracking.sort import Sort


class Tracker:
    def __init__(self, nonactive_track_max_times: int = 15, min_hits_to_track: int = 3, iou_threshold: float = .3):
        self._tracker = Sort(max_age=nonactive_track_max_times,
                             min_hits=min_hits_to_track,
                             iou_threshold=iou_threshold)

    def update(self, detection_boxes: List[List[float]]) -> Dict[int, np.array]:
        detections_for_trackers = []
        for det in detection_boxes:
            new_det = []
            for d in det[:4]:
                new_det.append(max(min(d, 1), 0))
            new_det.append(max(min(det[4], 1), 0))
            detections_for_trackers.append(new_det)
        detections_for_trackers = np.array(detections_for_trackers)

        if detections_for_trackers.shape[0] == 0:
            detections_for_trackers = np.empty((0, 5))
        tracker_result = self._tracker.update(detections_for_trackers)
        return {int(t[4]): t[:4] for t in tracker_result}
