import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.lib.detectors.ctdet import CtdetDetector
from src.lib.opts import Opts
from src.tracking.tracker import Tracker


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    def __iter__(self):
        for p in (self.x1, self.y1, self.x2, self.y2):
            yield p


class Polygon:
    def __init__(self, points: List[List[int]]):
        self._pts = points

    def is_point_inside(self, x: int, y: int) -> bool:
        return cv2.pointPolygonTest(contour=np.array(self._pts), pt=(x, y), measureDist=False) >= 0

    @staticmethod
    def read_polygons_from_json(video_path: str):
        video_name = video_path.split('/')[-1]
        polygons = json.load(open(f'{video_name}.json', 'r'))
        return [Polygon(points) for points in polygons]

    @staticmethod
    def draw_polygons_on_image(polygons: List, image: np.array):
        for polygon in polygons:
            cv2.polylines(image, pts=[np.array(polygon._pts).reshape((-1, 1, 2))], isClosed=True, color=(255, 100, 0))

    def __repr__(self):
        return repr(self._pts)


def gaps_from_inter_times(times: List[float]) -> np.array:
    y = [times[0]] if len(times) > 0 else []
    for j in range(1, len(times)):
        y.append((times[j] - times[j - 1]))
    return np.array(y)


def create_video_writer(fps: int, img_size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter('output.mp4', fourcc, fps, img_size)


def convert_real_to_norm_bbox_coords(bbox: List[float], img_size: Tuple[int, int]) -> List[float]:
    """
    :param bbox: coordinates in range [0, w/h] in (x1,y1,x2,y2, score) format
    :param img_size: (w,h)
    :return List[float]: coordinates in range [0, 1] in (x1,y1,x2,y2, score) format
    """
    assert len(bbox) == 5
    return [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1], bbox[4]]


def convert_norm_to_real_bbox_coords(bbox: List[float], img_size: Tuple[int, int]) -> List[int]:
    """
    :param bbox: coordinates in range [0, 1] in (x1,y1,x2,y2, score) format
    :param img_size: (w,h)
    :return List[float]: coordinates in range [0, w/h] in (x1,y1,x2,y2, score) format
    """
    assert len(bbox) == 4
    return [int(bbox[0] * img_size[0]), int(bbox[1] * img_size[1]),
            int(bbox[2] * img_size[0]), int(bbox[3] * img_size[1])]


def handle_times_gap(polygons: List[Polygon], inter_fn_per_polygon: List[List[float]],
                     img_res: np.ndarray, fps: float):
    for i in range(len(polygons)):
        gaps = gaps_from_inter_times(inter_fn_per_polygon[i])
        frame_diff = gaps[-1] / fps if len(gaps) > 0 else 0
        cv2.putText(img_res, f'Gap time {i} {frame_diff:.2f}', (30, 50 * (i + 1)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
    cv2.putText(img_res, f'Count L {len(inter_fn_per_polygon[0])}    R {len(inter_fn_per_polygon[1])}',
                (30, 50 * (len(polygons) + 1)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))


def plot_time_graphics(inter_fn_per_polygon: List[List[float]], fps: int):
    print('Plotting')
    fig = plt.figure()
    for i, times in enumerate(inter_fn_per_polygon):
        y = gaps_from_inter_times(times)
        X = np.array(times) / fps
        y = np.array(y) / fps
        plt.plot(X, y, label=f'{i}', marker='*')
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Gap (sec)')
    plt.title('Gaps along time')
    fig.savefig('cl_times.png')


def save_inter_times_to_record(inter_fn_per_polygon: List[List[float]], filename: str, fps: float):
    print(inter_fn_per_polygon)
    merged_array = []
    for i in range(len(inter_fn_per_polygon)):
        merged_array += inter_fn_per_polygon[i]

    merged_array.sort()
    merged_array = list(np.array(merged_array) / fps)
    with open(filename, 'w') as f:
        f.write('\n'.join(map(str, merged_array)))


def demo(opt: argparse.Namespace, img_size: Tuple[int, int], save_record: bool):
    cam = cv2.VideoCapture(opt.demo)
    fps = cam.get(cv2.CAP_PROP_FPS)
    assert cam.isOpened(), f'Video reading is broken'
    out = create_video_writer(fps, img_size)

    # Configure model
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)  # TODO: learn it deeper
    # miner (1) -> .1
    # seat (2) -> .3
    # loopmarker (3)
    # rider (4)
    vis_conf_thresholds = {1: .3, 2: .3, 3: .3, 4: .3}
    activated_classes = tuple([2, 4])
    color_by_class = {1: (0, 0, 255), 2: (255, 255, 255), 3: (255, 0, 0), 4: (0, 255, 0)}
    detector = CtdetDetector(opt, vis_conf_thresholds=vis_conf_thresholds)
    detector.pause = False
    tracker_by_class = {class_id: Tracker(min_hits_to_track=3, iou_threshold=.1, nonactive_track_max_times=30)
                        for class_id in activated_classes}

    polygons = Polygon.read_polygons_from_json(opt.demo)

    inter_fn_per_polygon = [[] for _ in range(len(polygons))]
    frame_number = 0
    handled_tracks = set([])
    while True:
        frame_number += 1
        is_opened, img = cam.read()
        if not is_opened:
            print('Finish by unabled images')
            break
        img = cv2.resize(img, dsize=img_size)
        img_res, ret = detector.run(img)
        img_dets = img_res.copy()

        # Apply tracking
        tracked_bboxes = {}
        for class_id in activated_classes:
            good_score_detections = list(filter(lambda bbox: bbox[4] >= vis_conf_thresholds[class_id],
                                                ret['results'][class_id]))
            for bbox in good_score_detections:
                bbox_int = list(map(int, bbox))
                cv2.rectangle(img_dets, bbox_int[:2], bbox_int[2:4], color=color_by_class[class_id], thickness=2)

            bboxes_for_tracker = [convert_real_to_norm_bbox_coords(bbox, img_size) for bbox in
                                  good_score_detections]
            tracked_bboxes[class_id] = tracker_by_class[class_id].update(bboxes_for_tracker)

        # Analyze tracks
        for class_id in activated_classes:
            for track_id in tracked_bboxes[class_id]:
                # Draw results for debugging
                norm_bbox = tracked_bboxes[class_id][track_id]
                real_bbox = convert_norm_to_real_bbox_coords(norm_bbox, img_size=img_size)
                cv2.rectangle(img_res, real_bbox[:2], real_bbox[2:4], color=color_by_class[class_id], thickness=2)
                cv2.putText(img_res,
                            f'{track_id}',
                            real_bbox[:2],
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))

                # Find intersections
                if not track_id in handled_tracks:
                    center = (real_bbox[0] + real_bbox[2]) // 2, (real_bbox[1] + real_bbox[3]) // 2
                    for i in range(len(polygons)):
                        if polygons[i].is_point_inside(*center):
                            inter_fn_per_polygon[i].append(frame_number)
                            handled_tracks.add(track_id)
                            print(f'Pol: {i} Track: {track_id}, center: {center}')
                            break
        handle_times_gap(polygons, inter_fn_per_polygon, img_res, fps)

        Polygon.draw_polygons_on_image(polygons, image=img_res)
        cv2.imshow('output', img_res)
        cv2.imshow('detections', img_dets)
        out.write(img_res)
        if cv2.waitKey(1) == ord(' '):
            print('Break the loop')
            break

    # Close all opened files
    cam.release()
    out.release()
    # Plot time graphic
    plot_time_graphics(inter_fn_per_polygon, fps)

    # Save times to file
    if save_record:
        record_path = os.path.join(os.path.abspath('.'), 'records', 'rec_4cl_2_min_hit_5.txt')
        save_inter_times_to_record(inter_fn_per_polygon, record_path, fps)


if __name__ == '__main__':
    FPS = 25
    IMG_SIZE = (640, 480)

    opt = Opts().init()
    demo(opt, img_size=IMG_SIZE, save_record=False)
