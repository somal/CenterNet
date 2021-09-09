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

image_ext = ('jpg', 'jpeg', 'png', 'webp')
video_ext = ('mp4', 'mov', 'avi', 'mkv')


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    def __iter__(self):
        for p in (self.x1, self.y1, self.x2, self.y2):
            yield p


def check_line_inter(line1, line2):
    img1 = np.zeros(shape=(640, 480), dtype=np.uint8)
    img2 = np.zeros(shape=(640, 480), dtype=np.uint8)
    cv2.line(img1, *line1, color=(255, 255, 255), thickness=3)
    cv2.line(img2, *line2, color=(255, 255, 255), thickness=3)
    return np.sum(np.logical_and(img1, img2)) > 0


def check_intersection_line_and_rect(line, bbox):
    x1, y1, x2, y2 = bbox[:4]
    return any([check_line_inter(line, ((x1, y1), (x1, y2))),
                check_line_inter(line, ((x2, y1), (x2, y2))),
                check_line_inter(line, ((x1, y1), (x2, y1))),
                check_line_inter(line, ((x1, y2), (x2, y2))),
                ])


def gaps_from_inter_times(times: List[float]):
    y = [times[0]] if len(times) > 0 else []
    for j in range(1, len(times)):
        y.append((times[j] - times[j - 1]))
    return np.array(y)


def create_video_writer(fps: int, img_size: Tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter('output.mp4', fourcc, fps, img_size)


class Polygon:
    def __init__(self, points: List[List[int]]):
        self._pts = points

    def is_point_inside(self, x: int, y: int) -> bool:
        return cv2.pointPolygonTest(contour=np.array(self._pts), pt=(x, y), measureDist=False)

    @staticmethod
    def read_polygons_from_json(video_path: str):
        video_name = video_path.split('/')[-1]
        polygons = json.load(open(f'{video_name}.json', 'r'))
        return [Polygon(points) for points in polygons]

    @staticmethod
    def draw_polygons_on_image(polygons: List, image: np.array):
        for polygon in polygons:
            cv2.polylines(image, pts=[np.array(polygon._pts).reshape((-1, 1, 2))], isClosed=True, color=(255, 100, 0))


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


def demo(opt: argparse.Namespace, fps: int, img_size: Tuple[int, int]):
    cam = cv2.VideoCapture(opt.demo)
    assert cam.isOpened(), f'Video reading is broken'
    out = create_video_writer(fps, img_size)

    # Configure model
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)  # TODO: learn it deeper
    # miner (1) -> .1
    # seat (2) -> .3
    # loopmarker (3)
    vis_conf_thresholds = {1: .1, 2: .1, 3: .3}
    activated_classes = tuple([1, 2])
    color_by_class = {1: (0, 0, 255), 2: (255, 255, 255), 3: (255, 0, 0)}
    detector = CtdetDetector(opt, vis_conf_thresholds=vis_conf_thresholds)
    detector.pause = False
    tracker_by_class = {class_id: Tracker(min_hits_to_track=3, iou_threshold=.1, nonactive_track_max_times=10)
                        for class_id in activated_classes}

    polygons = Polygon.read_polygons_from_json(opt.demo)

    min_frames = 10 * fps
    inter_times_per_polygon = [[], []]
    frame_number = 0
    rrr = {0: 'L', 1: 'R'}
    while True:
        frame_number += 1
        is_opened, img = cam.read()
        if not is_opened:
            print('Finish by unabled images')
            break
        img = cv2.resize(img, dsize=img_size)
        img_res, ret = detector.run(img)

        # Apply tracking
        tracked_bboxes = {}
        for class_id in activated_classes:
            good_score_detections = list(filter(lambda bbox: bbox[4] >= vis_conf_thresholds[class_id],
                                                ret['results'][class_id]))
            bboxes_for_tracker = [convert_real_to_norm_bbox_coords(bbox, img_size) for bbox in
                                  good_score_detections]
            tracked_bboxes[class_id] = tracker_by_class[class_id].update(bboxes_for_tracker)

        for class_id in activated_classes:
            for track_id in tracked_bboxes[class_id]:
                norm_bbox = tracked_bboxes[class_id][track_id]
                real_bbox = convert_norm_to_real_bbox_coords(norm_bbox, img_size=img_size)
                cv2.rectangle(img_res, real_bbox[:2], real_bbox[2:4], color=color_by_class[class_id], thickness=2)
                cv2.putText(img_res,
                            f'{track_id}',
                            real_bbox[:2],
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))

        # for class_id in (1, 2):
        #     tracker.update(ret['results'][class_id])
        #     for bbox in ret['results'][class_id]:
        #         if bbox[4] > vis_conf_thresholds[class_id]:
        #             rect_bbox = tuple(map(int, bbox[:4]))
        #
        #             center = (rect_bbox[0] + rect_bbox[2]) // 2, (rect_bbox[1] + rect_bbox[3]) // 2
        #             for i in range(len(polygons)):
        #                 if (len(inter_times_per_polygon[i]) > 0
        #                     and frame_number - inter_times_per_polygon[i][-1] >= min_frames) or \
        #                         len(inter_times_per_polygon[i]) == 0:
        #
        #                     if polygons[i].is_point_inside(*center):
        #                         inter_times_per_polygon[i].append(frame_number)
        #                         print(i, frame_number)
        Polygon.draw_polygons_on_image(polygons, image=img_res)
        cv2.imshow('output', img_res)

        for i in range(len(polygons)):
            gaps = gaps_from_inter_times(inter_times_per_polygon[i])
            frame_diff = gaps[-1] / fps if len(gaps) > 0 else 0
            cv2.putText(img_res, f'Gap time {rrr[i]} {frame_diff:.2f}', (30, 50 * (i + 1)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
        cv2.putText(img_res, f'Count L {len(inter_times_per_polygon[0])}    R {len(inter_times_per_polygon[1])}',
                    (30, 50 * (len(polygons) + 1)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
        out.write(img_res)
        if cv2.waitKey(1) == ord(' '):
            print('Break the loop')
            break

    # Close all opened files
    cam.release()
    out.release()
    # Plot time graphic
    print('Plotting')
    fig = plt.figure()
    for i, times in enumerate(inter_times_per_polygon):
        y = gaps_from_inter_times(times)
        X = np.array(times) / fps
        y = np.array(y) / fps
        plt.plot(X, y, label=f'{rrr[i]}', marker='*')
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Gap (sec)')
    plt.title('Gaps along time')
    fig.savefig('cl_times.png')


if __name__ == '__main__':
    FPS = 25
    IMG_SIZE = (640, 480)

    opt = Opts().init()
    demo(opt, fps=FPS, img_size=IMG_SIZE)
