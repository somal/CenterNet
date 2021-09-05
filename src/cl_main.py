import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import Opts

image_ext = ('jpg', 'jpeg', 'png', 'webp')
video_ext = ('mp4', 'mov', 'avi', 'mkv')
time_stats = ('tot', 'load', 'pre', 'net', 'dec', 'post', 'merge')


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


def demo(opt: argparse.Namespace):
    FPS = 25
    IMG_SIZE = (640, 480)
    cam = cv2.VideoCapture(opt.demo)
    assert cam.isOpened(), f'Video reading is broken'
    out = create_video_writer(FPS, IMG_SIZE)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)  # TODO: learn it deeper
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    detector.pause = False
    # lines = [((90, 350), (420, 110)), ((358, 400), (550, 100))] # cl2.mp4
    lines = [((261, 155), (330, 480)),
             ((480, 480), (315, 135))]  # loopmarkers/ip:10.224.44.8_id:20_start:07:39:30.mp4

    min_frames = 10 * FPS
    inter_times_per_line = [[], []]
    frame_number = 0
    rrr = {0: 'L', 1: 'R'}
    while True:
        frame_number += 1
        # if frame_number > (19 * 60 + 14) * FPS:
        #     print('Finish by time')
        #     break
        is_opened, img = cam.read()
        if not is_opened:
            print('Finish by unabled images')
            break
        # img = cv2.resize(img, (640, 480))
        # cv2.imshow('input', img)
        # [cv2.line(img, *line, color=(255, 255, 255), thickness=3) for line in lines]
        img_res, ret = detector.run(img)

        # seat (2) -> .3
        # miner (1) -> .1
        threshold = {1: .3, 2: .3, 3: .3}
        for class_id in (1, 2, 3):
            for bbox in ret['results'][class_id]:
                if bbox[4] > threshold[class_id]:
                    rect_bbox = tuple(map(int, bbox[:4]))
                    for i in range(len(lines)):
                        if (len(inter_times_per_line[i]) > 0
                            and frame_number - inter_times_per_line[i][-1] >= min_frames) or \
                                len(inter_times_per_line[i]) == 0:

                            if check_intersection_line_and_rect(lines[i], rect_bbox):
                                inter_times_per_line[i].append(frame_number)
                                print(i, frame_number)

        # loopmaker_detected = False
        # for bbox in ret['results'][3]:
        #     if bbox[4] > .2:
        #         loopmaker_detected = True
        #         print(f'Loopmaker detected, time: {frame_number / FPS}, thresh: {bbox[4]}')
        #         break
        # if loopmaker_detected:
        #     break
        # max_confs = {1: 0, 3: 0}
        # for i in max_confs.keys():
        #     for bbox in ret['results'][i]:
        #         max_confs[i] = max(max_confs[i], bbox[4])
        # print(max_confs)

        for i in range(len(lines)):
            gaps = gaps_from_inter_times(inter_times_per_line[i])
            frame_diff = gaps[-1] / FPS if len(gaps) > 0 else 0
            cv2.putText(img_res, f'Gap time {rrr[i]} {frame_diff:.2f}', (30, 50 * (i + 1)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
        cv2.putText(img_res, f'Count L {len(inter_times_per_line[0])}    R {len(inter_times_per_line[1])}',
                    (30, 50 * (len(lines) + 1)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
        # print(ret['results'])
        out.write(img_res)
        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        if cv2.waitKey(1) == ord(' '):
            print('Break the loop')
            break

    # Close all opened files
    cam.release()
    out.release()
    # Plot time graphic
    print('Plotting')
    fig = plt.figure()
    for i, times in enumerate(inter_times_per_line):
        y = gaps_from_inter_times(times)
        X = np.array(times) / FPS
        y = np.array(y) / FPS
        plt.plot(X, y, label=f'{rrr[i]}', marker='*')
    plt.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel('Gap (sec)')
    plt.title('Gaps along time')
    fig.savefig('cl_times.png')


if __name__ == '__main__':
    opt = Opts().init()
    demo(opt)
