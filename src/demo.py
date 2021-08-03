import os

import cv2
import numpy as np

from src.lib.detectors.detector_factory import detector_factory
from src.lib.opts import opts

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


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


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:

        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (640, 480))

        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        lines = [((90, 350), (420, 110)), ((358, 400), (550, 100))]

        min_frames = 10 * fps
        inter_times_per_line = [[], []]
        frame_number = 0
        while True:
            frame_number += 1
            _, img = cam.read()
            img = cv2.resize(img, (640, 480))
            cv2.imshow('input', img)
            # [cv2.line(img, *line, color=(255, 255, 255), thickness=3) for line in lines]
            img_res, ret = detector.run(img)

            for bbox in ret['results'][2]:
                if bbox[4] > .3:
                    rect_bbox = tuple(map(int, bbox[:4]))
                    for i in range(len(lines)):
                        if (len(inter_times_per_line[i]) > 0
                            and frame_number - inter_times_per_line[i][-1] >= min_frames) or \
                                len(inter_times_per_line[i]) == 0:
                            # rect_inter = cv2.rotatedRectangleIntersection(rects[i], rect_bbox)
                            if check_intersection_line_and_rect(lines[i], rect_bbox):
                                # print(rect_inter)
                                inter_times_per_line[i].append(frame_number)
                                print(i, inter_times_per_line)

            rrr = {0: 'L', 1: 'R'}
            for i in range(len(lines)):
                frame_diff = (inter_times_per_line[i][-1]) / len(inter_times_per_line[i]) \
                    if len(inter_times_per_line[i]) else 0
                cv2.putText(img_res, f'Avg gap time {rrr[i]} {frame_diff / fps:.2f}', (30, 50 * (i + 1)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
            cv2.putText(img_res, f'Count L {len(inter_times_per_line[0])}    R {len(inter_times_per_line[1])}',
                        (30, 50 * (len(lines) + 1)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.75, color=(255, 255, 255))
            # print(ret['results'])
            out.write(img_res)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            # print(time_str)
            if cv2.waitKey(1) == 27:
                cam.release()
                out.release()
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
