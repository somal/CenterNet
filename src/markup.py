import json

import cv2
import numpy as np


def markup_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['points'].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(param['points']) > 0:
        param['points'].pop(-1)


def markup(video_path: str):
    cam = cv2.VideoCapture(video_path)
    is_opened, img = cam.read()
    assert is_opened, f'Video reading is broken'
    img = cv2.resize(img, (640, 480))

    points = []
    polygons = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', markup_mouse_callback, param={'points': points})
    while cv2.waitKey(1) & 0xFF != 27:
        imgc = img.copy()

        for x, y in points:
            cv2.circle(imgc, (x, y), 8, (255, 100, 0), -1)
        cv2.polylines(imgc, pts=[np.array(points).reshape((-1, 1, 2))], isClosed=True, color=(255, 100, 0))

        cv2.imshow('image', imgc)
        if cv2.waitKey(20) & 0xFF == 0x0D:  # Enter
            print('Saving points ...')
            if len(points) > 0:
                polygons.append(points.copy())
                points.clear()
                video_name = video_path.split('/')[-1]
                json.dump(polygons, open(f'{video_name}.json', 'w'))
                print('Successful')
            else:
                print('Points are empty')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    markup('../videos/for_demo/ip:10.160.67.23_id:12_start:01:21:00.mp4')
