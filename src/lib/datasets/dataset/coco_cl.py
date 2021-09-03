import json
import os
from collections import Counter
from typing import List, Dict, Iterable

import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from torch.utils import data


class COCO_CL(data.Dataset):
    num_classes = 3
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape((1, 1, 3))
    class_name = ('miner', 'seat', 'loopmarker')
    max_objs = 128
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    def __init__(self, opt, split, annotation_folder: str):
        super(COCO_CL, self).__init__()
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing coco cabel_line {} data.'.format(split))
        self.data_dir = opt.data_dir
        self.images = set([])

        self.img_dir = os.path.join(self.data_dir, annotation_folder, '1')
        self.annot_path = os.path.join(self.img_dir, 'lbl', 'COCO_annotation.json')
        self.coco = coco.COCO(self.annot_path)
        self._valid_ids = self.coco.getCatIds(catNms=self.class_name)

        self._cat_stats = {}  # type: Dict[int, int]
        for i, cat_id in enumerate(self._valid_ids):
            img_ids = self.coco.getImgIds(catIds=cat_id)
            self.images.update(set(img_ids))

            # Collect statistics
            class_name = self.class_name[i]
            self._cat_stats[class_name] = len(img_ids)
        assert isinstance(self._valid_ids, list) and isinstance(self._valid_ids[0], int)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}  # type: Dict[int]
        self.num_samples = len(self.images)  # type: int
        self.images = list(self.images)  # type: List[int]

        # print(f'Loaded {split} {self.num_samples} samples with classes {self.class_name}')

    def get_stats_by_category(self) -> Dict[int, int]:
        return self._cat_stats

    @staticmethod
    def _to_float(x):
        return float(f"{x:.2f}")

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(COCO_CL._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    # if len(bbox) > 5:
                    #     extreme_points = list(map(COCO_CL._to_float, bbox[5:13]))
                    #     detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open(f'{save_dir}/results.json', 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


class MultipleAnnotationsCOCOCL:
    @staticmethod
    def build(combined_dataset_cls, opt, split: str, coco_annotation_folders: Iterable[str]):
        assert issubclass(combined_dataset_cls, COCO_CL)
        datasets = [combined_dataset_cls(opt, split, annotation_folder=ann_path) for ann_path in
                    coco_annotation_folders]
        dataset = data.ConcatDataset(datasets=datasets)

        print(f'Img stats: {MultipleAnnotationsCOCOCL.get_stats_by_category(dataset)}')
        return dataset

    @staticmethod
    def get_stats_by_category(concat_dataset: data.ConcatDataset) -> Dict[int, int]:
        stats = Counter()
        for d in concat_dataset.datasets:
            current_stat = d.get_stats_by_category()
            current_stat = Counter(current_stat)
            stats += current_stat
        return dict(stats)
