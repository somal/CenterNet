import argparse
import json
import math
import os
from collections import Counter
from typing import List, Dict, Iterable

import cv2
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from torch.utils import data

from src.lib.trains.base_trainer import BaseTrainer
from src.lib.utils.image import color_aug
from src.lib.utils.image import draw_dense_reg
from src.lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from src.lib.utils.image import get_affine_transform, affine_transform


class COCO_CL_CTDet(data.Dataset):
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
    val_size = .1

    def __init__(self, opt: argparse.Namespace, split: str, annotation_folder: str):
        super(COCO_CL_CTDet, self).__init__()
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        self.data_dir = opt.data_dir
        self.images = set([])

        self._annotation_folder = annotation_folder
        self.img_dir = os.path.join(self.data_dir, annotation_folder, '1')
        self.annot_path = os.path.join(self.img_dir, 'lbl', 'COCO_annotation.json')
        self.coco = coco.COCO(self.annot_path)

        # Get categories in needed order
        self._annotated_cat_ids = tuple(self.coco.getCatIds(catNms=[c])[0] for c in self.class_name
                                        if len(self.coco.getCatIds(catNms=[c])) > 0)
        assert isinstance(self._annotated_cat_ids, tuple) and isinstance(self._annotated_cat_ids[0], int)

        # Map classes and categories (which can be different from markup to markup)
        annotated_class_ids = range(len(self._annotated_cat_ids))
        self._coco_category_id_to_class_id = dict(zip(self._annotated_cat_ids, annotated_class_ids))
        self._class_id_to_coco_category_id = dict(zip(annotated_class_ids, self._annotated_cat_ids))

        self._cat_stats = {}  # type: Dict[int, int]
        for i, cat_id in enumerate(self._annotated_cat_ids):
            img_ids = self.coco.getImgIds(catIds=cat_id)
            self.images.update(set(img_ids))

            # Collect statistics
            class_name = self.class_name[self._coco_category_id_to_class_id[cat_id]]
            self._cat_stats[class_name] = len(img_ids)

        if self.split == 'train':
            self._idx_start = 0
            self.num_samples = int(len(self.images) * (1 - COCO_CL_CTDet.val_size))  # type: int
        else:
            self._idx_start = int(len(self.images) * (1 - COCO_CL_CTDet.val_size))
            self.num_samples = len(self.images) - self._idx_start
        self.images = list(self.images)  # type: List[int]

        # print(f'Loaded {split} {self.num_samples} samples with classes {self.class_name}')

    def __len__(self):
        return self.num_samples

    def get_stats_by_category(self) -> Dict[int, int]:
        return self._cat_stats

    @staticmethod
    def _to_float(x):
        return float(f"{x:.2f}")

    @staticmethod
    def convert_eval_format(all_bboxes: Dict[int, Dict[int, List]], category_by_class: Dict[int, int]) -> List[Dict]:
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = category_by_class.get(cls_ind - 1)
                if category_id is not None:
                    for bbox in all_bboxes[image_id][cls_ind]:
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        score = bbox[4]
                        bbox_out = list(map(COCO_CL_CTDet._to_float, bbox[0:4]))

                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": COCO_CL_CTDet._to_float(score)
                        }
                        # if score > .5:
                        detections.append(detection)
        return detections

    def run_eval(self, results, save_dir):
        converted_results = self.convert_eval_format(results, category_by_class=self._class_id_to_coco_category_id)
        json.dump(converted_results, open(f'{save_dir}/results.json', 'w'))
        coco_dets = self.coco.loadRes(f'{save_dir}/results.json')
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    @staticmethod
    def _coco_box_to_bbox(box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    @staticmethod
    def _get_border(border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index: int):
        img_id = self.images[index + self._idx_start]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        assert img is not None, f'Img from {img_path} is none'

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = COCO_CL_CTDet._get_border(128, img.shape[1])
                h_border = COCO_CL_CTDet._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = COCO_CL_CTDet._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self._coco_category_id_to_class_id[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret


class MultipleAnnotationsCOCOCL:
    @staticmethod
    def build(opt, split: str, coco_annotation_folders: Iterable[str]):
        datasets = [COCO_CL_CTDet(opt, split, annotation_folder=ann_path) for ann_path in
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

    @staticmethod
    def dataset_to_val_dataloader(dataset: data.Dataset):
        return data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )

    @staticmethod
    def run_eval(trainer: BaseTrainer, concat_dataset: data.ConcatDataset, opt: argparse.Namespace):
        for dataset in concat_dataset.datasets:
            print(dataset._annotation_folder)
            val_data_loader = MultipleAnnotationsCOCOCL.dataset_to_val_dataloader(dataset)
            ret, preds = trainer.val(0, val_data_loader)
            dataset.run_eval(preds, opt.save_dir)
