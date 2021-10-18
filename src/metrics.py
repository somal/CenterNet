from pathlib import Path

import numpy as np


def calc_metric(gt_path: Path, records_path: Path, max_dist: float = 5.0):
    gt_times = list(map(float, open(gt_path).readlines()))
    records_times = np.array(list(map(float, open(records_path).readlines())))

    tp, fp, tn, fn = 0, 0, 0, 0
    for gt_time in gt_times:
        ready_records = records_times.copy()
        idx_with_minimal_distance = np.argmin(np.abs(ready_records - gt_time))
        print(gt_time)
        nearest_rec_value = ready_records[idx_with_minimal_distance]
        dist = np.abs(nearest_rec_value - gt_time)
        if 0 < dist < max_dist:
            tp += 1

            records_times = np.delete(records_times, idx_with_minimal_distance)
        else:
            fn += 1
            print(dist, nearest_rec_value)

    fp = records_times.shape[0]
    print(tp, fp, tn, fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'Precision: {precision:.3f}\n'
          f'Recall: {recall:.3f}\n'
          f'F1: {f1:.3f}\n')


if __name__ == '__main__':
    gt_path = Path('./records/gt_centered.txt').absolute()
    records_path = Path('./records/rec_4cl_3_min_hit_5.txt').absolute()
    calc_metric(gt_path, records_path, max_dist=5.0)
