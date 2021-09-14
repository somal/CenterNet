from src.lib.datasets.dataset.coco_cl_ctdet import COCO_CL_CTDet

from src.lib.datasets.dataset.coco import COCO
from src.lib.datasets.dataset.coco_hp import COCOHP
from src.lib.datasets.dataset.kitti import KITTI
from src.lib.datasets.dataset.pascal import PascalVOC
from src.lib.datasets.sample.ctdet import CTDetDataset
from src.lib.datasets.sample.ddd import DddDataset
from src.lib.datasets.sample.exdet import EXDetDataset
from src.lib.datasets.sample.multi_pose import MultiPoseDataset

dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset
}


def get_dataset(dataset: str, task: str):
    if dataset == 'coco_cl':
        return COCO_CL_CTDet

    class CombinedDataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return CombinedDataset
