from src.lib.datasets.dataset.coco import COCO
from src.lib.datasets.dataset.coco_cl import COCO_CL
from src.lib.datasets.dataset.coco_hp import COCOHP
from src.lib.datasets.dataset.kitti import KITTI
from src.lib.datasets.dataset.pascal import PascalVOC
from src.lib.datasets.sample.ctdet import CTDetDataset
from src.lib.datasets.sample.ddd import DddDataset
from src.lib.datasets.sample.exdet import EXDetDataset
from src.lib.datasets.sample.multi_pose import MultiPoseDataset

dataset_factory = {
    'coco': COCO,
    'coco_cl': COCO_CL,
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


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
