from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.lib.trains.ctdet import CtdetTrainer
from src.lib.trains.ddd import DddTrainer
from src.lib.trains.exdet import ExdetTrainer
from src.lib.trains.multi_pose import MultiPoseTrainer

train_factory = {
    'exdet': ExdetTrainer,
    'ddd': DddTrainer,
    'ctdet': CtdetTrainer,
    'multi_pose': MultiPoseTrainer,
}
