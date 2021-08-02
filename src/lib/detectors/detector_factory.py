from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.lib.detectors.ctdet import CtdetDetector
from src.lib.detectors.ddd import DddDetector
from src.lib.detectors.exdet import ExdetDetector
from src.lib.detectors.multi_pose import MultiPoseDetector

detector_factory = {
    'exdet': ExdetDetector,
    'ddd': DddDetector,
    'ctdet': CtdetDetector,
    'multi_pose': MultiPoseDetector,
}
