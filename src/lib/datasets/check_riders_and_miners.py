import ast
import random
from pathlib import Path

import cv2
import pandas as pd

folders = ('esaul_20', 'esaul_21', 'raspd-2_30', 'rcocs-1_12')
sub_folders = ('1', '2')

base_path = Path(r'C:\Users\vpavl\Desktop\CableLine')
annotation_path = r'lbl\via_annotation2.csv'

path_for_cropped_riders = r'C:\Users\vpavl\Desktop\CroppedRiders'
path_for_original_riders = r'C:\Users\vpavl\Desktop\OriginalRiders'

path_for_cropped_miners = r'C:\Users\vpavl\Desktop\CroppedMiners'
path_for_original_miners = r'C:\Users\vpavl\Desktop\OriginalMiners'


def process_one_class(dataset, folder, sub_folder, path_for_cropped, path_for_original):
    for _, row in dataset.iterrows():
        image_name, annotation = row['filename'], row['region_shape_attributes']
        annotation = ast.literal_eval(annotation)

        full_image_path = base_path.joinpath(folder, sub_folder, image_name)

        image = cv2.imread(str(full_image_path))
        if image is None:
            print(full_image_path)

        x1, y1, x2, y2 = \
            int(annotation['x']),\
            int(annotation['y']), \
            int(annotation['x']) + int(annotation['width']), \
            int(annotation['y']) + int(annotation['height'])

        cropped_object = image[y1:y2, x1:x2]
        random_name = random.getrandbits(128)
        cv2.imwrite(str(Path(path_for_cropped).joinpath(str(random_name) + '.jpg')), cropped_object)
        cv2.imwrite(str(Path(path_for_original).joinpath(str(random_name) + '.jpg')), image)


def process_annotation():
    for folder in folders:
        for sub_folder in sub_folders:
            annotations = pd.read_csv(base_path.joinpath(folder, sub_folder, annotation_path))
            miners = annotations[annotations['region_attributes'] == '{"type":"miner"}']
            riders = annotations[annotations['region_attributes'] == '{"type":"rider"}']

            print(folder, sub_folder)

            process_one_class(miners, folder, sub_folder, path_for_cropped_miners, path_for_original_miners)
            process_one_class(riders, folder, sub_folder, path_for_cropped_riders, path_for_original_riders)
