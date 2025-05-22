import os
import platform

dpath_training_records = r'b_data_train/a_records_train'

dpath_data_raw = r'b_data_train/data_a_raw'
dpath_data_cache = r'b_data_train/data_b_cache'
dpath_data_cook = r'b_data_train/data_c_cook'
dpath_esam_weights = r'FoSAM/l_sam/esam_weight'

dpath_data_raw_cityscape_X = os.path.join(dpath_data_raw, r'leftImg8bit')
dpath_data_raw_cityscape_Y = os.path.join(dpath_data_raw, r'gtFine')

fpath_data_raw_lvis_train = os.path.join(dpath_data_raw, r'lvis_v1_train', r'lvis_v1_train.json')
fpath_data_raw_lvis_valid = os.path.join(dpath_data_raw, r'lvis_v1_val', r'lvis_v1_val.json')

dpath_data_raw_coco_train = os.path.join(dpath_data_raw, r'coco2017', r'train2017')
dpath_data_raw_coco_valid = os.path.join(dpath_data_raw, r'coco2017', r'val2017')
dpath_data_raw_coco_test = os.path.join(dpath_data_raw, r'coco2017', r'test2017')

dpath_data_raw_ade20k = os.path.join(dpath_data_raw, r'ADE20K_2021_17_01')

os.makedirs(dpath_training_records, exist_ok=True)
