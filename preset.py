import os
import platform

# # 获取 Windows 的环境变量 COMPUTERNAME

# PC_NAME_server_H100 = 'sn4622121202'
# PC_NAME_server_4090 = 'sn4622121201'
# PC_NAME_local = 'XPS'

# if platform.system() == 'Windows':
#     pc_name = os.environ.get('COMPUTERNAME')
# else:
#     pc_name = os.environ.get('HOSTNAME')

# dpath_training_records = None

# dpath_training_records_local = r'C:\Users\harry\Dropbox\AGI\CS-GY-997X\DynamicFocus\a_records_train'
# dpath_training_records_server = r'/home/lwx/test/DynamicFocus/a_records_train'

# dpath_data_raw = ''

# if pc_name == PC_NAME_local:
#     dpath_training_records = dpath_training_records_local

#     dpath_data_raw = r'D:\b_data_train\data_a_raw'
#     dpath_data_cache = r'D:\b_data_train\data_b_cache'
#     dpath_data_cook = r'D:\b_data_train\data_c_cook'



# elif pc_name == PC_NAME_server_H100 or pc_name == PC_NAME_server_4090:
dpath_training_records = r'/home/wang/b_data_train/a_records_train'

dpath_data_raw = r'/home/wang/b_data_train/data_a_raw'
dpath_data_cache = r'/home/wang/b_data_train/data_b_cache'
dpath_data_cook = r'/home/wang/b_data_train/data_c_cook'
dpath_esam_weights = r'/home/wang/FoSAM/l_sam/esam_weight'
#dpath_sam_checkpoints = r'/root/autodl-tmp/h_project/sam/checkpoints'

# dpath_data_raw_cityscape_X = os.path.join(dpath_data_raw, r'leftImg8bit_trainvaltest', r'leftImg8bit')
# dpath_data_raw_cityscape_Y = os.path.join(dpath_data_raw, r'gtFine_trainvaltest', r'gtFine')

dpath_data_raw_cityscape_X = os.path.join(dpath_data_raw, r'leftImg8bit')
dpath_data_raw_cityscape_Y = os.path.join(dpath_data_raw, r'gtFine')

fpath_data_raw_lvis_train = os.path.join(dpath_data_raw, r'lvis_v1_train', r'lvis_v1_train.json')
fpath_data_raw_lvis_valid = os.path.join(dpath_data_raw, r'lvis_v1_val', r'lvis_v1_val.json')

dpath_data_raw_coco_train = os.path.join(dpath_data_raw, r'coco2017', r'train2017')
dpath_data_raw_coco_valid = os.path.join(dpath_data_raw, r'coco2017', r'val2017')
dpath_data_raw_coco_test = os.path.join(dpath_data_raw, r'coco2017', r'test2017')

dpath_data_raw_ade20k = os.path.join(dpath_data_raw, r'ADE20K_2021_17_01')

os.makedirs(dpath_training_records, exist_ok=True)
