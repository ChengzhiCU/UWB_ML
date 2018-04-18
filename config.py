import os

ROOT_PATH = '/home/maocz/Project'
MODEL_PATH = os.path.join(ROOT_PATH, 'models')
FIG_PATH = os.path.join(ROOT_PATH, 'fig_path')

MAT_PLOT_PATH = os.path.join(ROOT_PATH, 'MatFiles')

# Data_root_path = '/home/maocz/data'
Data_root_path = '/mnt/sdb/data_mao'
MatDataPath = os.path.join(Data_root_path, 'mat_data')
MatData_6F_ALL_Path = os.path.join(Data_root_path, 'mat_data_6f')
MatData_6F_OLDNEW_ALL_Path = os.path.join(Data_root_path, 'mat_data_6f_old_new')
MatData_6F_NLOS_Path = os.path.join(Data_root_path, 'mat_data_nlos_6f')
MatDataLOS6F_OLD_Path = os.path.join(Data_root_path, 'mat_data_los_6f_old_board')
MatDataLOS6F_NEW_Path = os.path.join(Data_root_path, 'mat_data_los_6f_new_board')
MatData_LOSNEW_NLOSOLD = os.path.join(Data_root_path, 'mat_data_6f_old_new')
UNLABELED_MATDATA_PATH = os.path.join(Data_root_path, 'unlabeled')

PAESED_FILES = os.path.join(Data_root_path, 'data_block')
PAESED_FILES_6F_ALL = os.path.join(Data_root_path, 'data_block_6f')
PAESED_FILES_6F_NLOS = os.path.join(Data_root_path, 'data_block_nlos_6f')
PARSED_FILES_LOSNEW_NLOSOLD = os.path.join(Data_root_path, 'data_block_losnew_nlosold')

LOS_PAESED_FILES = os.path.join(Data_root_path, 'los_data_block')
LOS_PAESED_FILES_NEW = os.path.join(Data_root_path, 'los_data_block_new_6f')
LOS_PAESED_FILES_OLD = os.path.join(Data_root_path, 'los_data_block_old_6f')
UNLABELED_PARSED = os.path.join(Data_root_path, 'unlabeled_parsed')

INPUT_DIM = 7


