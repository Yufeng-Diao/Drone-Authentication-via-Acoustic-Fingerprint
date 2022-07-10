# -*- coding: utf-8 -*-

# Drone attribute
name_set_drone = {}
name_set_drone['prefix'] = ['_uav_', '_noise_']

# name_set_drone['date'] = ['_20210803_','_20210804_','_20210805_','_20210806_',
#                         '_20210807_','_20210808_','_20210809_','_20210817_',
#                         '_20210818_','_20210821_',
#                         '_20210831_',
#                         '_20210902_','_20210904_','_20210905_','_20210907_',
#                         '_20210912','_20210913_','_20210917_']

name_set_drone['date'] = ['_20220227_', '_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                          '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                          '_20220326_',
                          '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_',
                          '_2022noise_']

# name_set_drone['drone_No'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_',
#                             '_d7_','_d8_','_d9_','_d10_','_d11_','_d12_',
#                             '_t','enviro']

name_set_drone['drone_No'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
                              '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
                              '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_',
                              '_n_']

# name_set_drone['state'] = ['_hover_','_lr_','_ud_','_xjbfly_','_noise_','_fz_']

name_set_drone['state'] = ['_hover_', '_noise_']

name_set_drone['distance'] = ['_1m_','_5m_']

name_set_drone['index'] = ['_1_','_2_']

name_set_drone['suffix'] = ['.wav']

# Drone set
drone_set = {}

drone_set['1to8'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_']

drone_set['9to16'] = ['_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_']

drone_set['17to24'] = ['_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']


# MFCC and label csv file
name_set_csv = {}
# The MFCC should be in lower case because of the algorithm in idd
name_set_csv['prefix'] = ['_mfcc_', '_label_']

name_set_csv['date'] = name_set_drone['date']

name_set_csv['num_filter'] = ['nf_']

name_set_csv['num_cep'] = ['nc_']

name_set_csv['winlen'] = ['wl_']

name_set_csv['winstep'] = ['ws_']

name_set_csv['multiset'] = ['_1to8_', '_9to16_', '_17to24_']

name_set_csv['suffix'] = ['.csv']

if __name__ == '__main__':
    import numpy as np
    label_dic = dict(zip(name_set_drone['drone_No'], 
                         np.arange(len(name_set_drone['drone_No']))))