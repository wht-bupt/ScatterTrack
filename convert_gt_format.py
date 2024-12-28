import os
import pandas as pd
from data_toolbox import *

sup_width = 3

if __name__ == '__main__':
    print('old data deleted!')
    data_path = 'data/mota_gts'
    mts_path = 'data/mts_gts'
    del_path(data_path)
    del_path(mts_path)
    mkdirs(data_path)
    mkdirs(mts_path)

    gt_root = 'data/gts'
    gts = sorted(os.listdir(gt_root))
    for gt_idx, gt in enumerate(gts, start=1):
        gt_path = os.path.join(gt_root, gt)
        data_dataframe = pd.read_csv(gt_path, sep=',')
        data_list = data_dataframe.values.tolist()
        det_file = os.path.join(data_path, '{:03d}.txt'.format(gt_idx))
        mts_file = os.path.join(mts_path, '{:03d}.txt'.format(gt_idx))
        with open(det_file, 'w') as f, open(mts_file, 'w') as fs:
            for data_line in data_list:
                frame_id, obj_id, frag_id, x, y, _, _, _ = data_line
                new_data_line = [int(frame_id), '{}{:02d}'.format(int(obj_id), int(frag_id)),
                                 x-sup_width, y-sup_width, 2*sup_width, 2*sup_width, 1, 1, 1]
                write_line = ','.join(str(x) for x in new_data_line)
                f.write(write_line)
                f.write('\n')

                if frag_id == 0:
                    data_line1_for_mts = [int(frame_id), '{}{:02d}'.format(int(obj_id), 1),
                                          x-sup_width, y-sup_width, 2*sup_width, 2*sup_width, 1, 1, 1]
                    data_line2_for_mts = [int(frame_id), '{}{:02d}'.format(int(obj_id), 2),
                                          x-sup_width, y-sup_width, 2*sup_width, 2*sup_width, 1, 1, 1]
                    for data_mts_line in [data_line1_for_mts, data_line2_for_mts]:
                        mts_line = ','.join(str(x) for x in data_mts_line)
                        fs.write(mts_line)
                        fs.write('\n')

                else:
                    fs.write(write_line)
                    fs.write('\n')

        print('mota and mts format gt for video {} generated!'.format(gt_idx))