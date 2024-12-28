import random
import time
import os
import argparse
import logging
from yacs.config import CfgNode as CN
from data_toolbox import *
from tracking_toolbox import *
from scatter_tracker import ScatterTracker
from scatter_datasets import ScatterObjectDataset


def main(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    default_yaml = 'config/scatternet.yaml'
    default_cfg = open(default_yaml)
    cfg = CN.load_cfg(default_cfg)

    if args.device:
        cfg.LOAD.DEV = 'cuda:{}'.format(args.device)
    if args.checkpoint:
        cfg.TRACK.MODEL = args.checkpoint

    cfg.freeze()

    time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    save_dir = os.path.join(cfg.TRACK.SAVE_DIR, time_name)
    mkdirs(save_dir)

    track_save_root = os.path.join(save_dir, 'tracks')
    mota_save_root = os.path.join(save_dir, 'tracks_mota')
    sda_save_root = os.path.join(save_dir, 'tracks_mts')

    mkdirs(track_save_root)
    mkdirs(mota_save_root)
    mkdirs(sda_save_root)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file_name = time_name + '.log'
    log_file = os.path.join(save_dir, log_file_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logger.info('--------------------------- initialize ------------------------------')
    logger.info('cfg file: {}'.format(default_yaml))
    if cfg.TRACK.SEED:
        random_seed = cfg.TRACK.SEED
        random_initiate(seed=random_seed)
        logger.info('random seed initiated {}'.format(random_seed))
    device = cfg.LOAD.DEV
    data_path = cfg.TRACK.IMG_ROOT
    video_names = os.listdir(data_path)
    logger.info('found {} videos: {}'.format(len(video_names), video_names))

    logger.info('------------------------- start tracking ----------------------------')

    time_used_list = []
    frame_num_list = []
    for video_num, video in enumerate(video_names):
        if int(video[1:]) <= len(video_names) * cfg.TRAIN.TRAIN_SET_RATIO:  
            continue
        tracker = ScatterTracker(cfg, logger)
        video_result_file = os.path.join(track_save_root, '{}.txt'.format(video))
        mota_result_file = os.path.join(mota_save_root, '{}.txt'.format(video))
        sda_result_file = os.path.join(sda_save_root, '{}.txt'.format(video))

        # 记录三个结果，f为正常的跟踪结果，fm为评价mota使用的跟踪结果，fs为评价分离判别准确率(separation discrimination accuracy)的跟踪结果。
        # f 格式：[frame_id, track_id, fragment_id, x, y]
        # fm 格式：[frame_id, combined_id, x, y, w, h, 1, -1, -1, -1]
        # fs 格式：[frame_id, separate_id, x, y, w, h, 1, -1, -1, -1]
        with open(video_result_file, 'w') as f, open(mota_result_file, 'w') as fm, open(sda_result_file, 'w') as fs:
            frame_data = ScatterObjectDataset(cfg, video, mode='test')
            frame_start = 1
            frame_end = len(frame_data) + 1
            logger.info('tracking for video {} started!'.format(video))

            time_video_start = time.time()
            for frame_id in range(frame_start, frame_end):
                detections, _ = frame_data[frame_id]
                tracks = tracker.update(detections)
                for track in tracks:
                    sup_width = cfg.TRACK.SUP_WIDTH
                    data_line = [frame_id, track.track_id, track.fragment_id, track.kf.x[0], track.kf.x[1]]
                    data_line_for_mota = [frame_id, '{}{:02d}'.format(track.track_id, track.fragment_id),
                                          track.kf.x[0] - sup_width, track.kf.x[1] - sup_width, 2*sup_width,
                                          2*sup_width, 1, -1, -1, -1]
                    write_line = ','.join(str(x) for x in data_line)
                    mota_line = ','.join(str(x) for x in data_line_for_mota)
                    f.write(write_line)
                    f.write('\n')
                    fm.write(mota_line)
                    fm.write('\n')

                    if track.fragment_id == 0:
                        data_line1_for_sda = [frame_id, '{}{:02d}'.format(track.track_id, 1),
                                              track.kf.x[0] - sup_width, track.kf.x[1] - sup_width, 2 * sup_width,
                                              2 * sup_width, 1, -1, -1, -1]
                        data_line2_for_sda = [frame_id, '{}{:02d}'.format(track.track_id, 2),
                                              track.kf.x[0] - sup_width, track.kf.x[1] - sup_width, 2 * sup_width,
                                              2 * sup_width, 1, -1, -1, -1]
                        for data_sda_line in [data_line1_for_sda, data_line2_for_sda]:
                            sda_line = ','.join(str(x) for x in data_sda_line)
                            fs.write(sda_line)
                            fs.write('\n')

                    else:
                        fs.write(mota_line)
                        fs.write('\n')

                if frame_id % cfg.TRACK.PRINT_INT == 0:
                    logger.info('processing frame {}'.format(frame_id))

            time_video_end = time.time()
            time_used_list.append(time_video_end - time_video_start)
            frame_num_list.append(len(frame_data))
            fps_video = len(frame_data) / (time_video_end - time_video_start)
            logger.info('track result for video {} generated! fps:{:.2f}'.format(video, fps_video))

    fps_all = sum(frame_num_list) / sum(time_used_list)
    logger.info('tracking for all videos finished! fps:{:.2f}'.format(fps_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate track results')
    parser.add_argument('--checkpoint', '-c', default=None, type=str, help='checkpoint for tracking')
    parser.add_argument('--device', '-d', default=None, type=str, help='device used for tracking')
    args = parser.parse_args()
    main(args)


















