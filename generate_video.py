import copy
import random
import cv2
import time
import os
import tifffile
import pandas as pd
import numpy as np
import argparse
import logging
from data_toolbox import *
from yacs.config import CfgNode as CN
from scatter_detector import *



class ScatterDataForVideo:
    def __init__(self, gts):
        self.gts = self.slice_gt(gts)

    def __getitem__(self, item):
        return self.gts[item]

    def __len__(self):
        return len(self.gts) - 1

    def slice_gt(self, gts):
        sliced_gts = [[]]      # 提前放入一个，从1开始记轨迹
        last_frame = int(gts[-1][0])
        det_anchor = 0

        gt_for_track = []
        for i in range(1, last_frame+1):
            gt_for_track = []
            for j in range(det_anchor, len(gts)):
                if gts[j][0] > i:
                    sliced_gts.append(gt_for_track)
                    det_anchor = j
                    break
                gt_for_track.append(gts[j])

        sliced_gts.append(gt_for_track)

        return sliced_gts

def generate_videos(video_size, cfg):
    bgs = sorted(os.listdir(cfg.DATA.BG_ROOT))
    enhanced_bg_list = bg_list_enhance(bgs)
    bg_list_len = len(enhanced_bg_list)
    gts = sorted(os.listdir(cfg.DATA.GT_ROOT))
    for gt_idx, gt in enumerate(gts):
        video_output_path = os.path.join(cfg.DATA.VIDEO_ROOT, '{:03d}.avi'.format(gt_idx+1))
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                              (video_size[1], video_size[0]), isColor=False)

        gt_path = os.path.join(cfg.DATA.GT_ROOT, gt)
        data_dataframe = pd.read_csv(gt_path, sep=',')
        data_list = data_dataframe.values.tolist()
        last_frame = int(data_list[-1][0])
        data_sliced = ScatterDataForVideo(data_list)
        mkdirs(os.path.join(cfg.DATA.IMG_ROOT, '{:03d}'.format(gt_idx+1)))
        mkdirs(os.path.join(cfg.DATA.IMG_NOISE_ROOT, '{:03d}'.format(gt_idx+1)))

        for frame_id in range(1, last_frame + 1):
            org_bg_figure = tifffile.imread(os.path.join(cfg.DATA.BG_ROOT, enhanced_bg_list[frame_id % bg_list_len]))
            bg_figure = copy.copy(org_bg_figure)
            stretched_bg_figure = bg_stretch(bg_figure)
            bg_org = cv2.resize(stretched_bg_figure, video_size, interpolation=cv2.INTER_LINEAR)

            img_path = os.path.join(cfg.DATA.IMG_ROOT, '{:03d}/{:04d}.jpg'.format(gt_idx+1, frame_id))
            img_noise_path = os.path.join(cfg.DATA.IMG_NOISE_ROOT, '{:03d}/{:04d}.jpg'.format(gt_idx+1, frame_id))

            data_in_frame = data_sliced[frame_id]
            bg_with_noise = np.random.randint(10, 30, size=video_size)
            
            for line in data_in_frame:
                _, _, _, x, y, shape_x, shape_y, snr = line
                x = int(x)
                y = int(y)
                shape_x = int(shape_x)
                shape_y = int(shape_y)
                target = generate_tgt(shape_x, shape_y, snr, dist_ratio=3, kernel_len=2, noise_std=5.14)
                x_top = max(x - shape_x // 2, 0)
                y_top = max(y - shape_y // 2, 0)
                if x + shape_x // 2 < video_size[0] and y + shape_y // 2 < video_size[1]:
                    tgt_area_for_noise_bg = bg_with_noise[x_top:x_top+shape_x, y_top:y_top+shape_y].astype(int) + target
                    tgt_area_for_noise_bg = np.clip(tgt_area_for_noise_bg, 0, 255)
                    bg_with_noise[x_top:x_top+shape_x, y_top:y_top+shape_y] = tgt_area_for_noise_bg

                    tgt_area_for_org_bg = bg_org[x_top:x_top+shape_x, y_top:y_top+shape_y].astype(int) + target
                    tgt_area_for_org_bg = np.clip(tgt_area_for_org_bg, 0, 255)
                    bg_org[x_top:x_top+shape_x, y_top:y_top+shape_y] = tgt_area_for_org_bg

                bg_with_noise = np.clip(bg_with_noise, 0, 255)
                bg_org = np.clip(bg_org, 0, 255)

            bg_with_noise = bg_with_noise.astype(np.uint8).T
            bg_org = bg_org.astype(np.uint8).T

            out.write(bg_with_noise)
            cv2.imwrite(img_noise_path, bg_with_noise)
            cv2.imwrite(img_path, bg_org)
            
            if frame_id % 100 == 0:
                print('processing video {}, frame {}'.format(gt_idx+1, frame_id))

        print('video {} generated!'.format(gt_idx+1))


def generate_detections(cfg):
    mkdirs(cfg.DET.DET_SAVE_DIR)
    mkdirs(cfg.DET.FULL_DET_SAVE_DIR)
    mkdirs(cfg.DET.LOGS_SAVE_DIR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.log'
    log_file = os.path.join(cfg.DET.LOGS_SAVE_DIR, log_file_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    video_root = cfg.DATA.VIDEO_ROOT
    videos = sorted(os.listdir(video_root))
    videos_time = []
    for video_idx, video in enumerate(videos, start=1):
        frame_root = os.path.join(cfg.DATA.IMG_NOISE_ROOT, '{}'.format(video[:3]))
        frames = sorted(os.listdir(frame_root))
        det_file = os.path.join(cfg.DET.DET_SAVE_DIR, '{}.txt'.format(video[:3]))
        full_det_file = os.path.join(cfg.DET.FULL_DET_SAVE_DIR, '{}.txt'.format(video[:3]))
        time_start = time.time()

        with open(det_file, 'w') as f, open(full_det_file, 'w') as ff:
            for frame_id, frame in enumerate(frames, start=1):
                frame_path = os.path.join(frame_root, frame)
                img = cv2.imread(frame_path)
                fragments = get_fragments_from_universe(img, cfg)

                # 存储[x, y]和[x, y, w, h]两种格式
                for fragment in fragments:
                    x, y, w, h = fragment
                    pos_x = x + w / 2.
                    pos_y = y + h / 2.
                    det_line = [frame_id, pos_x, pos_y]
                    full_det_line = [frame_id, x, y, w, h, 1, 1]
                    det_write_line = ','.join(str(x) for x in det_line)
                    full_det_write_line = ','.join(str(x) for x in full_det_line)
                    f.write(det_write_line)
                    f.write('\n')
                    ff.write(full_det_write_line)
                    ff.write('\n')

                if frame_id % 200 == 0:
                    logger.info('processing frame {} of video {}'.format(frame_id, video_idx))

        time_used = time.time() - time_start
        fps_video = frame_id / time_used
        videos_time.append(fps_video)
        logger.info('detection result for video {} generated! fps: {:.2f}'.format(video_idx, fps_video))

    avg_fps = np.mean(videos_time)
    logger.info('detection result for all videos generated! average fps: {:.2f}'.format(avg_fps))



def main():
    parser = argparse.ArgumentParser(description='Generate videos and detection results.')
    parser.add_argument('--cfg', type=str, default='config/scatternet.yaml', help='Name of the cfg.')
    parser.add_argument('--generate_new_gts', type=bool, default=False, help='Generate new gts or use ours.')
    
    args = parser.parse_args()
    cfg = CN.load_cfg(open(args.cfg))
    cfg.freeze()

    #init settings
    video_size = (500, 500)
    gt_path = cfg.DATA.GT_ROOT
    bg_path = cfg.DATA.BG_ROOT
    det_path = cfg.DET.SAVE_DIR

    video_root = cfg.DATA.VIDEO_ROOT
    img_root = cfg.DATA.IMG_ROOT
    img_noise_root = cfg.DATA.IMG_NOISE_ROOT

    del_path(video_root)
    del_path(img_root)
    del_path(img_noise_root)
    del_path(det_path)
    mkdirs(video_root)
    mkdirs(img_root)
    mkdirs(img_noise_root)
    mkdirs(det_path)

    # generate gts
    if args.generate_new_gts:
        del_path(gt_path)
        mkdirs(gt_path)
        for video_seq in range(1, 21):
            generate_hard_track(video_seq, video_size, speed_fix_rate=5)
            print('gt {} generated!'.format(video_seq))
    else:
        print('gts is ready')

    # generate videos
    generate_videos(video_size, cfg)

    # generate detections
    generate_detections(cfg)



if __name__ == "__main__":
    main()

