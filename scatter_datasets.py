import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from tracking_toolbox import *


class ScatterObjectDataset(Dataset):
    def __init__(self, cfg, data_root, mode='train'):
        super(ScatterObjectDataset, self).__init__()
        det_path = os.path.join(cfg.DATA.DET_ROOT, '{}.txt'.format(data_root))
        gt_path = os.path.join(cfg.DATA.GT_ROOT, '{}.txt'.format(data_root))
        self.dets = pd.read_csv(det_path, header=None).values.tolist()
        self.dets = self.transform_det_list(self.dets)
        self.mode = mode
        if self.mode == 'train':
            self.gts = pd.read_csv(gt_path, header=None).values.tolist()
            self.gts = self.transform_list(self.gts)
        else:
            self.gts = []

    def __len__(self):
        if self.mode == 'train':
            return min(len(self.dets), len(self.gts)) - 1
        else:
            return len(self.dets) - 1

    def transform_list(self, input_list):
        max_frame = int(max(item[0] for item in input_list) if input_list else 0)
        new_list = [[] for _ in range(max_frame + 1)]

        for item in input_list:
            frame, main_id, frag_id, x, y = item[:5]
            new_list[int(frame)].append([main_id, frag_id, x, y])

        return new_list

    def transform_det_list(self, input_list):
        input_list = sorted(input_list, key=lambda x: (x[0], x[1]))
        max_frame = int(max(item[0] for item in input_list) if input_list else 0)
        new_list = [[] for _ in range(max_frame + 1)]

        for item in input_list:
            frame, x, y = item
            new_list[int(frame)].append([x, y])

        return new_list

    def __getitem__(self, index):
        if self.mode == 'train':
            gt = self.gts[index]
            for i in range(len(gt)):
                gt[i][0] = int(gt[i][0])
        else:
            gt = []

        det = self.dets[index] if index < len(self.dets) else []

        return det, gt


class GroundTruthTracks(Dataset):
    # 为上分支制作数据集
    # 这一数据集结构专为生成gt的历史轨迹，数据集初始化时会遍历所有gt并且分配到对应的轨迹号里面，
    # 主轨迹会同时抄送给两个子轨迹。此数据集的__getitem__方法输出特定轨迹号的完整gt轨迹。
    def __init__(self, video_names, cfg):
        super(GroundTruthTracks, self).__init__()
        self.videos = video_names
        self.cfg = cfg
        self.gt_tracks = self.generate_tracks()

    def __getitem__(self, item):
        return self.gt_tracks[item]

    def __len__(self):
        return len(self.videos)

    def generate_tracks(self):
        gt_tracks = {}
        for video in self.videos:
            frame_data = ScatterObjectDataset(self.cfg, video, mode='train')
            gt_track = [[] for _ in range(self.cfg.PARAM.MAX_TRACK)]
            for frame_id in range(1, len(frame_data) + 1):
                _, gts = frame_data[frame_id]
                for gt in gts:
                    main_id, frag_id, x, y = gt
                    track_id = int('{}{:02d}'.format(int(main_id), int(frag_id)))

                    frag_1 = int('{}{:02d}'.format(int(main_id), 1))
                    frag_2 = int('{}{:02d}'.format(int(main_id), 2))
                    if frag_id == 0:
                        gt_track[int(track_id)].append([frame_id] + [x, y])
                        gt_track[int(frag_1)].append([frame_id] + [x, y])
                        gt_track[int(frag_2)].append([frame_id] + [x, y])
                    else:
                        gt_track[int(track_id)].append([frame_id] + [x, y])


            gt_tracks.update({video: gt_track})

        return gt_tracks


class ScatterTrackDataset(Dataset):
    def __init__(self, pred_track_list, det_track_list, track_positions, detections, det_group_result, cfg):
        super(ScatterTrackDataset, self).__init__()
        self.pred_track_list = pred_track_list
        self.det_track_list = det_track_list
        self.cfg = cfg
        device = self.cfg.LOAD.DEV
        if device == 'cpu':
            self.track_positions = track_positions
            self.detections = detections
            self.det_group_result = det_group_result
        else:
            self.track_positions = track_positions.to(device)
            self.detections = detections.to(device)
            self.det_group_result = det_group_result.to(device)

    def __len__(self):
        return len(self.pred_track_list)

    def __getitem__(self, item):
        pred_track_list_end = self.pred_track_list[item].index(-10)
        det_track_list_end = self.det_track_list[item].index(-10)

        end_positions = [pred_track_list_end, det_track_list_end]

        return self.pred_track_list[item], self.det_track_list[item], \
               self.track_positions[item], self.detections[item], self.det_group_result[item], end_positions
