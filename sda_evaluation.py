import os
import pandas as pd
import numpy as np
from tracking_toolbox import *
from collections import Counter



class SDAEvaluationDataset():
    def __init__(self, track_path, gt_path):
        super(SDAEvaluationDataset, self).__init__()
        self.track_data = pd.read_csv(track_path, header=None).values.tolist()
        self.gt_data = pd.read_csv(gt_path, header=None).values.tolist()
        self.len_for_track = int(max(item[0] for item in self.track_data))
        self.len_for_gt = int(max(item[0] for item in self.gt_data))
        self.len_for_data = max(self.len_for_track, self.len_for_gt)
        self.tracks_by_frame = self.transform_list(self.track_data)
        self.gt_by_frame = self.transform_list(self.gt_data)

    def __getitem__(self, item):
        return self.tracks_by_frame[item], self.gt_by_frame[item]

    def __len__(self):
        return self.len_for_data - 1

    def transform_list(self, input_list):
        # 格式：[frame_id, main_id, frag_id, x, y]
        new_list = [[] for _ in range(self.len_for_data + 1)]

        for item in input_list:
            frame, main_id, frag_id, x, y = item[:5]
            new_list[int(frame)].append([main_id, frag_id, x, y])

        return new_list

    def get_gt_inform(self):
        max_track_for_gt = int(max(item[1] for item in self.gt_data))
        gt_breakpoint_list = [0 for _ in range(max_track_for_gt + 1)]

        for line in self.gt_data:
            frame, main_id, frag_id, x, y = line[:5]
            if frag_id != 0:
                if gt_breakpoint_list[int(main_id)] == 0:
                    gt_breakpoint_list[int(main_id)] = frame

        return max_track_for_gt, gt_breakpoint_list


def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_most_common_element(lst):

    if len(lst) == 0:
        return 0, 0
    counter = Counter(lst)
    most_common_element, count = counter.most_common(1)[0]

    return most_common_element, count


def sda_evaluation(cfg, track_result_path, gt_data_path):
    all_track_results = sorted(os.listdir(track_result_path))
    all_gts = sorted(os.listdir(gt_data_path))
    file_list = []
    total_tracks = 0
    total_success = 0
    for track_file in all_track_results:
        for gt_file in all_gts:
            if track_file[-7:-4] == gt_file[-7:-4]:
                file_list.append([track_file, gt_file])

    for file_pair in file_list:
        track_file, gt_file = file_pair
        file_name = track_file[-7:-4]
        track_full_path = os.path.join(track_result_path, track_file)
        gt_full_path = os.path.join(gt_data_path, gt_file)
        eval_data = SDAEvaluationDataset(track_full_path, gt_full_path)
        max_track, gt_breakpoint_list = eval_data.get_gt_inform()
        track_result_list_main = [[] for _ in range(max_track + 1)]
        track_result_list_sub1 = [[] for _ in range(max_track + 1)]
        track_result_list_sub2 = [[] for _ in range(max_track + 1)]

        for track_id in range(1, len(track_result_list_main)):
            frame_low = max(1, int(gt_breakpoint_list[track_id]) - cfg.SDA.JUDGE_LEN)
            frame_high = min(int(gt_breakpoint_list[track_id]) + cfg.SDA.JUDGE_LEN, len(eval_data))

            for i, frame_id in enumerate(range(frame_low, frame_high)):
                gt_main = []
                gt_sub1 = []
                gt_sub2 = []
                tracks, gts = eval_data[frame_id]
                for line in gts:
                    main_id, sub_id, x_gt, y_gt = line
                    if main_id == track_id:
                        if sub_id == 0:
                            gt_main = [x_gt, y_gt]
                        elif sub_id == 1:
                            gt_sub1 = [x_gt, y_gt]
                        elif sub_id == 2:
                            gt_sub2 = [x_gt, y_gt]

                if len(gt_main) > 0:
                    min_dist = cfg.SDA.MAX_DIST
                    match_track_id = 0
                    for line in tracks:
                        main_id, sub_id, x_track, y_track = line
                        if sub_id == 0:
                            obj_dist = euclid_dist(gt_main[0], gt_main[1], x_track, y_track)
                            if obj_dist <= min_dist:
                                match_track_id = main_id
                                min_dist = obj_dist

                    track_result_list_main[track_id].append(int(match_track_id))

                if len(gt_sub1) > 0:
                    min_dist = cfg.SDA.MAX_DIST
                    match_track_id = 0
                    for line in tracks:
                        main_id, sub_id, x_track, y_track = line
                        if sub_id != 0:
                            obj_dist = euclid_dist(gt_sub1[0], gt_sub1[1], x_track, y_track)
                            if obj_dist <= min_dist:
                                match_track_id = main_id
                                min_dist = obj_dist

                    track_result_list_sub1[track_id].append(int(match_track_id))

                if len(gt_sub2) > 0:
                    min_dist = cfg.SDA.MAX_DIST
                    match_track_id = 0
                    for line in tracks:
                        main_id, sub_id, x_track, y_track = line
                        if sub_id != 0:
                            obj_dist = euclid_dist(gt_sub2[0], gt_sub2[1], x_track, y_track)
                            if obj_dist <= min_dist:
                                match_track_id = main_id
                                min_dist = obj_dist

                    track_result_list_sub2[track_id].append(int(match_track_id))


        success_num = 0

        for i, (main_list, sub1_list, sub2_list) in enumerate(zip(track_result_list_main, track_result_list_sub1,
                                                      track_result_list_sub2), start=1):
            main_id, main_cnt = find_most_common_element(main_list)
            sub1_id, sub1_cnt = find_most_common_element(sub1_list)
            sub2_id, sub2_cnt = find_most_common_element(sub2_list)

            if main_id == sub1_id and main_id == sub2_id:
                if (main_cnt + min(sub1_cnt, sub2_cnt)) >= cfg.SDA.SUCCESS_FRAME:
                    success_num += 1

        total_tracks += max_track
        total_success += success_num
        print('evaluation of video {} finished! sda: {:.3f}'.format(file_name, success_num / max_track))

    print('evaluation finished! total sda: {:.3f}'.format(total_success / total_tracks))


