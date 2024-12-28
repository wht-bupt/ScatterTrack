import numpy as np
import torch
from queue import Queue
from filterpy.kalman import KalmanFilter
from scatter_model import ScatterNet
from tracking_toolbox import *
from yacs.config import CfgNode as CN


class TrackBuffer:
    def __init__(self, buffer_size, position_now):
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)

        for _ in range(buffer_size):
            padding_tensor = position_now
            self.queue.put(padding_tensor)

    def __len__(self):
        return self.buffer_size

    @property
    def as_list(self):
        result_list = []
        for i in range(self.buffer_size):
            element = self.queue.queue[i]
            result_list.append(element)

        return result_list

    def update(self, position):
        tensor = torch.tensor(position)

        if self.queue.full():
            self.queue.get()

        self.queue.put(tensor)


class Track:
    def __init__(self, detection, track_id, cfg, fragment_id=0):

        # 轨迹状态在轨迹类中是没有显式定义的，判断方式为：
        # track_id（主轨迹号）> 0即为confirmed;
        # track_id（主轨迹号）< 0即为unconfirmed；
        # is_alive为False的轨迹会被删除。

        self.cfg = cfg
        self.track_id = track_id
        self.fragment_id = fragment_id
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.scatter_hits = 0
        self.scatter_gap = 0
        self.scatter_position1 = []
        self.scatter_position2 = []
        self.dt = 1
        self.is_alive = True
        self.buffer_size = cfg.PARAM.TRACK_REWIND - 1
        self.track_buffer = TrackBuffer(self.buffer_size, torch.Tensor(detection))

        # 创建Kalman Filter对象
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # 系统状态转移矩阵
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])

        # 系统状态噪声协方差矩阵
        self.kf.Q = np.eye(4) * cfg.KF.Q_GAIN

        # 观测矩阵
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])

        # 观测噪声协方差矩阵
        self.kf.R = np.eye(2) * 1

        # 初始化状态向量和协方差矩阵
        self.kf.x = np.array([float(t) for t in detection] + [0., 0.])
        self.kf.P = np.eye(4) * 1. * cfg.KF.P_GAIN


def scatter_distance(tracks, detections, model, cfg):
    if len(tracks) == 0 or len(detections) == 0:
        return [], []
    else:
        tracks_hist = np.clip([[np.array(t) for t in track.track_buffer.as_list[::-1]] for track in tracks], 1, cfg.PARAM.MAX_LEN)
        tracks_now = np.clip([track.kf.x[:2] for track in tracks], 1, cfg.PARAM.MAX_LEN)
        detections = np.clip(torch.tensor(np.array(detections)).unsqueeze(0), 1, cfg.PARAM.MAX_LEN)
        tracks = torch.tensor(np.array([np.concatenate((np.reshape(a, (1, len(a))), b), axis=0) for a, b in
                                        zip(tracks_now, tracks_hist)])).to(cfg.LOAD.DEV).unsqueeze(0)
        end_marks = torch.tensor([[len(tracks[0])], [len(detections[0])]]).to(cfg.LOAD.DEV)

        tracks_suppled = tensor_supple(tracks, cfg.PARAM.MAX_TGT, pad_value=cfg.PARAM.MAX_LEN+1)
        detections_suppled = tensor_supple_2d(detections, cfg.PARAM.MAX_DET, cfg.PARAM.MAX_LEN+1)
        tracks_suppled_tensor = torch.stack(tracks_suppled).to(cfg.LOAD.DEV)
        detections_suppled_tensor = torch.stack(detections_suppled).to(cfg.LOAD.DEV)

        match_score, group_score, _ = model(tracks_suppled_tensor, detections_suppled_tensor, end_marks)
        match_score_np = match_score[0].detach().cpu().numpy()
        group_score_np = group_score[0].detach().cpu().numpy()
        return match_score_np, group_score_np



class ScatterTracker:
    def __init__(self, cfg, logger, distance_method='scatter'):
        self.tracks = []
        self.cfg = cfg
        self.logger = logger
        self.device = cfg.TRACK.DEV
        self.confirm_age = cfg.TRACK.CONFIRM_AGE     # 目标被激活为一条轨迹的确认帧数
        self.scatter_age = cfg.TRACK.SCATTER_AGE     # 目标确认分离所需的连续击中帧数
        self.max_age = cfg.TRACK.MAX_AGE             # 未匹配轨迹最大存活帧数
        self.max_dist = cfg.TRACK.MAX_DIST           # 轨迹-检测的关联代价阈值
        self.print_remain = cfg.TRACK.PRINT_REMAIN   # 轨迹暂时消失后仍旧预测输出的帧数
        self.del_age = cfg.TRACK.DEL_AGE             # 若need_hit=True，输出为轨迹所最少需要的成功匹配次数
        self.scatter_max_gap = cfg.TRACK.MAX_GAP
        self.track_counter = 1
        self.unconfirmed_track_counter = -1
        self.distance_method = distance_method
        self.need_hit = cfg.TRACK.NEED_HIT
        self.model_path = cfg.TRACK.MODEL
        self.track_model = ScatterNet(cfg, logger=logger, is_eval=True).to(self.device)


        checkpoint = torch.load(self.model_path)
        self.track_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.track_model.eval()

    def get_tracks(self):
        return self.tracks

    def update(self, detections, img=None):

        if len(self.tracks) == 0:

            # 首帧初始化，对每个检测结果初始化一条轨迹
            for i in range(len(detections)):
                track = Track(detections[i], self.unconfirmed_track_counter, cfg=self.cfg)
                self.unconfirmed_track_counter -= 1
                self.tracks.append(track)
        else:

            # 更新所有轨迹的位置，取出预测位置
            for track in self.tracks:
                track.kf.predict()
                track.time_since_update += 1
                track.age += 1

            if self.distance_method == 'scatter':
                match_scores, group_scores = scatter_distance(self.tracks, detections, self.track_model, self.cfg)
            else:
                raise ValueError

            if len(match_scores) > 0:

                # 处理从属同一目标概率矩阵
                # mask掉下三角部分，仅除主对角线以外的上三角部分可能出现小于1的值，取出group_scores上三角部分值为1的匹配对
                group_std = np.ones_like(group_scores)
                tri_mask = np.ones_like(group_scores)
                tri_mask[np.triu_indices(tri_mask.shape[0], k=1)] = 0
                masked_group_scores = group_std - group_scores + tri_mask

                # 使用lapjv给出上三角部分的唯一分配，将lapjv给出的分配结果拆成记录行号和列号的两个向量det_group_rows(r)和det_group_cols(c)
                det_group_rows, det_group_cols = lap_assignment(masked_group_scores, cost_limit=self.cfg.TRACK.GROUP_MAX_COST)

                # 分离正确性判别
                # 1.分别mask掉行号集合和列号集合对应的检测，使得被认定为同一目标的两个检测同一时间只能有一个参与匹配
                # 2.判别分离正确性，条件：同组的两个检测匹配到的轨迹必须主轨迹号相同，且子轨迹号为0

                # 处理轨迹-检测匹配代价矩阵，对每一个储存的列号，mask掉其对应的所有轨迹-检测匹配得分，使其不能够与任何轨迹匹配
                # Mask掉所有位于c中的检测，进行线性分配，得到一组轨迹-检测匹配A1：t1-d1
                asso_std = np.ones_like(match_scores)
                group_mask_col = np.zeros_like(match_scores)
                group_mask_col[:, det_group_cols] = 1
                cost_cmask = asso_std - match_scores + group_mask_col

                row_inds_cmask, col_inds_cmask = lap_assignment(cost_cmask, extend_cost=True, cost_limit=self.cfg.TRACK.ASSO_MAX_COST)

                # 使用与上一步相同的方法，mask掉行号对应的匹配
                # Mask掉所有位于r中的检测，进行线性分配，得到一组轨迹-检测匹配A2：t2-d2
                group_mask_row = np.zeros_like(match_scores)
                group_mask_row[:, det_group_rows] = 1
                cost_rmask = asso_std - match_scores + group_mask_row

                row_inds_rmask, col_inds_rmask = lap_assignment(cost_rmask, extend_cost=True, cost_limit=self.cfg.TRACK.ASSO_MAX_COST)

                # 遍历A1的每个t-d，若d不位于r中，正常更新轨迹；若d位于r中，检查t的fragment_id，不为0则把d以及d在c中的对应值d0
                # 都添加到补充列表S中，为0则判断：d在A1中对应的t.track_id和d0在A2中对应的t0.track_id是否相等（这意味着两条子
                # 轨迹关联到了同一条主轨迹上面），若相等则执行带分离计数和分离位置的轨迹更新，若不相等则直接跳过
                supple_dets = []
                used_dets = []
                for row_ind, col_ind in zip(row_inds_cmask, col_inds_cmask):
                    if row_ind >= 0:
                        if col_ind not in det_group_rows:
                            self.tracks[row_ind].kf.update(detections[col_ind])
                            self.tracks[row_ind].track_buffer.update(self.tracks[row_ind].kf.x[:2])
                            self.tracks[row_ind].time_since_update = 0
                            self.tracks[row_ind].scatter_gap += 1
                            self.tracks[row_ind].hits += 1
                            if self.tracks[row_ind].scatter_gap >= self.scatter_max_gap:
                                self.tracks[row_ind].scatter_hits = 0
                            used_dets.append(col_ind)
                            continue

                        else:
                            # 行号：fragment_ind, 列号：col_ind, 列号对应的track：self.tracks[row_ind]
                            group_ind = np.where(det_group_rows == col_ind)[0][0]
                            fragment_ind = det_group_cols[group_ind]
                            det_row_ind_position = np.where(col_inds_rmask == fragment_ind)[0]
                            if len(det_row_ind_position) > 0:
                                det_row_ind_position = det_row_ind_position[0]
                                det_row_ind_track = row_inds_rmask[det_row_ind_position]
                                if self.tracks[det_row_ind_track].track_id == self.tracks[row_ind].track_id and \
                                    self.tracks[row_ind].fragment_id == 0:
                                    self.tracks[row_ind].scatter_position1 = detections[col_ind]
                                    self.tracks[row_ind].scatter_position2 = detections[fragment_ind]
                                    self.tracks[row_ind].kf.update(detections[col_ind])
                                    self.tracks[row_ind].track_buffer.update(self.tracks[row_ind].kf.x[:2])
                                    self.tracks[row_ind].time_since_update = 0
                                    self.tracks[row_ind].hits += 1
                                    self.tracks[row_ind].scatter_hits += 1
                                    self.tracks[row_ind].scatter_gap = 0
                                    used_dets.append(col_ind)
                                    used_dets.append(fragment_ind)
                                    continue

                            supple_dets.append(fragment_ind)
                            supple_dets.append(col_ind)

                # 重新取出轨迹-检测匹配代价矩阵，mask掉所有已经完成匹配的轨迹与所有子轨迹号为0的轨迹，以及所有不在列号副本中的检测，进行二次匹配（存疑：是否应该mask子轨迹号为0的轨迹？）
                supple_track_mask_list = []
                for i, track in enumerate(self.tracks):
                    if track.time_since_update == 0:
                        supple_track_mask_list.append(i)

                supple_track_mask = np.zeros_like(match_scores)
                supple_track_mask[supple_track_mask_list, :] = 1
                supple_det_mask = np.ones_like(match_scores)
                supple_det_mask[:, supple_dets] = 0
                supple_cost = asso_std - match_scores + supple_track_mask + supple_det_mask

                supple_rows, supple_cols = lap_assignment(supple_cost, extend_cost=True, cost_limit=self.cfg.TRACK.ASSO_MAX_COST)

                for supple_row, supple_col in zip(supple_rows, supple_cols):
                    if supple_row >= 0:
                        self.tracks[supple_row].kf.update(detections[supple_col])
                        self.tracks[supple_row].track_buffer.update(self.tracks[supple_row].kf.x[:2])
                        self.tracks[supple_row].time_since_update = 0
                        self.tracks[supple_row].hits += 1
                        self.tracks[supple_row].scatter_gap += 1
                        if self.tracks[supple_row].scatter_gap >= self.scatter_max_gap:
                            self.tracks[supple_row].scatter_hits = 0
                        used_dets.append(supple_col)

            # 更新未匹配到目标的轨迹状态，处理可以被转化为confirmed的轨迹，以及处理应当分离的轨迹
            for track in self.tracks:
                if track.time_since_update != 0:
                    track.kf.update(track.kf.x[:2])
                    track.track_buffer.update(track.kf.x[:2])
                    track.hits = 0
                    track.scatter_gap += 1
                    if track.scatter_gap >= self.scatter_max_gap:
                        track.scatter_hits = 0
                    if track.time_since_update > self.max_age:
                        track.is_alive = False

                elif track.track_id < 0 and track.hits > self.confirm_age:
                    track.track_id = self.track_counter
                    self.track_counter += 1

            # 分离轨迹处理：删除主轨迹，初始化两条confirmed状态的子轨迹，位置在分离计数时已经存储，主轨迹号小于0的时候不会处理
                elif track.scatter_hits >= self.scatter_age:
                    track.is_alive = False
                    scatter_track1 = Track(track.scatter_position1, track.track_id, self.cfg, fragment_id=1)
                    scatter_track2 = Track(track.scatter_position2, track.track_id, self.cfg, fragment_id=2)
                    self.tracks.append(scatter_track1)
                    self.tracks.append(scatter_track2)

            # 为未匹配到轨迹的检测建立unconfirmed状态的新轨迹
            for i, det in enumerate(detections):
                if i not in used_dets:
                    track = Track(detections[i], self.unconfirmed_track_counter, cfg=self.cfg)
                    self.unconfirmed_track_counter -= 1
                    self.tracks.append(track)

            # 删除is_alive状态为False的轨迹
            self.tracks = [x for x in self.tracks if x.is_alive]

        if self.need_hit:
            tracks_print = [x for x in self.tracks if x.time_since_update <= self.print_remain and x.hits >= self.del_age and x.track_id > 0]
        else:
            tracks_print = [x for x in self.tracks if x.time_since_update <= self.print_remain and x.track_id > 0]
        return tracks_print








