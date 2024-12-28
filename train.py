import random
import time
import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from yacs.config import CfgNode as CN
from tracking_toolbox import *
from scatter_model import ScatterNet
from scatter_datasets import ScatterObjectDataset, GroundTruthTracks, ScatterTrackDataset
from scatter_tracker import Track
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


class PredictedTrack:
    def __init__(self, track, pred, frame_id, rewind_num):
        '''
        data format:
        :param track: a list containing item [frame_id, x, y, w, h]
        :param pred: an array [track_id, x, y, w, h]
        :param frame_id: serial number of current frame
        :param rewind_num: length of track
        '''

        # 该结构整合轨迹信息并为后续上下分支的标签制作提供信息。结构接收某条轨迹的当前帧号、
        # 当前帧位置和GroundTruthTracks给出的这条完整轨迹，以及需要的历史帧数。该结构的
        # get_track()方法会输出一个以当前帧向前回溯固定帧数的历史轨迹包，若历史帧数不足使
        # 用方法linear_interpolation补充。

        self.track = track
        self.pred = pred
        self.frame_id = int(frame_id)
        self.track_id = int(pred[0])
        self.rewind_num = rewind_num

    def get_id(self):
        return self.track_id

    def get_track(self):
        predicted_track = [[] for _ in range(self.rewind_num)]
        predicted_track[0] = self.pred[1:]
        for position in self.track[::-1]:
            if self.frame_id - position[0] >= self.rewind_num:
                break
            elif int(self.frame_id - position[0]) >= 1:
                predicted_track[int(self.frame_id - position[0])] = position[1:]

        predicted_track = linear_interpolation(predicted_track)
        return predicted_track


def main(args):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    default_yaml = 'config/scatternet.yaml'
    default_cfg = open(default_yaml)
    cfg = CN.load_cfg(default_cfg)
    
    if args.device:
        cfg.LOAD.DEV = 'cuda:{}'.format(args.device)
    if args.checkpoint:
        cfg.LOAD.MODEL = args.checkpoint
    cfg.freeze()

    time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    save_dir = os.path.join(cfg.TRAIN.SAVE_DIR, time_name)
    mkdirs(save_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.log'
    log_file = os.path.join(save_dir, log_file_name)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logger.info('--------------------------1. initialize---------------------------')
    logger.info('cfg file: {}'.format(default_yaml))
    if cfg.PARAM.SEED:
        random_seed = cfg.PARAM.SEED
        random_initiate(seed=random_seed)
        logger.info('random seed initiated {}'.format(random_seed))
    device = cfg.LOAD.DEV
    data_path = cfg.DATA.IMG_ROOT
    video_names = os.listdir(data_path)
    logger.info('found {} videos: {}'.format(len(video_names), video_names))

    track_model = ScatterNet(cfg, logger=logger, is_eval=False).to(device)
    
    hist_epoch = 0
    if cfg.LOAD.MODEL != 'None':
        checkpoint = torch.load(cfg.LOAD.MODEL)
        track_model.load_state_dict(checkpoint['state_dict'], strict=False)
        hist_epoch = checkpoint['epoch']
        logger.info('loaded track model from {}, epoch start from {}'.format(cfg.LOAD.MODEL, hist_epoch))
    else:
        logger.info('loaded a new track model without training!')

    loss_function = nn.BCELoss()
    det_loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(track_model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONE, gamma=cfg.TRAIN.GAMMA)

    logger.info('generating gt tracks......')
    all_gt_tracks = GroundTruthTracks(video_names, cfg)
    logger.info('gt tracks generated!')
    rep_interval = cfg.OUT.REP_INT

    mkdirs(cfg.TRAIN.SAVE_DIR)
    logger.info('preparation finished!')
    time_start = time.time()

    logger.info('--------------------------2. data preparation---------------------------')

    datasets_for_model = {}
    for video_num, video in enumerate(video_names):
        if int(video[1:]) > len(video_names) * cfg.TRAIN.TRAIN_SET_RATIO:   
            continue
        frame_data = ScatterObjectDataset(cfg, video)
        tracks = []
        track_id_list = -1 * np.ones(cfg.PARAM.MAX_TRACK)
        loss_list = []
        frame_start = 1
        frame_end = len(frame_data) + 1

        # 取出当前视频的gt轨迹
        gt_tracks = all_gt_tracks[video]

        # 训练时模拟跟踪器的迭代过程，每个训练数据包提供某帧时所有参与关联的信息，以及计算loss时需要的gt。各变量存储信息如下：
        # track_positions_all：尺寸[frame, max_num, rewind_len, 2]，存储所有帧中每帧包含的轨迹信息；（模型输入）
        # detections_all：尺寸[frame, max_num, 2]，存储所有帧中每帧给定的检测信息；（模型输入）
        # pred_track_list_all：frame个长度为max_num的list，按顺序存储当前帧所有轨迹的轨迹号；（用于生成上分支gt）
        # det_track_list_all：frame个长度为max_num的list，按顺序存储当前帧的检测结果对应的是哪一个轨迹。（用于生成上分支gt）
        # det_group_result_all：frame个尺寸为[max_num, max_num]矩阵的list，存储应当被划定为从属同一目标的检测。（下分支gt） 

        # 对于上分支（跟踪任务），关联模型应当给出一个det_track_list_all所有有效检测与pred_track_list_all所有
        # 有效轨迹之间的相似度矩阵，对于每一个prediction和每一个detection，应当可以在这个矩阵中找到一个
        # 关联相似度。训练时将上分支作为一个二分类任务训练，通过数据包中的pred_track_list_all和det_track_list_all，
        # 可以知道某个prediction和detection是否应当关联，关联则分类为1，否则为0，而模型会给出一个0到1之间的关联相似度。
        # 在预测时，上分支在关联矩阵接受下分支修正后，将相似度反向作为关联代价，使用线性分配算法给出唯一的轨迹-检测分配。

        pred_track_list_all = [pad_list([], cfg.PARAM.MAX_TGT)]  # 提前放入一个空值，因为第一帧不输入
        det_track_list_all = [pad_list([], cfg.PARAM.MAX_DET)]
        track_positions_tensor_all = [torch.zeros(1, cfg.PARAM.TRACK_REWIND, 2)]
        detections_tensor_all = [torch.zeros(1, 2)]
        det_group_result_all = [torch.zeros(cfg.PARAM.MAX_DET, cfg.PARAM.MAX_DET)]
        scatter_age_list = np.zeros(cfg.PARAM.MAX_MAIN_TRACK + 1)

        for frame_id in range(frame_start, frame_end):
            detections, raw_gts = frame_data[frame_id]
            predictions = []
            attached_track_list = []

            # 重组gt，其中gts为每帧的track更新提供数据，因此主轨迹数据会被抄送到两个子轨迹；
            # gts_for_allocate为当前帧检测和gt之间的对应关系提供gt位置信息，因此只保留唯一轨迹号。
            # 轨迹号由两部分组成，假设主轨迹号为5，子轨迹号1，唯一轨迹号会被按照{}{:02d}的格式命名为501.

            gts = []
            gts_for_allocate = []
            for gt in raw_gts:
                main_id, frag_id, x, y = gt
                track_id = int('{}{:02d}'.format(int(main_id), int(frag_id)))
                frag_1 = int('{}{:02d}'.format(int(main_id), 1))
                frag_2 = int('{}{:02d}'.format(int(main_id), 2))
                attached_track_list.append(track_id)
                if frag_id == 0:
                    gts.append([track_id, x, y])
                    gts.append([frag_1, x, y])
                    gts.append([frag_2, x, y])
                else:
                    gts.append([track_id, x, y])
                    scatter_age_list[main_id] += 1

                gts_for_allocate.append([track_id, x, y])

            # 这一部分使用Track结构来维护构建数据包时的gt轨迹，对于每一帧，算法输入所有的目标位置来为轨迹提供
            # 当前坐标，同时给出需要的轨迹预测结果，作为训练时的prediction。需注意的是，被抄送的子轨迹在轨迹
            # 坐标更新时应当被输入，但由于这并不是当前帧真实存在的轨迹，在预测时这一部分轨迹号不应给出预测结果。
            # attached_track_list：存储当前帧所有应当给出预测的轨迹号。
            # 结构Track通过update()方法完成卡尔曼滤波对于轨迹坐标的一次更新，通过predict()方法完成卡尔曼滤波
            # 给出的预测结果。每次对轨迹更新必须调用update()方法，但可以不调用predict()方法。

            # 找到所有刚刚分裂的主轨迹号，记录其子轨迹号
            critical_indices = np.where((scatter_age_list >= 1) & (scatter_age_list <= cfg.PARAM.SCATTER_PERIOD))[0]
            sub_tracks = []
            for ind in critical_indices:
                sub_track1 = int('{}{:02d}'.format(ind, 1))
                sub_track2 = int('{}{:02d}'.format(ind, 2))
                sub_tracks.append([sub_track1, sub_track2])

            for track in tracks:
                track.kf.predict()
                if track.track_id in attached_track_list:
                    predictions.append([track.track_id] + [float(d) for d in track.kf.x[:2]])
            for gt in gts:
                track_id, x, y = gt
                if track_id_list[track_id] == -1:
                    track_id_list[track_id] = len(tracks)
                    track = Track([x, y], track_id, cfg=cfg)
                    tracks.append(track)
                else:
                    tracks[int(track_id_list[track_id])].kf.update([x, y])
            if frame_id == frame_start:
                continue

            # 这一部分计算检测目标和预测位置之间的分配关系，同时将各种信息填充到每帧的数据包中。算法通过
            # allocate_detections_for_gt分配检测和预测之间的对应关系，具体来说，算法通过欧氏距离结合
            # 线性分配算法来分配目标的匹配对，这一做法为检测目标赋予了对应的轨迹真值。需注意的是，本任务中
            # 目标只考虑中心位置，在正常的跟踪任务中，这一算法兼具计算IoU和欧氏距离两个匹配代价。config
            # 文件中具有此部分的分配阈值，超过一定欧氏距离或低于一定IoU的两个目标一定不会关联。对于MOT17
            # 等数据集，对此处的gt构建需要慎重，因为这一分配方式并不一定完全准确，实验表明不同的阈值参数
            # 会极大幅度影响训练效果。
            # 完成计算后，为打包batch，算法将各种数据的维度通过tensor_supple, pad_list等函数填充统一。

            length_limit = cfg.PARAM.MAX_LEN
            predicted_tracks = []
            for prediction in predictions:
                predicted_tracks.append(PredictedTrack(
                    gt_tracks[int(prediction[0])], prediction, frame_id, cfg.PARAM.TRACK_REWIND))

            track_positions = []
            pred_track_list = []
            det_group_result = torch.zeros(cfg.PARAM.MAX_DET, cfg.PARAM.MAX_DET)
            det_track_list = allocate_detections_for_gt(detections, gts_for_allocate, cfg)

            for sub_track in sub_tracks:
                det_direct = np.array(det_track_list)
                sub1, sub2 = sub_track
                sub1_pos = np.where(det_direct == sub1)[0]
                sub2_pos = np.where(det_direct == sub2)[0]
                if len(sub1_pos) > 0 and len(sub2_pos) > 0:
                    det_group_result[sub1_pos[0], sub2_pos[0]] = 1
                    det_group_result[sub2_pos[0], sub1_pos[0]] = 1

            for pred in predicted_tracks:
                pred_track_list.append(pred.get_id())
                track_position = np.array(pred.get_track())
                track_position[track_position >= length_limit] = length_limit - 1
                track_positions.append(track_position.tolist())

            track_positions_tensor = torch.tensor(track_positions, dtype=torch.float32)
            track_positions_tensor[track_positions_tensor >= length_limit] = length_limit - 1

            detections_tensor = torch.tensor(detections, dtype=torch.float32)
            detections_tensor[detections_tensor >= length_limit] = length_limit - 1

            pred_track_list = pad_list(pred_track_list, cfg.PARAM.MAX_TGT)
            det_track_list = pad_list(det_track_list, cfg.PARAM.MAX_DET)
            pred_track_list_all.append(pred_track_list)
            det_track_list_all.append(det_track_list)
            det_group_result_all.append(det_group_result)
            track_positions_tensor_all.append(track_positions_tensor)
            detections_tensor_all.append(detections_tensor)

        track_positions_suppled = tensor_supple(track_positions_tensor_all, cfg.PARAM.MAX_TGT,
                                                pad_value=cfg.PARAM.MAX_LEN + 1)
        detections_suppled = tensor_supple_2d(detections_tensor_all, cfg.PARAM.MAX_DET,
                                              pad_value=cfg.PARAM.MAX_LEN + 1)
        track_positions_all = torch.stack(track_positions_suppled)
        detections_all = torch.stack(detections_suppled)
        det_group_result_all = torch.stack(det_group_result_all)

        if len(pred_track_list_all) != len(det_track_list_all):
            raise AssertionError('length of pred_track_list_all ({}) and det_track_list_all ({})'
                                 'should be same'.format(len(pred_track_list_all), len(det_track_list_all)))

        dataset_for_video = ScatterTrackDataset(pred_track_list_all, det_track_list_all,
                                                track_positions_all, detections_all, det_group_result_all, cfg)
        datasets_for_model.update({video: dataset_for_video})
        logger.info('training data for video {} generated!'.format(video))

    logger.info('--------------------------3. training---------------------------')

    best_loss = [99999, 0]

    for epoch in range(1, cfg.TRAIN.EPOCH + 1):
        process_num = 0
        if epoch <= hist_epoch:
            continue
        logger.info('epoch {} started!'.format(epoch))
        time_org = time.time()
        all_asso_loss_list = []
        all_group_loss_list = []
        all_total_loss_list = []
        # 打乱视频训练顺序
        random.shuffle(video_names)
        for video_num, video in enumerate(video_names):
            if int(video[1:]) > len(video_names) * cfg.TRAIN.TRAIN_SET_RATIO:  
                continue
            else:
                process_num += 1
            logger.info('processing video {}'.format(video))

            asso_loss_list = []
            group_loss_list = []
            total_loss_list = []
            dataset = datasets_for_model[video]
            lingx_dataloader = DataLoader(dataset, batch_size=cfg.PARAM.BATCH_SIZE,
                                          shuffle=True, num_workers=cfg.PARAM.WORKERS, drop_last=True)

            for batch in lingx_dataloader:
                batch_asso_loss = torch.zeros((1, 1), requires_grad=True).to(device)
                batch_group_loss = torch.zeros((1, 1), requires_grad=True).to(device)
                batch_total_loss = torch.zeros((1, 1), requires_grad=True).to(device)
                pred_track_lists, det_track_lists, track_positions, detections, det_group_results, end_marks = batch
                pred_track_lists = torch.stack(pred_track_lists).permute(1, 0)
                det_track_lists = torch.stack(det_track_lists).permute(1, 0)
                match_scores, group_scores, match_results = track_model(track_positions, detections, end_marks)

                tracklet_marks, det_marks = end_marks
                for batch_pos, (
                match_score, match_result, pred_track_list_all, det_track_list_all, track_position, detection,
                tracklet_single_mark, det_single_mark, group_score, group_result) in \
                        enumerate(zip(match_scores, match_results, pred_track_lists, det_track_lists, track_positions,
                                      detections, tracklet_marks, det_marks, group_scores, det_group_results)):
                    tracklet_mark, detection_mark = end_marks
                    pred_track_list = pred_track_list_all[:tracklet_mark[batch_pos]]
                    det_track_list = det_track_list_all[:detection_mark[batch_pos]]
                    gt_allocation = torch.zeros(len(pred_track_list), dtype=torch.int64).to(device)
                    for i in range(len(pred_track_list)):
                        for j in range(len(det_track_list)):
                            if det_track_list[j] == pred_track_list[i]:
                                gt_allocation[i] = j
                                break
                            gt_allocation[i] = len(det_track_list)

                    if len(match_result) == 0 or len(gt_allocation) == 0:
                        continue

                    # 新gt
                    _, max_index = match_score.shape
                    gt_matrix = torch.zeros_like(match_score).to(device)
                    # 根据给定的索引值，在每一行中将对应位置置为1
                    for i, index in enumerate(gt_allocation):
                        if index < max_index:
                            gt_matrix[i][index] = 1


                    match_score_for_train = score_assignment(match_score, device=device)
                    group_score_for_train = group_score


                    asso_loss = loss_function(match_score_for_train, gt_matrix)
                    group_loss = det_loss_function(group_score_for_train, group_result[:det_single_mark, :det_single_mark])
                    total_loss = cfg.TRAIN.A_LOSS * asso_loss + cfg.TRAIN.G_LOSS * group_loss
                    batch_asso_loss += asso_loss
                    batch_group_loss += group_loss
                    batch_total_loss += total_loss

                asso_loss_list.append(batch_asso_loss)
                group_loss_list.append(batch_group_loss)
                total_loss_list.append(batch_total_loss)
                all_asso_loss_list.append(batch_asso_loss)
                all_group_loss_list.append(batch_group_loss)
                all_total_loss_list.append(batch_total_loss)
                batch_total_loss.requires_grad_(True)
                batch_total_loss.backward()
                optimizer.step()

            max_memory = torch.cuda.max_memory_allocated(device=device)

            current_lr = optimizer.param_groups[0]['lr']
            asso_loss_for_log = (sum(asso_loss_list) / len(asso_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
            group_loss_for_log = (sum(group_loss_list) / len(group_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
            total_loss_for_log = (sum(total_loss_list) / len(total_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
            logger.info(
                'epoch {} finished {:.2f}%, processed video {}, association loss: {:.4f}, grouping loss: {:.4f}, '
                'total loss: {:.4f}, lr:{:.4f}, '
                'epoch used time: {}min {} sec, total used time: {}h {}min {}sec, max memory allocated: {:.2f} MB'.format(
                    epoch,
                    100 * process_num / (len(video_names) * cfg.TRAIN.TRAIN_SET_RATIO),
                    video,
                    asso_loss_for_log,
                    group_loss_for_log,
                    total_loss_for_log,
                    current_lr,
                    int((time.time() - time_org) / 60), int(time.time() - time_org) % 60,
                    int((time.time() - time_start) / 3600), int(((time.time() - time_start) / 60) % 60),
                    int((time.time() - time_start) % 60),
                    max_memory / 1024 ** 2))


        epoch_asso_loss = (sum(all_asso_loss_list) / len(all_asso_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
        epoch_group_loss = (sum(all_group_loss_list) / len(all_group_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
        epoch_total_loss = (sum(all_total_loss_list) / len(all_total_loss_list)).cpu().item() / cfg.PARAM.BATCH_SIZE
        logger.info(
            'epoch {} finished! total association loss: {:.4f}, total grouping loss: {:.4f}, total loss: {:.4f}, '
            'epoch used time: {}h {}min {}sec, total used time: {}h {}min {}sec'.format(
                epoch,
                epoch_asso_loss,
                epoch_group_loss,
                epoch_total_loss,
                int((time.time() - time_org) / 3600), int(((time.time() - time_org) / 60) % 60),
                int((time.time() - time_org) % 60),
                int((time.time() - time_start) / 3600), int(((time.time() - time_start) / 60) % 60),
                int((time.time() - time_start) % 60))
        )

        if epoch_total_loss <= best_loss[0]:
            best_loss[0] = epoch_total_loss
            best_loss[1] = epoch

        scheduler.step()

        if best_loss[1] == epoch:
            save_name = os.path.join(save_dir, 'best_loss.pth')
            checkpoint = {
                'state_dict': track_model.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, save_name)
            logger.info('model of best loss refreshed at epoch {}!'.format(epoch))
        else:            
            logger.info('Now model of best loss is at epoch {}!'.format(best_loss[1]))

        for ckpt in cfg.TRAIN.CKPT:
            if epoch == ckpt:
                save_path = save_dir
                save_name = os.path.join(save_path, 'epoch_{}.pth'.format(ckpt))
                checkpoint = {
                    'state_dict': track_model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, save_name)
                logger.info('checkpoint saved to {} at epoch {}'.format(save_path, ckpt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate train results')
    parser.add_argument('--checkpoint', '-c', default=None, type=str, help='checkpoint for continue training')
    parser.add_argument('--device', '-d', default='0', type=str, help='device used for training')
    args = parser.parse_args()
    main(args)





