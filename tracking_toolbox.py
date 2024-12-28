import random
import os
import cv2
import numpy as np
import torch
import copy
import lap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def random_initiate(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iou(rec1_org, rec2_org):
    # 计算iou
    # 输入为[x, y, w, h]格式
    rec1 = copy.deepcopy(rec1_org)
    rec2 = copy.deepcopy(rec2_org)
    rec1[2] = rec1[2] + rec1[0]
    rec2[2] = rec2[2] + rec2[0]
    rec1[3] = rec1[3] + rec1[1]
    rec2[3] = rec2[3] + rec2[1]
    s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    sum_area = s_rec1 + s_rec2
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def assure_unzero(matrix, device):
    len1, len2 = matrix.shape
    add_mat = torch.zeros((len1, len2)).to(device)
    for i in range(len1):
        for j in range(len2):
            if matrix[i][j] < 1e-9:
                add_mat[i][j] = 1e-9

    return matrix + add_mat


def bbox_dist(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xc1 = x1+w1/2
    yc1 = y1+h1/2
    xc2 = x2+w2/2
    yc2 = y2+h2/2
    return np.sqrt((xc1-xc2)**2+(yc1-yc2)**2)


def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]


def pos_int(detection, frame):
    h_frame, w_frame, _ = frame.shape
    x1, y1, x2, y2 = detection
    w = x2 - x1
    h = y2 - y1
    w = max(round(w), 1)
    h = max(round(h), 1)
    x1 = max(round(x1), 1)
    x1 = min(x1, w_frame - 1)
    y1 = max(round(y1), 1)
    y1 = min(y1, h_frame - 1)
    x2 = min(x1 + w, w_frame)
    y2 = min(y1 + h, h_frame)
    return [x1, y1, x2, y2]


def assure_kf_pos(kf_x):
    x, y, w, h, dx, dy, dw, dh = kf_x
    if w < 1:
        w = 1
        dw = 0
    if h < 1:
        h = 1
        dh = 0
    return [x, y, w, h, dx, dy, dw, dh]


def cv2_draw_box(img, tracks):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_map = [[255, 255, 0], [255, 127, 127], [0, 0, 255],
                 [255, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 255], [0, 127, 127], [127, 127, 0], [127, 0, 127]]
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    for track in tracks:
        track_id, x, y, w, h = track[1:]
        color = color_map[(track_id-1) % 10]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        text_size, _ = cv2.getTextSize(str(track_id), font, font_scale, thickness)
        cv2.putText(img, str(track_id), (x, y), font, font_scale, (0, 255, 0), thickness)
    return img



def assignment_3D(result_1, result_2):
    ids_1 = [row[0] for row in result_1]
    ids_2 = [row[0] for row in result_2]
    assignment_matrix = np.zeros((len(result_1), len(result_2)))
    for i in range(len(ids_1)):
        for j in range(len(ids_2)):
            if (ids_1[i] == 3 and ids_2[j] == 2) or (ids_1[i] == 1 and ids_2[j] == 1) or \
                    (ids_1[i] == 2 and ids_2[j] == 3):
                assignment_matrix[i][j] = 0
            else:
                assignment_matrix[i][j] = 99

    id1, id2 = linear_sum_assignment(assignment_matrix)
    return id1, id2


def convert_xywh_to_xy(positions):
    if len(positions) == 0:
        return np.zeros((0, 0))
    if not isinstance(positions, torch.Tensor):
        positions = torch.stack(positions)
    positions_xy = np.array(positions)[:, :2]
    positions_wh = np.array(positions)[:, 2:]
    positions = positions_xy + 0.5 * positions_wh
    return positions


def convert_xy_to_np(positions):
    if not isinstance(positions, torch.Tensor):
        positions = torch.stack(positions)
    positions = positions.numpy()
    return positions


def self_adjacent_matrix(positions, cfg):
    if len(positions) == 0:
        return np.zeros((0, 0))

    positions = convert_xy_to_np(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    result_matrix = np.where(distances < cfg.ASSO.MAX_DIST, 1, 0)
    return result_matrix


def cross_adjacent_matrix(predictions, detections, cfg):
    if len(predictions) == 0 or len(detections) == 0:
        return np.zeros((0, 0))

    predictions = convert_xy_to_np(predictions)
    detections = convert_xy_to_np(detections)
    diff = predictions[:, np.newaxis, :] - detections[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    result_matrix = np.where(distances < cfg.ASSO.MAX_DIST, 1, 0)
    return result_matrix


def split_and_concatenate(input_tensor, split_sizes):
    result_list = []
    start = 0
    size_sum = sum(split_sizes)
    if size_sum != len(input_tensor):
        raise ValueError('size sum {}, input {}'.format(size_sum, len(input_tensor)))
    for size in split_sizes:
        result_list.append(input_tensor[start:start+size])
        start += size
    return result_list


def pad_list(lst_org, pad_len=100, pad_value=-10):
    lst = copy.copy(lst_org)
    if len(lst) < pad_len:
        lst += [pad_value] * (pad_len - len(lst))
        return lst
    else:
        raise ValueError('max supple num {} not enough, got {}'.format(pad_len, len(lst)))


def allocate_detections_for_gt(detections, gts, cfg):
    iou_dist_list = np.zeros((len(detections), len(gts)))
    assignment = np.empty((0, 2), dtype=int)

    for i, detection in enumerate(detections):
        for j, gt in enumerate(gts):
            real_euclid_dist = np.linalg.norm(np.array(detection) - np.array(gt[1:]))
            euclid_dist = 999 if real_euclid_dist >= cfg.GT_ASSO.MAX_DIST else real_euclid_dist
            iou_dist_list[i][j] = euclid_dist
    if iou_dist_list.size:
        _, assignment, _ = lap.lapjv(iou_dist_list, extend_cost=True, cost_limit=cfg.GT_ASSO.MAX_DIST)
    output_detections = []

    # 检查输入输出维度是否相等，给没分配到gt的检测补-1标签
    for i in range(len(detections)):
        if assignment[i] == -1:
            output_detections.append(-1)
        else:
            output_detections.append(gts[assignment[i]][0])

    return output_detections


def allocate_detections(detections, gts, cfg):
    iou_dist_list = np.zeros((len(detections), len(gts)))
    assignment = np.empty((0, 2), dtype=int)

    for i, detection in enumerate(detections):
        for j, gt in enumerate(gts):
            real_euclid_dist = np.linalg.norm(np.array(detection) - np.array(gt[1:]))
            euclid_dist = 999 if real_euclid_dist >= cfg.GT_ASSO.MAX_DIST else real_euclid_dist
            iou_dist_list[i][j] = euclid_dist
    if iou_dist_list.size:
        _, assignment, _ = lap.lapjv(iou_dist_list, extend_cost=True, cost_limit=1)
    output_detections = []

    # 检查输入输出维度是否相等，给没分配到gt的检测补-1标签
    for i in range(len(detections)):
        if assignment[i] == -1:
            output_detections.append(-1)
        else:
            output_detections.append(gts[assignment[i]][0])

    return output_detections


def target_supple(targets, supple_num):
    # 输入：一个维度未对齐的二维list
    suppled_targets = []
    for target in targets:
        if len(target) > supple_num:
            raise AssertionError('max supple num {} not enough, got {}'.format(supple_num, len(target)))
        else:
            line = target + [-10 for _ in range(supple_num - len(target))]
            suppled_targets.append(line)

    return suppled_targets


def tensor_supple(targets, supple_num, pad_value=2001):
    # 输入：一个维度未对齐的三维tensor列表，对齐第一个维度，用-1补齐
    suppled_targets = []
    for target in targets:
        x_len, _, _ = target.shape
        supple_len = supple_num - x_len
        if supple_len < 0:
            raise ValueError('max supple num {} not enough, got {}'.format(supple_num, x_len))
        else:
            padded_tensor = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, supple_len), value=pad_value)
            suppled_targets.append(padded_tensor)

    return suppled_targets

def tensor_supple_2d(targets, supple_num, pad_value=2001):
    # 输入：一个维度未对齐的二维tensor列表，对齐第一个维度，用-1补齐
    suppled_targets = []
    for target in targets:
        x_len, _ = target.shape
        supple_len = supple_num - x_len
        if supple_len < 0:
            raise ValueError('max supple num {} not enough, got {}'.format(supple_num, x_len))
        else:
            padded_tensor = torch.nn.functional.pad(target, (0, 0, 0, supple_len), value=pad_value)
            suppled_targets.append(padded_tensor)

    return suppled_targets


def linear_interpolation(predicted_track):
    # 从后向前找到第一个非空列表
    last_non_empty_idx = len(predicted_track) - 1
    while last_non_empty_idx >= 0 and not predicted_track[last_non_empty_idx]:
        last_non_empty_idx -= 1

    if last_non_empty_idx < 0:
        # 所有列表都为空，无法进行插值
        return []

    track_length = len(predicted_track)
    interpolated_track = predicted_track

    # 复制最后一个非空列表的值到相应位置
    interpolated_track[last_non_empty_idx] = predicted_track[last_non_empty_idx]

    # 从最后一个非空列表向前遍历进行线性插值
    for i in range(last_non_empty_idx - 1, -1, -1):
        if not predicted_track[i]:
            # 遇到空列表，找到缺省值片段的头尾索引
            start_idx = i
            while start_idx >= 0 and not predicted_track[start_idx]:
                start_idx -= 1
            end_idx = i + 1
            while end_idx < track_length and not predicted_track[end_idx]:
                end_idx += 1

            # 使用线性插值补充缺省值
            start_val = predicted_track[start_idx]
            end_val = predicted_track[end_idx]
            for j in range(start_idx + 1, end_idx):
                alpha = (j - start_idx) / (end_idx - start_idx)
                interpolated_val = [
                    start_val[k] + alpha * (end_val[k] - start_val[k])   # 需要取整的话在这一行加int
                    for k in range(len(start_val))
                ]
                interpolated_track[j] = interpolated_val

    supp_len = len(predicted_track) - last_non_empty_idx - 1
    interpolated_track = interpolated_track[:last_non_empty_idx+1] + [[-1, -1] for _ in range(supp_len)]
    return interpolated_track


def save_lookup_table(epoch, save_path, model, cfg):
    interval = cfg.TRAIN.SAVE_EMBED_INT
    if interval > 0:
        if epoch % interval == 1 or interval == 1:
            fig_save_path = os.path.join(save_path, 'figures/epoch_{}'.format(epoch))
            mkdirs(fig_save_path)
            x_path = os.path.join(fig_save_path, 'lookup_table_x.png')
            y_path = os.path.join(fig_save_path, 'lookup_table_y.png')
            w_path = os.path.join(fig_save_path, 'lookup_table_w.png')
            h_path = os.path.join(fig_save_path, 'lookup_table_h.png')

            lookup_table_x = model.feature_extraction.lookup_table.lookup_table_x.cpu().detach().numpy()
            make_fig_from_embedding(x_path, lookup_table_x)
            lookup_table_y = model.feature_extraction.lookup_table.lookup_table_y.cpu().detach().numpy()
            make_fig_from_embedding(y_path, lookup_table_y)

            return True

        else:
            return False

    else:
        raise ValueError('cfg.TRAIN.SAVE_EMBED_INT should not be 0')


def make_fig_from_embedding(fig_name, embedding, value_min=-5, value_max=5, set_color='green'):
    colors = ['blue', 'white', set_color]
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    flattened_list = [num for sublist in embedding for num in sublist]
    vmin = min(flattened_list)
    vmax = max(flattened_list)
    # 将越界值截断至边界值
    embedding[embedding < vmin] = vmin
    embedding[embedding > vmax] = vmax
    # 绘制特征矩阵图像
    plt.figure(figsize=(10, 6))
    plt.imshow(embedding.T, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Feature Matrix Visualization')
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Index')
    plt.savefig(fig_name)
    plt.close()


def find_max_positions(input_tensor):
    # 沿着每行找到最大值的索引
    max_positions = torch.argmax(input_tensor, dim=1)
    return max_positions


def fill_empty(vector_org, fill_len=None):
    vector = copy.copy(vector_org)
    mask = vector == -1
    vector[mask] = fill_len if fill_len else len(vector)

    return vector


def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:, 2:] += ret[:, :2]
    return ret


def euclid_distance_of_positions(tracks, dets):
    tracks_xy = tracks.detach().cpu()
    dets_xy = dets.detach().cpu()
    expanded_tracks_xy = tracks_xy.unsqueeze(1)
    expanded_dets_xy = dets_xy.unsqueeze(0)
    result = torch.sqrt(torch.sum((expanded_tracks_xy - expanded_dets_xy)**2, dim=2)).numpy()

    _, assignment, _ = lap.lapjv(result, extend_cost=True, cost_limit=20) # TODO:参数
    assignment = fill_empty(assignment, fill_len=len(dets_xy))

    return assignment


def compare_arrays(arr1, arr2):
    # 检查两个数组的长度是否一致
    if len(arr1) != len(arr2):
        raise AssertionError('len of two arrays are not same, got {} and {}'.format(len(arr1), len(arr2)))
    # 计算匹配的元素数量
    matched_count = np.sum(arr1 == arr2)

    return matched_count


def zero_diag(matrix):
    diagonal = torch.diag(matrix)
    # 将对角线元素设置为0
    diagonal_zeroed = torch.zeros_like(diagonal)
    # 将修改后的对角线重新嵌入到原始矩阵中
    modified_matrix = matrix - torch.diag(diagonal) + torch.diag(diagonal_zeroed)

    return modified_matrix


def combine_feature_lists(patch_a, patch_b, a_weight=0.001, b_weight=1):
    result = []
    for tensor_a, tensor_b in zip(patch_a, patch_b):
        combined_tensor = a_weight * tensor_a + b_weight * tensor_b
        result.append(combined_tensor)

    return result


def lap_assignment(cost, extend_cost=True, cost_limit=None):
    if cost_limit:
        _, cols, _ = lap.lapjv(cost, extend_cost=extend_cost, cost_limit=cost_limit)
    else:
        _, cols, _ = lap.lapjv(cost, extend_cost=extend_cost)
    rows = np.arange(0, len(cols))
    del_indices = np.where(cols == -1)[0]
    rows = np.delete(rows, del_indices)
    cols = np.delete(cols, del_indices)

    return rows, cols


def score_assignment(tensor, device=None):
    # 转换为numpy数组，因为lapjv不支持PyTorch张量
    cost_matrix = tensor.detach().cpu().numpy()
    base_mat = np.ones_like(cost_matrix)
    cost_matrix = base_mat - cost_matrix
    # 使用lapjv进行线性分配
    row_ind, col_ind, _ = lap.lapjv(cost_matrix, extend_cost=True)
    # 创建mask矩阵
    col_org = torch.arange(len(col_ind))
    mask = np.zeros_like(cost_matrix)
    mask[col_org, col_ind] = 1
    # 转换回PyTorch张量
    mask_tensor = torch.tensor(mask, dtype=tensor.dtype, device=tensor.device)

    if device:
        mask_tensor_dev = mask_tensor.to(device)
        tensor_dev = tensor.to(device)
        result = mask_tensor_dev * tensor_dev

    else:
        result = tensor * mask_tensor

    return result



