import torch
import torch.nn as nn
import torch.nn.functional as F
from scatter_modules import *
from tracking_toolbox import *
from torch_geometric.data import Data, DataLoader


class ScatterNet(nn.Module):
    def __init__(self, cfg, logger=None, is_eval=False):
        super(ScatterNet, self).__init__()
        self.feature_extraction = ExtractionModel(cfg.PARAM.MAX_LEN, cfg)
        self.gat_layers = nn.ModuleList([GATLayer(cfg) for _ in range(cfg.GAT.NUM)])
        self.det_gat_layers = nn.ModuleList([GATLayer(cfg) for _ in range(cfg.GAT.NUM)])
        self.asso_layers = nn.ModuleList([GATLayer(cfg) for _ in range(cfg.GAT.ASSO_NUM)])
        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)
        self.classify_layer = ClassifyLayer(cfg)
        self.cfg = cfg
        self.is_eval = is_eval
        self.device = cfg.LOAD.DEV
        self.batch_size = cfg.PARAM.BATCH_SIZE
        self.logger = logger

    def forward(self, tracks_all, detections, end_marks):

        # 1. 位置特征编码生成
        tracklet_marks, detection_marks = end_marks
        track_embeddings = self.feature_extraction(tracks_all)
        det_embeddings = self.feature_extraction(detections)

        # 2. 轨迹内特征融合
        batch_size, max_len, rewind_len, dimension = track_embeddings.shape
        track_spatial_embeddings = track_embeddings.reshape((batch_size * max_len, rewind_len, dimension))
        encoder_output = self.encoder(track_spatial_embeddings)
        reshaped_encoder_output = encoder_output.reshape((batch_size, max_len, rewind_len, dimension))
        track_overall_embeddings = reshaped_encoder_output[:, :, 0, :].squeeze()
        org_embeddings = track_embeddings[:, :, 0, :].squeeze()
        if self.is_eval or batch_size == 1:
            track_overall_embeddings = track_overall_embeddings.unsqueeze(0)
            predicted_features = org_embeddings.unsqueeze(0) + self.cfg.TF.TF_RATE * track_overall_embeddings

        else:
            predicted_features = self.cfg.TF.TF_RATE * track_overall_embeddings + org_embeddings

        # 3. 输出头1，将轨迹token和检测token拼合，计算邻接矩阵，送入GAT网络，得到轨迹-检测关联矩阵
        asso_dataset = []
        asso_memory = tracklet_marks + detection_marks
        asso_edges = []
        classify_edges = []
        feature_patch = []

        for i, (predicted_feature, det_embedding, track_mark, det_mark) in \
                enumerate(zip(predicted_features, det_embeddings, tracklet_marks, detection_marks)):
            track_fragment = predicted_feature[:track_mark, :]
            det_fragment = det_embedding[:det_mark, :]
            feature_list = torch.cat((track_fragment, det_fragment), dim=0)
            feature_patch.append(feature_list)
            hist_positions = [track[1].cpu() for track in tracks_all[i]][:track_mark]    # 提取上一帧的所有轨迹目标位置
            predictions = [track[0].cpu() for track in tracks_all[i]][:track_mark]    # 提取所有预测轨迹目标位置
            detections_sliced = detections[i][:det_mark].cpu()

            track_adjacent_matrix = self_adjacent_matrix(hist_positions, self.cfg)
            det_adjacent_matrix = self_adjacent_matrix(detections_sliced, self.cfg)
            asso_adjacent_matrix = cross_adjacent_matrix(predictions, detections_sliced, self.cfg)

            len_prd = len(predictions)
            adjacent_matrix = np.zeros((len(feature_list), len(feature_list)))
            adjacent_matrix[:len_prd, :len_prd] = track_adjacent_matrix
            adjacent_matrix[len_prd:, len_prd:] = det_adjacent_matrix
            adjacent_matrix[:len_prd, len_prd:] = asso_adjacent_matrix
            adjacent_matrix[len_prd:, :len_prd] = asso_adjacent_matrix.T
            adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float32).to(self.device)
            asso_index = torch.nonzero(adjacent_matrix, as_tuple=False).t()  # 邻接矩阵转换成边索引矩阵
            asso_edges.append(asso_index)
            classify_edges.append(adjacent_matrix)

            data = Data(x=feature_list, edge_index=asso_index).to(self.device)
            asso_dataset.append(data)

        for gat_layer in self.gat_layers:
            dataloader = DataLoader(asso_dataset, self.batch_size, shuffle=False)
            asso_dataset = []
            for data_batch in dataloader:  # 局部处理，只有一个batch
                gat_result = gat_layer(data_batch.x, data_batch.edge_index)
                asso_spatial_inter_gcn = split_and_concatenate(gat_result, asso_memory)
                asso_spatial_inter = combine_feature_lists(asso_spatial_inter_gcn, feature_patch,
                                                           a_weight=self.cfg.GAT.GAT_W, b_weight=self.cfg.GAT.ORG_W)
                for tracks, gat_edge in zip(asso_spatial_inter, asso_edges):
                    data = Data(x=tracks, edge_index=gat_edge).to(self.device)
                    asso_dataset.append(data)


        associate_results = []
        scores = []
        for i, asso_fragment in enumerate(asso_spatial_inter):
            score, asso_result = self.classify_layer(asso_fragment, classify_edges[i], tracklet_marks[i])
            associate_results.append(asso_result)
            scores.append(score)

        # 4. 输出头2，只将检测token送入GAT网络，网络之间不共享权重，得到从属同一目标概率矩阵
        asso_dataset = []
        det_feature_patch = []
        det_edges = []
        for i, (det_embedding, det_mark) in enumerate(zip(det_embeddings, detection_marks)):
            det_fragment = det_embedding[:det_mark, :]
            det_feature_patch.append(det_fragment)
            detections_sliced = detections[i][:det_mark].cpu()
            det_adjacent_matrix = self_adjacent_matrix(detections_sliced, self.cfg)
            det_adjacent_matrix = torch.tensor(det_adjacent_matrix, dtype=torch.float32).to(self.device)
            asso_index = torch.nonzero(det_adjacent_matrix, as_tuple=False).t()
            data = Data(x=det_fragment, edge_index=asso_index).to(self.device)
            det_edges.append(det_adjacent_matrix)
            asso_dataset.append(data)

        for det_gat_layer in self.det_gat_layers:
            dataloader = DataLoader(asso_dataset, self.batch_size, shuffle=False)
            asso_dataset = []
            for data_batch in dataloader:  # 局部处理，只有一个batch
                gat_result = det_gat_layer(data_batch.x, data_batch.edge_index)
                det_asso_spatial_inter_gcn = split_and_concatenate(gat_result, detection_marks)
                det_asso_spatial_inter = combine_feature_lists(det_asso_spatial_inter_gcn, det_feature_patch,
                                                           a_weight=self.cfg.GAT.DET_W, b_weight=self.cfg.GAT.ORG_W)
                for tracks, gat_edge in zip(det_asso_spatial_inter, asso_edges):
                    data = Data(x=tracks, edge_index=gat_edge).to(self.device)
                    asso_dataset.append(data)


        det_scores = []
        for i, asso_fragment in enumerate(det_asso_spatial_inter):
            raw_det_score, _ = self.classify_layer(asso_fragment, det_edges[i], 0, head=2)
            det_score = zero_diag(raw_det_score)
            det_scores.append(det_score)

        return scores, det_scores, associate_results

    def generate_graph_edge(self, positions):
        track_adjacent_matrix_np = self_adjacent_matrix(positions, self.cfg)
        track_adjacent_matrix = torch.from_numpy(track_adjacent_matrix_np)
        edge_index = torch.nonzero(track_adjacent_matrix, as_tuple=False).t()  # 邻接矩阵转换成边索引矩阵
        return edge_index


