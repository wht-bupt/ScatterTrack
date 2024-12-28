import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class LookupTableLayer(nn.Module):
    # 定义特征提取模块中的查找表，根据std中的方法进行2到2d的特征升维，依据索引输出对应向量，需要保证输入是int
    def __init__(self, length, dimension, cfg):
        super(LookupTableLayer, self).__init__()
        self.cfg = cfg
        self.rand_num = cfg.PARAM.EMPTY_NUM
        self.empty_index = cfg.PARAM.MAX_LEN + 1
        self.table_len = length + self.rand_num
        self.dimension = dimension
        self.initiate_mode = cfg.EXTRACT.INIT

        table_tensor = self.initiate_by_position(length, cfg).to(cfg.LOAD.DEV)
        self.fixed_lookup_table = table_tensor.to(torch.float32)

        if self.initiate_mode == 'random':
            self.lookup_table_x = torch.nn.Parameter(torch.randn(self.table_len, dimension)).to(torch.float32)
            self.lookup_table_y = torch.nn.Parameter(torch.randn(self.table_len, dimension)).to(torch.float32)
        elif self.initiate_mode == 'position':
            self.lookup_table_x = torch.nn.Parameter(table_tensor).to(torch.float32)
            self.lookup_table_y = torch.nn.Parameter(table_tensor).to(torch.float32)

    def forward(self, positions):
        if positions.ndim != 3 and positions.ndim != 4:
            raise ValueError('positions dim should be 3 for detection or 4 for tracks, got {}'.format(positions.ndim))

        elif positions.ndim == 4:               # [batch_size, max_len, rewind_len, position]
            batch_size, max_len, rewind_len, position_len = positions.shape
            positions_slices = positions.chunk(2, dim=3)
            lookup_tables = [self.lookup_table_x, self.lookup_table_y]
            total_len = batch_size * max_len * rewind_len
            result_tensors = []
            max_value_x = torch.max(self.lookup_table_x)
            max_value_y = torch.max(self.lookup_table_y)
            max_values = [max_value_x, max_value_y]
            for positions_slice, org_lookup_table_slice, max_value in zip(positions_slices, lookup_tables, max_values):
                std_lookup_table_slice = org_lookup_table_slice / max_value
                lookup_table_slice = self.cfg.EXTRACT.ADD_RATE * std_lookup_table_slice + self.fixed_lookup_table
                flattened_positions = positions_slice.reshape(total_len)
                flattened_positions = torch.where(flattened_positions < 0, torch.tensor(1., dtype=flattened_positions.dtype).to(self.cfg.LOAD.DEV),
                                                  flattened_positions)
                concat_positions = torch.arange(flattened_positions.shape[0]).to(self.cfg.LOAD.DEV)
                coalesced_indices = torch.stack([concat_positions, flattened_positions])
                values = torch.ones_like(concat_positions)
                sparse_positions = torch.sparse_coo_tensor(coalesced_indices, values, size=(total_len, self.table_len)).to(torch.float32)
                encode_result = torch.sparse.mm(sparse_positions, lookup_table_slice)
                position_embeddings = encode_result.reshape((batch_size, max_len, rewind_len, self.dimension))
                result_tensors.append(position_embeddings)

            concat_position_embeddings = torch.cat(tuple(result_tensors), dim=-1)
            shortened_concat_position_embeddings = concat_position_embeddings.view(batch_size, max_len, rewind_len, self.dimension, 2).sum(dim=-1)

        elif positions.ndim == 3:
            batch_size, max_len, position_len = positions.shape
            positions_slices = positions.chunk(2, dim=2)
            lookup_tables = [self.lookup_table_x, self.lookup_table_y]
            max_value_x = torch.max(self.lookup_table_x)
            max_value_y = torch.max(self.lookup_table_y)
            max_values = [max_value_x, max_value_y]
            total_len = batch_size * max_len
            result_tensors = []
            for positions_slice, org_lookup_table_slice, max_value in zip(positions_slices, lookup_tables, max_values):
                std_lookup_table_slice = org_lookup_table_slice / max_value
                lookup_table_slice = self.cfg.EXTRACT.ADD_RATE * std_lookup_table_slice + self.fixed_lookup_table
                flattened_positions = positions_slice.reshape(total_len)
                flattened_positions = torch.where(flattened_positions < 0, torch.tensor(1., dtype=flattened_positions.dtype).to(self.cfg.LOAD.DEV),
                                                  flattened_positions)
                concat_positions = torch.arange(flattened_positions.shape[0]).to(self.cfg.LOAD.DEV)
                coalesced_indices = torch.stack([concat_positions, flattened_positions])
                values = torch.ones_like(concat_positions)
                sparse_positions = torch.sparse_coo_tensor(coalesced_indices, values, size=(total_len, self.table_len)).to(
                    torch.float32)
                encode_result = torch.sparse.mm(sparse_positions, lookup_table_slice)
                position_embeddings = encode_result.reshape((batch_size, max_len, self.dimension))
                result_tensors.append(position_embeddings)

            concat_position_embeddings = torch.cat(tuple(result_tensors), dim=-1)
            shortened_concat_position_embeddings = concat_position_embeddings.view(batch_size, max_len,
                                                                                   self.dimension, 2).sum(dim=-1)

        return shortened_concat_position_embeddings

    def old_initiate_by_position(self, length, cfg):
        tensor_1 = torch.arange(length, dtype=torch.float) / torch.tensor(1000) - torch.tensor(0.5)
        tensor_2 = torch.full((self.rand_num,), -5, dtype=torch.float)
        result_tensor = torch.cat((tensor_1, tensor_2), dim=0)
        final_tensor = result_tensor.unsqueeze(1).expand(-1, self.dimension)

        return final_tensor

    def initiate_by_position(self, length, cfg):
        start = cfg.EXTRACT.START
        pos_step = cfg.EXTRACT.POS_STEP
        embed_step = cfg.EXTRACT.EMB_STEP
        empty_value = cfg.EXTRACT.SUP_VAL
        tensor_list = []
        for i in range(self.dimension):
            tensor_start = start + i * embed_step
            tensor_values = torch.arange(tensor_start, tensor_start + length * pos_step, pos_step)
            tensor_list.append(tensor_values.unsqueeze(1))
        tensor_1 = torch.cat(tensor_list, dim=1)
        tensor_2 = torch.full((self.rand_num, self.dimension), empty_value)
        result = torch.cat((tensor_1, tensor_2), dim=0)

        return result


class ExtractionModel(nn.Module):
    # 完整的特征提取网络
    # 输入：回溯的轨迹集合，形状为[track_num, rewind_len, 4]；或检测目标的位置集合，形状为[det_num, 4]
    # 输出：编码后的轨迹集合，形状为[track_num, rewind_len, d]；或编码后的检测目标集合，形状为[det_num, d]
    def __init__(self, length, cfg):
        dimension = cfg.EXTRACT.DIM
        super(ExtractionModel, self).__init__()
        self.lookup_table = LookupTableLayer(length, dimension, cfg)
        self.linear = nn.Linear(2*dimension, dimension)
        self.linear_for_val = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(cfg.EXTRACT.DROP)

    def forward(self, positions_all):

        position_embeddings = self.lookup_table(positions_all)
        # position_embeddings = self.linear(position_embeddings)
        # position_embeddings = self.linear_for_val(position_embeddings)
        # position_embeddings = self.dropout(position_embeddings)

        return position_embeddings


class AddAndNorm(nn.Module):
    # GAT模块中的残差&归一化模块
    # 归一化方法待定
    def __init__(self, hidden_dim, dropout_rate):
        super(AddAndNorm, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_1d = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_2d = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, residual):
        x = self.dropout(x)
        x = x + residual
        x = self.batch_norm_1d(x)
        return x


class GATLayer(torch.nn.Module):
    # GAT空间注意力模块，处理轨迹间的空间关系
    # 输入：每条轨迹的当前帧特征集合[track_num, d]
    # 输出：聚合后的特征集合[track_num, d]
    def __init__(self, cfg):
        super(GATLayer, self).__init__()
        heads = cfg.GAT.HEADS
        in_features, hidden_dim, out_features = cfg.GAT.DIM
        self.conv1 = GATConv(in_features, int(hidden_dim / heads), heads=heads)
        self.conv2 = GATConv(hidden_dim, out_features, heads=1)
        self.add_norm = AddAndNorm(hidden_dim, cfg.GAT.DROP)

    def forward(self, x, edge_index):
        x_residual = x.clone()
        x = self.conv1(x, edge_index)
        x = self.add_norm(x, x_residual)
        x = F.relu(x)
        x_residual = x.clone()
        x = self.conv2(x, edge_index)
        x = self.add_norm(x, x_residual)
        return x


class TransformerEncoder(nn.Module):
    # Transformer Encoder时间注意力模块，处理每条轨迹内部的时间信息
    # 输入：当前帧轨迹embedding集合[track_num, rewind_len, d]
    # 输出：时间聚合后embedding集合[track_num, rewind_len, d]
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        input_dim = cfg.TF.INPUT_DIM
        dim_ffn = cfg.TF.FFN_DIM
        num_layers = cfg.TF.ENCODER_NUM
        num_heads = cfg.TF.HEADS
        dropout_rate = cfg.TF.DROP
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, embeddings):
        embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        encoder_output = self.transformer_encoder(embeddings)
        return encoder_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)


class TransformerDecoder(nn.Module):
    # Transformer Decoder模块，将轨迹信息与检测信息进行交互
    # 输入：
    # 聚合后的轨迹最新帧集合[track_num, d]
    # 聚合后的检测目标集合[det_num, d]
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        input_dim = cfg.TF.INPUT_DIM
        dim_ffn = cfg.TF.FFN_DIM
        num_layers = cfg.TF.DECODER_NUM
        num_heads = cfg.TF.HEADS
        dropout_rate = cfg.TF.DROP
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

    def forward(self, embeddings, encoder_output):
        embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        encoder_output = encoder_output.permute(1, 0, 2)
        decoder_output = self.transformer_decoder(embeddings, encoder_output)
        return decoder_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)


# 余弦相似度输出层
class ClassifyLayer(torch.nn.Module):
    def __init__(self, cfg):
        super(ClassifyLayer, self).__init__()
        self.device = cfg.LOAD.DEV

    def forward(self, embeddings, W, len_prd, head=1):
        # 先将邻接矩阵W转化成mask矩阵（对上分支只保留右上分块）
        mask = W[:len_prd, len_prd:] if head == 1 else W
        mask = mask.float()
        mask_org = torch.ones_like(mask).to(self.device)
        mask = -99 * (mask_org - mask)

        # 计算余弦距离，使用relu滤除余弦距离小于0的以及被mask的元素，并进行归一化
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        for i in range(len(norms)):
            if norms[i] <= 1e-9:
                norms[i] = 1e-9
        normalized_matrix = embeddings / norms
        similarity_matrix = torch.matmul(normalized_matrix, normalized_matrix.t())
        similarity_matrix = similarity_matrix[:len_prd, len_prd:] if head == 1 else similarity_matrix
        masked_similarity_matrix = similarity_matrix + mask
        x_len, y_len = masked_similarity_matrix.shape
        inner_matrix = torch.ones((x_len, y_len)).to(self.device)
        score_matrix = inner_matrix - masked_similarity_matrix
        output_matrix = torch.relu(masked_similarity_matrix)
        output_matrix = torch.clamp(output_matrix, max=1)
        half_normalized_matrix = torch.div(torch.ones((x_len, y_len)).to(self.device), score_matrix)
        row_sum = torch.sum(half_normalized_matrix, dim=1, keepdim=True)

        for i in range(len(row_sum)):
            if row_sum[i] <= 1e-9:
                row_sum[i] = 1e-9
        normalized_matrix = half_normalized_matrix / row_sum

        # 对空轨迹的处理，使用masked_similarity_matrix筛选空轨迹行，在normalized_matrix右侧添加一列代表空轨迹，
        # 空轨迹行左侧均为0，右侧一列数值为1. 筛选标准为：1.所有目标余弦相似度小于0且不符合距离要求；2.网络给出的评分
        # 小于阈值（暂未添加）。
        supple_tensor = torch.tensor([0 if torch.any(row > 0) else 1 for row in masked_similarity_matrix]).to(self.device)
        processed_matrix = normalized_matrix.clone()
        for i, val in enumerate(supple_tensor):
            if val.item() == 1:
                processed_matrix[i] = 0
        supple_tensor_transposed = supple_tensor.view(-1, 1)
        extended_matrix = torch.cat((processed_matrix, supple_tensor_transposed), dim=1)

        return output_matrix, extended_matrix





