import torch
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from torch import nn
from torch.nn.init import normal_

from .encoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from .position_embedding import get_position_embedding
from plugin.models.utils import _get_clones, _get_activation_fn
from mmdet3d.registry import MODELS


@MODELS.register_module()
class DeformableTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            bev_h=200,
            bev_w=150,
            num_bev_query=30000,
            num_queries=80,
            bev_encoder_cfg=dict(),
            point_cloud_range=[-30, -60, -5.0, 30, 60, 3.0]
    ):
        super().__init__()

        self.d_model = d_model
        self.point_cloud_range = point_cloud_range
        self.nhead = nhead
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.point_embedding = nn.Embedding(num_bev_query, self.d_model)

        # self.bev_embedding = nn.Embedding(num_bev_query, self.d_model)
        # self.point_pos_embedding = nn.Linear(3, d_model, bias=False)
        # self.bev_encoder = MODELS.build(bev_encoder_cfg)

        self.query_embed = nn.Embedding(num_queries, self.d_model * 2)
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, bev_h, bev_w)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            1,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    def get_bev_ref_points(self, device):
        """
        从激光雷达坐标系转换到自车坐标系，xy坐标需要互换。
        :return: 自车坐标系下车周围均匀采样的点
        """
        pt_range = self.point_cloud_range
        ref_x, ref_y = torch.meshgrid(
            torch.linspace(pt_range[4] - 0.5, pt_range[1] + 0.5, pt_range[4] - pt_range[1], dtype=torch.float32,
                           device=device),
            torch.linspace(pt_range[3] - 0.5, pt_range[0] + 0.5, pt_range[3] - pt_range[0], dtype=torch.float32,
                           device=device),
            indexing='ij'
        )
        ref_points = torch.stack([ref_x, ref_y], dim=-1)
        ref_points = ref_points.flatten(0, 1)

        ref_points_3d = torch.cat(
            [ref_points, torch.ones([ref_points.size(0), 1], dtype=ref_points.dtype, device=ref_points.device) * -1.5],
            dim=-1)

        return ref_points_3d

    def forward(self, srcs, inputs):
        # prepare input for encoder
        bs, num_cams, _, _, _ = inputs['imgs'].shape
        point_queries = self.point_embedding.weight
        point_queries = point_queries.unsqueeze(0).repeat(bs, 1, 1)
        # point_pos_embeds = self.point_pos_embedding(inputs['bev_points'])
        # point_queries = point_queries + point_pos_embeds

        src_flatten = []
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            _, c, h, w = src.shape
            src_pos = get_position_embedding(h, w, c, src.device)
            src = src.reshape(bs, num_cams, c, h, w)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(3).permute(0, 1, 3, 2)
            src = src + src_pos.flatten(0, 1) + self.level_embed[lvl]
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)),
             spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # encoder
        point_feat = self.encoder(
            point_queries,
            src_flatten,
            spatial_shapes,
            level_start_index,
            inputs,
        )

        # bev_queries = self.bev_embedding.weight
        # bev_ref_points = self.get_bev_ref_points(bev_queries.device)
        # bev_pos_embeds = self.point_pos_embedding(bev_ref_points)
        # bev_queries = bev_queries + bev_pos_embeds
        # bev_queries = bev_queries.unsqueeze(0).repeat(bs, 1, 1)
        # bev_feat = self.bev_encoder(bev_queries, point_feat)

        query_embeds = self.query_embed.weight
        query_embeds, tgt = torch.split(query_embeds, self.d_model, dim=1)

        query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        ref_points = self.reference_points(query_embeds).sigmoid()
        ref_points = ref_points.unsqueeze(2)
        spatial_shape = torch.tensor([[self.bev_h, self.bev_w]], device=point_feat.device)
        start_index = torch.tensor([0], device=point_feat.device)
        hs, inter_references = self.decoder(tgt, ref_points, point_feat, spatial_shape, start_index, query_embeds)

        return hs, ref_points, inter_references


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=1,
            n_heads=8,
            n_points=4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_heads, n_levels, n_points, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        query2 = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query2)
        query = self.norm3(query)
        return query

    def forward(
            self,
            query,
            query_pos,
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
    ):
        # self attention
        q = k = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1)
        )[0].transpose(0, 1)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            key=src,
            value=src,
            reference_points=reference_points,
            spatial_shapes=src_spatial_shapes,
            level_start_index=level_start_index,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(
            self,
            query,
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            query_pos
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                src_level_start_index,
            )
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points
