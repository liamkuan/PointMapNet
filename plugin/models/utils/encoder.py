import torch
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from torch import nn
from plugin.models.utils import _get_clones, _get_activation_fn


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MultiScaleDeformableAttention(d_model, n_heads, n_levels, n_points, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.d_model = d_model
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
            self,
            query,
            key,
            value,
            reference_points_cam,
            spatial_shapes,
            level_start_index,
            bev_mask,
    ):
        slots = torch.zeros_like(query)
        inp_residual = query
        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)
        bs, num_cams, l, embed_dims = key.shape
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        queries_rebatch = query.new_zeros(
            [bs, num_cams, max_len, self.d_model])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        key = key.reshape(bs * num_cams, l, self.d_model)
        value = value.reshape(bs * num_cams, l, self.d_model)

        queries = self.self_attn(query=queries_rebatch.view(bs * num_cams, max_len, self.d_model),
                                 key=key, value=value,
                                 reference_points=reference_points_rebatch.view(bs * num_cams, max_len,
                                                                                D, 2),
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index).view(bs, num_cams, max_len,
                                                                           self.d_model)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        query = self.dropout1(slots) + inp_residual
        query = self.norm1(query)
        query = self.forward_ffn(query)
        return query


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, bev_h, bev_w):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(
            self,
            bev_queries: torch.Tensor,
            src,
            spatial_shapes,
            level_start_index,
            inputs,
    ):
        reference_points_cam = inputs['ref_points_cam']
        bev_mask = inputs['bev_mask']
        for layer in self.layers:
            bev_queries = layer(
                bev_queries,
                src,
                src,
                reference_points_cam,
                spatial_shapes,
                level_start_index,
                bev_mask,
            )
        return bev_queries
