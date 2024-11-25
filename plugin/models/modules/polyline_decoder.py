import copy
import math
import warnings
import torch
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmdet3d.registry import MODELS


@MODELS.register_module()
class PolylineDecoder(TransformerLayerSequence):
    def __init__(
            self,
            *args,
            return_intermediate=True,
            **kwargs,
    ):
        super(PolylineDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
            self,
            instance_query,
            pts_query,
            key,
            value,
            **kwargs,
    ):
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                instance_query,
                pts_query,
                key,
                value,
                **kwargs,
            )
            instance_query, pts_query = torch.split(output, output.size(-1) // 2, dim=-1)
            bs, num_query, dim = instance_query.shape
            instance_query = instance_query.view(bs, 50, 20, dim).mean(2)
            intermediate.append(output)

        return torch.stack(intermediate)


@MODELS.register_module()
class PolylineLayer(BaseTransformerLayer):
    def __init__(
            self,
            attn_cfgs,
            feedforward_channels,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type="ReLU"),
            norm_cfg=dict(type="LN"),
            ffn_num_fcs=2,
            embed_dims=256,
            **kwargs,
    ):
        super(PolylineLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(
                type="FFN",
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=ffn_num_fcs,
                ffn_drop=ffn_dropout,
                act_cfg=act_cfg,
            ),
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            batch_first=True,
            **kwargs,
        )
        # assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(
            self,
            instance_query,
            pts_query,
            key=None,
            value=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        # temp_instance = instance_query.unsqueeze(2).expand(-1, -1, 20, -1).flatten(1, 2)
        # query = torch.cat([temp_instance, pts_query], dim=-1)
        identity = None
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                # instance_query, pts_query = torch.split(query, query.size(-1) // 2, dim=-1)
                bs, num_query, dim = instance_query.shape
                # instance_query = instance_query.view(bs, 50, 20, dim).mean(2)
                if attn_index == 0:
                    query = instance_query
                else:
                    bs, num_query, dim = pts_query.shape
                    query = pts_query.view(bs, 50, 20, dim).flatten(0, 1)
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                if attn_index == 0:
                    instance_query = query
                else:
                    pts_query = query.view(bs, 50, 20, dim).flatten(1, 2)
                attn_index += 1
                temp_instance = instance_query.unsqueeze(2).repeat(1, 1, 20, 1).flatten(1, 2)
                query = torch.cat([temp_instance, pts_query], dim=-1)
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
                query = query.to(key.dtype)

            elif layer == "cross_attn":
                # bs, num_query, embed_dim = query.shape
                # temp_query = query.view((bs, 50, 20, embed_dim)).mean(2)
                # q_scaled = temp_query / math.sqrt(embed_dim)
                
                # attention_weight = torch.bmm(q_scaled, key.transpose(-2, -1))
                # attention_weight = torch.softmax(attention_weight, dim=-1)
                # mask = attention_weight > 0.012
                # mask = mask.unsqueeze(2).unsqueeze(1).repeat(1, 8, 1, 20, 1).flatten(0, 1).flatten(1, 2)
                # temp = torch.count_nonzero(mask, dim=2).float().mean()
                # if temp < 20:
                #     mask = None
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query