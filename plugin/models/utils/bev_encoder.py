from torch import nn
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
            self,
            query,
            memory,
    ):
        out = self.encoder(query, memory)
        return out
