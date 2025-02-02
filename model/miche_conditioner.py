import torch
from torch import nn
from beartype import beartype
from miche.encode import load_model

# helper functions

def exists(val):
    return val is not None

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None


@beartype
class PointConditioner(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_latent = None,
        model_name = 'miche',
        use_meta = False,
        meta_dim = 128,
        cond_dim = 768, # from miche
        cond_drop_prob = 0,
        freeze = True,
    ):
        super().__init__()

        assert model_name == 'miche-256-feature', "open-source version only supports 'miche-256-feature' model"
            
        ckpt_path = 'miche/shapevae-256.ckpt'
        config_path = 'miche/shapevae-256.yaml'
        self.feature_dim = 1024
        self.cond_length = 257

        self.point_encoder = load_model(ckpt_path=ckpt_path, config_path=config_path)
        self.cond_head_proj = nn.Linear(cond_dim, self.feature_dim)
        self.cond_proj = nn.Linear(cond_dim, self.feature_dim)

        if freeze:
            for parameter in self.point_encoder.parameters():
                parameter.requires_grad = False
        
        self.freeze = freeze
        self.use_meta = use_meta
        self.model_name = model_name
        self.dim_latent = default(dim_latent, self.feature_dim)

        if self.use_meta: 
            self.to_meta_embed = nn.Sequential(
                nn.Linear(3, meta_dim),
                nn.ReLU(),
                nn.Linear(meta_dim, meta_dim),
            )
            self.dim_latent += meta_dim


        self.cond_drop_prob = cond_drop_prob
        
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.feature_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def embed_pc(self, pc_normal):
        point_feature = self.point_encoder.encode_latents(pc_normal)
        pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
        pc_embed = self.cond_proj(point_feature[:, 1:])
        pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)

        return pc_embed

    def forward(
        self,
        pc = None,
        meta = None,
        pc_embeds = None,
        cond_drop_prob = None,
    ):
        if pc_embeds is None:
            pc_embeds = self.embed_pc(pc.to(next(self.buffers()).dtype))

        if self.use_meta:
            meta_embeds = self.to_meta_embed(meta).unsqueeze(1)
            pc_embeds = torch.cat([pc_embeds, meta_embeds], dim=-1)
            
        assert not torch.any(torch.isnan(pc_embeds)), 'NAN values in pc embedings'
        
        return pc_embeds, None