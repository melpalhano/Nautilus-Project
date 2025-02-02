from math import ceil

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from pytorch_custom_utils import save_load
from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, Any
from torch_cluster import knn
from einops import rearrange, repeat, pack
from data.data_utils import undiscretize
from model.x_transformers import Decoder
from torch.nn.utils import weight_norm
from torch_scatter import scatter_max
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
)
from .miche_conditioner import PointConditioner
from einops import unpack
from tqdm import tqdm


# helper functions

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = (pos[row] - pos_dst[col]).to(basis)   # each ball center to pos

        if basis is not None:
            embeddings = torch.einsum('bd,de->be', out, basis)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
            out = torch.cat([out, embeddings], dim=1)

        if self.local_nn is not None:
            out = self.local_nn(out)

        out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


@save_load()
class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, int]] = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            attn_qk_norm = True
        ),
        dropout = 0.,
        pad_id = -1,
        coor_continuous_range = (-1., 1.),
        num_discrete_coors = 64,
        u_size = 1024,
        v_size = 2048,
        encoder_name = 'miche-256-feature',
        encoder_freeze = True,
    ):
        super().__init__()

        vocab_size = 2 * u_size + v_size
        
        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = vocab_size
        self.token_embed = nn.Embedding(vocab_size + 1, dim)
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range
        self.u_size = u_size
        self.v_size = v_size
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len
        self.conditioner = None
        
        assert self.u_size * self.v_size == self.num_discrete_coors ** 3

        print(f'Point cloud encoder: {encoder_name} | freeze: {encoder_freeze}')
        self.conditioner = PointConditioner(model_name=encoder_name, freeze=encoder_freeze)
        cross_attn_dim_context = self.conditioner.dim_latent
        
        # main autoregressive attention network
        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = True,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = 4,  # needed for preventing nan when dropping out text condition
            **attn_kwargs
        )

        self.to_logits = nn.Linear(dim, vocab_size + 1)
        self.pad_id = pad_id

        self.pconv_embedding_dim = 48
        e = torch.pow(2, torch.arange(self.pconv_embedding_dim // 6)).float() * torch.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.pconv_embedding_dim // 6),
                       torch.zeros(self.pconv_embedding_dim // 6)]),
            torch.cat([torch.zeros(self.pconv_embedding_dim // 6), e,
                       torch.zeros(self.pconv_embedding_dim // 6)]),
            torch.cat([torch.zeros(self.pconv_embedding_dim // 6),
                       torch.zeros(self.pconv_embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16
        self.pconv = PointConv(
            local_nn=nn.Sequential(weight_norm(nn.Linear(3 + self.pconv_embedding_dim, 256)), nn.ReLU(True),
                                   weight_norm(nn.Linear(256, 256))),
            global_nn=nn.Sequential(weight_norm(nn.Linear(256, 256)), nn.ReLU(True), weight_norm(nn.Linear(256, dim))),
        )
        self.proj_local_feats = nn.Linear(dim, dim)

        self.cond_norm = nn.LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        pc: Optional[Tensor] = None,
        cond_embeds: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None,
        tqdm_position = 0,
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if cond_embeds is None:
            cond_embeds, _ = self.conditioner(pc = pc)
            cond_embeds = self.cond_norm(cond_embeds).to(torch.bfloat16)

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = None
        pc_cache = None

        for _ in tqdm(range(curr_length, max_seq_len), position=tqdm_position,
                      desc=f'Process: {tqdm_position}', dynamic_ncols=True, leave=False):

            output = self.forward_on_codes(
                codes,
                return_cache = cache_kv,
                append_eos = False,
                cond_embeds = cond_embeds,
                pc=pc,
                cache = cache,
                pc_cache=pc_cache,
            )

            if cache_kv:
                logits, cache, pc_cache = output

            else:
                logits = output

            logits = logits[:, -1]
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate
            is_eos_codes = (codes == self.eos_token_id)
            if is_eos_codes.any(dim = -1).all():
                break

        # mask out to padding anything after the first eos
        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # early return of raw residual quantizer codes
        if return_codes:
            return codes

        face_coords, face_mask = self.decode_codes(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward_on_codes(
        self,
        codes = None,
        return_cache = False,
        append_eos = True,
        cache = None,
        cond_drop_prob = None,
        pc = None,
        cond_embeds = None,
        pc_cache = None,
    ):
        assert exists(pc) | exists(cond_embeds), 'point cloud should be given'

        # preprocess faces and vertices
        if not exists(cond_embeds):
            cond_embeds, _ = self.conditioner(
                pc = pc,
                pc_embeds = cond_embeds,
                cond_drop_prob = cond_drop_prob,
            )
            cond_embeds = self.cond_norm(cond_embeds)

        attn_context_kwargs = dict(
            context = cond_embeds,
            context_mask = None,
        )

        if codes.shape[1] > 2:
            if pc_cache is not None:
                all_local_feats = self.get_pc_features_cached(codes=codes, pc=pc, cache=pc_cache).unsqueeze(1)
                all_local_feats = torch.cat((pc_cache, all_local_feats), dim=1)
            else:
                all_local_feats = self.get_pc_features_uncached(codes=codes, pc=pc)
        else:
            all_local_feats = torch.zeros(codes.shape[0], codes.shape[1], self.sos_token.shape[0]).to(cond_embeds)

        # take care of codes that may be flattened
        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable
        batch, seq_len, device = *codes.shape, codes.device
        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token
        if append_eos:
            assert exists(codes)
            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)
            codes = F.pad(codes, (0, 1), value = 0)  # value=-1
            batch_arange = torch.arange(batch, device = device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')
            codes[batch_arange, code_lens] = self.eos_token_id

        # token embed (each residual VQ id)
        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions
        seq_arange = torch.arange(codes.shape[-2], device = device)
        codes = codes + self.abs_pos_emb(seq_arange)

        if all_local_feats.shape[1] > 0:
            codes = codes + self.proj_local_feats(all_local_feats)

        # auto prepend sos token
        sos = repeat(self.sos_token, 'd -> b d', b = batch)
        codes, _ = pack([sos, codes], 'b * d')

        # attention
        attended, intermediates_with_cache = self.decoder(
            codes,
            cache = cache,
            return_hiddens = True,
            **attn_context_kwargs
        )

        # logits
        logits = self.to_logits(attended)

        if not return_cache:
            return logits

        return logits, intermediates_with_cache, all_local_feats

    def get_pc_conv_features(self, codes, pc, center_mask, k=100):

        B, N, D = pc[..., :3].shape  # only xyz
        pos = pc[..., :3].view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        center_num = center_mask.sum(-1)
        center_mask = center_mask.clone()   # important!
        center_mask[center_num % 2 != 0, -1] = False    # remove the hanging center_blk token
        center_num = center_num // 2
        queries_batch = torch.repeat_interleave(torch.arange(B).to(center_num), center_num)

        # decode centers
        center_codes = codes[center_mask].view(-1, 2)
        u_id, v_id = unpack(center_codes, [[1], [1]], 'bn *')
        u_id -= self.u_size + self.v_size
        v_id -= self.u_size
        discrete_coords = []
        sum_coords = u_id * self.v_size + v_id
        for i in [2, 1, 0]:
            axis = (sum_coords // self.num_discrete_coors ** i)
            sum_coords %= self.num_discrete_coors ** i
            discrete_coords.append(axis)
        discrete_coords = torch.cat(discrete_coords, dim=-1)
        queries = undiscretize(discrete_coords, num_discrete=self.num_discrete_coors).to(pos)

        row, col = knn(pos, queries, k, batch, queries_batch)  # row: queries index, col: pos index
        edge_index = torch.stack([col, row], dim=0)
        pc_feats = self.pconv(pos, queries, edge_index, self.basis)

        # pc_feats: [N1+N2+...Nk, D], num_complete_queries = (N1, N2, ..., Nk)
        # seperate pc_feats into different batches by num_complete_queries
        pc_feats = [pc_feats[queries_batch == b] for b in range(B)]

        return pc_feats

    def get_pc_features_uncached(self, codes, pc):

        B, N, D = pc[..., :3].shape  # only xyz

        code_mask = codes != self.pad_id
        center_u_mask = ((self.u_size + self.v_size <= codes) &
                             (codes < self.u_size + self.v_size + self.u_size))
        center_v_mask = torch.cat([torch.ones(B, 1).bool().to(center_u_mask),
                                        center_u_mask[:, :-1]], dim=-1)
        center_mask = center_u_mask | center_v_mask
        center_feats = self.get_pc_conv_features(codes=codes, pc=pc, center_mask=center_mask, k=100)

        center_idxs = (torch.cumsum(center_mask, dim=-1) - 1) // 2
        center_idxs[center_mask | ~code_mask] = -1

        all_center_feats = []
        for b in range(B):
            # project center_feats to each codes
            center_feat = center_feats[b]
            center_idx = center_idxs[b].clone()  # Clone to avoid in-place modification
            center_idx = torch.where(center_idx == -1, torch.tensor(len(center_feat), device=center_idx.device), center_idx)
            # append a all zeros to the end of center_feat of dim 0
            center_feat = torch.cat([center_feat, torch.zeros(1, center_feat.shape[1]).to(center_feat)], dim=0)
            all_center_feats.append(center_feat[center_idx])

        all_local_feats = torch.stack(all_center_feats, dim=0)

        return all_local_feats

    def get_pc_features_cached(self, codes, pc, cache):

        # default
        next_features = cache[:, -1].clone()

        # case 1: Xv -> Cu/eos/pad
        is_spec = ((self.u_size + self.v_size <= codes[:, -1]) | (codes[:, -1] == self.pad_id))
        if torch.any(is_spec):
            next_features[is_spec] *= 0

        # case 2: Cu, Cv -> Xu/Xv
        is_aftc = ((self.u_size + self.v_size <= codes[:, -3]) &
                   (codes[:, -3] < self.u_size + self.v_size + self.u_size))
        if torch.any(is_aftc):
            center_codes = codes[is_aftc, -3:-1]
            center_pc = pc[is_aftc]
            center_mask = torch.Tensor([[1, 1] * center_codes.shape[0]]).to(torch.bool)
            new_center_feats = self.get_pc_conv_features(codes=center_codes, pc=center_pc, center_mask=center_mask, k=100)
            next_features[is_aftc] = torch.concat(new_center_feats, dim=0)

        return next_features