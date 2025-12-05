# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def _pack_by_mask(tensor, mask, seq_lens):
    if mask is None:
        return tensor, None, None, False

    mask = mask.bool()
    device = tensor.device
    batch, max_len = tensor.shape[:2]
    head_dim = tensor.size(3)
    num_heads = tensor.size(2)
    lengths = []
    indices = []
    max_active = 0
    for i in range(batch):
        valid_len = int(seq_lens[i].item())
        idx = torch.nonzero(mask[i, :valid_len], as_tuple=False).flatten()
        indices.append(idx)
        active = idx.numel()
        lengths.append(active)
        if active > max_active:
            max_active = active

    lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.int32)
    if max_active == 0:
        return None, lengths_tensor, indices, True

    packed = tensor.new_zeros((batch, max_active, num_heads, head_dim))
    for i, idx in enumerate(indices):
        if idx.numel() > 0:
            packed[i, :idx.numel()] = tensor[i, idx]
    return packed, lengths_tensor, indices, False


def _scatter_by_mask(src, indices, target):
    if src is None or indices is None:
        return target
    for i, idx in enumerate(indices):
        if idx is not None and idx.numel() > 0:
            target[i, idx] = src[i, :idx.numel()]
    return target


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, query_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        rope_q = rope_apply(q, grid_sizes, freqs)
        rope_k = rope_apply(k, grid_sizes, freqs)

        if query_mask is not None:
            packed_q, q_lengths, indices, all_empty = _pack_by_mask(
                rope_q, query_mask, seq_lens)
            if all_empty:
                attn_out = rope_q.new_zeros((b, s, n, d))
            else:
                attn_fg = flash_attention(
                    q=packed_q,
                    k=rope_k,
                    v=v,
                    q_lens=q_lengths,
                    k_lens=seq_lens.to(dtype=torch.int32, device=rope_q.device),
                    window_size=self.window_size)
                attn_out = rope_q.new_zeros((b, s, n, d))
                attn_out = _scatter_by_mask(attn_fg, indices, attn_out)
        else:
            attn_out = flash_attention(
                q=rope_q,
                k=rope_k,
                v=v,
                k_lens=seq_lens.to(dtype=torch.int32, device=rope_q.device),
                window_size=self.window_size)

        # output
        attn_out = attn_out.flatten(2)
        attn_out = self.o(attn_out)
        if query_mask is not None:
            attn_out = attn_out * query_mask.unsqueeze(-1).to(attn_out.dtype)
        return attn_out


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self,
                x,
                context,
                context_lens,
                seq_lens,
                query_mask=None,
                block_index=None,
                attn_runtime=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if query_mask is not None:
            packed_q, q_lengths, indices, all_empty = _pack_by_mask(
                q, query_mask, seq_lens)
            if all_empty:
                attn = q.new_zeros(q.shape)
            else:
                k_lens = None if context_lens is None else context_lens.to(
                    dtype=torch.int32, device=q.device)
                attn_fg = flash_attention(
                    q=packed_q,
                    k=k,
                    v=v,
                    q_lens=q_lengths,
                    k_lens=k_lens)
                attn = q.new_zeros(q.shape)
                attn = _scatter_by_mask(attn_fg, indices, attn)
                if attn_runtime is not None:
                    attn_runtime.capture(
                        block_index=block_index,
                        q=packed_q,
                        k=k,
                        lengths=q_lengths,
                        indices=indices,
                        head_dim=self.head_dim)
        else:
            k_lens = None if context_lens is None else context_lens.to(
                dtype=torch.int32, device=q.device)
            attn = flash_attention(q, k, v, k_lens=k_lens)
            if attn_runtime is not None:
                attn_runtime.capture(
                    block_index=block_index,
                    q=q,
                    k=k,
                    lengths=seq_lens,
                    indices=None,
                    head_dim=self.head_dim)

        # output
        attn = attn.flatten(2)
        attn = self.o(attn)
        if query_mask is not None:
            attn = attn * query_mask.unsqueeze(-1).to(attn.dtype)
        return attn


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, seq_lens, query_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        if query_mask is not None:
            packed_q, q_lengths, indices, all_empty = _pack_by_mask(
                q, query_mask, seq_lens)
            if all_empty:
                attn = q.new_zeros(q.shape)
                img_attn = q.new_zeros(q.shape)
            else:
                k_lens = None if context_lens is None else context_lens.to(
                    dtype=torch.int32, device=q.device)
                attn_fg = flash_attention(
                    q=packed_q,
                    k=k,
                    v=v,
                    q_lens=q_lengths,
                    k_lens=k_lens)
                img_fg = flash_attention(
                    q=packed_q,
                    k=k_img,
                    v=v_img,
                    q_lens=q_lengths,
                    k_lens=None)
                attn = q.new_zeros(q.shape)
                img_attn = q.new_zeros(q.shape)
                attn = _scatter_by_mask(attn_fg, indices, attn)
                img_attn = _scatter_by_mask(img_fg, indices, img_attn)
        else:
            k_lens = None if context_lens is None else context_lens.to(
                dtype=torch.int32, device=q.device)
            img_attn = flash_attention(q, k_img, v_img, k_lens=None)
            attn = flash_attention(q, k, v, k_lens=k_lens)

        # output
        attn = attn.flatten(2)
        img_attn = img_attn.flatten(2)
        attn = attn + img_attn
        attn = self.o(attn)
        if query_mask is not None:
            attn = attn * query_mask.unsqueeze(-1).to(attn.dtype)
        return attn


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        fg_mask=None,
        bg_mask=None,
        cache_writer=None,
        cache_reader=None,
        block_index=None,
        attn_runtime=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if (cache_writer is not None or cache_reader is not None) and block_index is None:
            raise ValueError("block_index must be provided when using cache callbacks.")
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        mask_fg_unsq = None
        if fg_mask is not None:
            mask_fg_unsq = fg_mask.unsqueeze(-1).to(x.dtype)

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs, query_mask=fg_mask)
        with amp.autocast(dtype=torch.float32):
            update = y * e[2]
            if mask_fg_unsq is not None:
                update = update * mask_fg_unsq
            x = x + update

        cached_bg = None
        if cache_reader is not None:
            cached_bg = cache_reader(block_index, x.device, x.dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, cached_bg,
                           attn_runtime):
            attn = self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                seq_lens,
                query_mask=fg_mask,
                block_index=block_index,
                attn_runtime=attn_runtime)
            x = x + (attn * mask_fg_unsq if mask_fg_unsq is not None else attn)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            if mask_fg_unsq is not None:
                y = y * mask_fg_unsq
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            if cached_bg is not None and bg_mask is not None:
                x = torch.where(bg_mask.unsqueeze(-1), cached_bg, x)
            elif cached_bg is not None:
                x = cached_bg
            return x

        x = cross_attn_ffn(x, context, context_lens, e, cached_bg,
                           attn_runtime)
        if cache_writer is not None:
            cache_writer(block_index, x)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        token_mask_fg=None,
        cache_writer=None,
        cache_reader=None,
        attn_recorder=None,
        attn_context=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            token_mask_fg (List[Tensor], *optional*):
                Foreground token masks aligned with flattened patch tokens for each sample
            cache_writer (Callable, *optional*):
                Callback used during activation recording, signature `(block_idx, tensor)`
            cache_reader (Callable, *optional*):
                Callback returning cached background activations during reuse, signature
                `(block_idx, device, dtype) -> Tensor or None`
            attn_recorder (AttentionRecorder, *optional*):
                Optional recorder used to capture cross-attention weights.
            attn_context (dict, *optional*):
                Per-call recorder context generated upstream (e.g. per step/branch).

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        fg_mask = None
        bg_mask = None
        if token_mask_fg is not None:
            masks_fg = []
            masks_bg = []
            device = x.device
            for mask, length in zip(token_mask_fg, seq_lens):
                if mask is None:
                    raise ValueError("token_mask_fg entries must be tensors when provided.")
                mask = mask.to(device=device, dtype=torch.bool)
                if mask.numel() != int(length.item()):
                    raise ValueError("Foreground mask length must match token length.")
                padded = torch.zeros(seq_len, dtype=torch.bool, device=device)
                padded[:mask.numel()] = mask
                masks_fg.append(padded)
                masks_bg.append(~padded)
            fg_mask = torch.stack(masks_fg)
            bg_mask = torch.stack(masks_bg)

        attn_runtime = None
        if attn_recorder is not None and attn_context is not None:
            attn_runtime = attn_recorder.prepare_runtime(attn_context, seq_lens,
                                                         grid_sizes, fg_mask)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            fg_mask=fg_mask,
            bg_mask=bg_mask,
            cache_writer=cache_writer,
            cache_reader=cache_reader,
            attn_runtime=attn_runtime)

        for idx, block in enumerate(self.blocks):
            x = block(x, block_index=idx, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
