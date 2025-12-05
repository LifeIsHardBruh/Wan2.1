# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class BlockActivationCache:

    def __init__(self,
                 num_layers,
                 seq_len,
                 mode='background',
                 background_mask=None,
                 storage_dtype=torch.float16,
                 storage_backend='memory',
                 cache_dir=None,
                 initialize_storage=True):
        """
        Args:
            num_layers (int): Number of transformer blocks.
            seq_len (int): Maximum sequence length per sample.
            mode (str): 'background' caches only masked background tokens, 'full' caches full block outputs.
            background_mask (Tensor, optional): Required when mode == 'background'. Boolean tensor [B, seq_len].
            storage_dtype (torch.dtype): Dtype used when storing cached activations on CPU/disk.
            storage_backend (str): 'memory' keeps tensors in RAM, 'disk' streams tensors to cache_dir.
            cache_dir (str, optional): Directory used when storage_backend == 'disk'.
            initialize_storage (bool): Internal flag to prevent overwriting metadata when reloading from disk.
        """
        mode = (mode or 'background').lower()
        if mode not in ('background', 'full'):
            raise ValueError(f"Unsupported cache mode {mode}.")
        if mode == 'background':
            if background_mask is None:
                raise ValueError(
                    "background_mask must be provided when using 'background' cache mode."
                )
            background_mask = background_mask.to(device='cpu', dtype=torch.bool)
            self.background_mask = background_mask
        else:
            self.background_mask = None

        storage_backend = (storage_backend or 'memory').lower()
        if storage_backend not in ('memory', 'disk'):
            raise ValueError(f"Unsupported storage backend {storage_backend}.")
        if storage_backend == 'disk' and cache_dir is None:
            raise ValueError("cache_dir must be provided when using disk storage.")

        self.mode = mode
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.storage_dtype = storage_dtype
        self.storage_backend = storage_backend
        self.cache_dir = cache_dir

        if storage_backend == 'memory':
            self.storage = {}
        else:
            os.makedirs(cache_dir, exist_ok=True)
            self.storage = None
            self._meta_path = os.path.join(cache_dir, 'meta.pt')
            if initialize_storage:
                self._write_metadata()

    @classmethod
    def load(cls, cache_dir):
        meta_path = os.path.join(cache_dir, 'meta.pt')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Cache metadata not found at {meta_path}")
        meta = torch.load(meta_path)
        mode = meta['mode']
        seq_len = meta['seq_len']
        num_layers = meta['num_layers']
        storage_dtype = getattr(torch, meta['storage_dtype'])
        background_mask = None
        if mode == 'background':
            background_mask = torch.load(
                os.path.join(cache_dir, 'background_mask.pt'))
        cache = cls(
            num_layers=num_layers,
            seq_len=seq_len,
            mode=mode,
            background_mask=background_mask,
            storage_dtype=storage_dtype,
            storage_backend='disk',
            cache_dir=cache_dir,
            initialize_storage=False)
        return cache

    def _write_metadata(self):
        meta = {
            'mode': self.mode,
            'seq_len': self.seq_len,
            'num_layers': self.num_layers,
            'storage_dtype': self.storage_dtype.__repr__().split('.')[-1]
        }
        torch.save(meta, self._meta_path)
        if self.mode == 'background' and self.background_mask is not None:
            torch.save(self.background_mask,
                       os.path.join(self.cache_dir, 'background_mask.pt'))

    def _disk_path(self, step_idx, branch, block_idx):
        filename = f"step{step_idx:05d}_{branch}_block{block_idx:03d}.pt"
        return os.path.join(self.cache_dir, filename)

    def _store_payload(self, block_cpu):
        if self.mode == 'background':
            payload = []
            for sample_idx in range(block_cpu.size(0)):
                mask = self.background_mask[sample_idx]
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                if idx.numel() == 0:
                    payload.append(None)
                else:
                    payload.append(block_cpu[sample_idx, idx].contiguous())
            return payload
        return block_cpu.contiguous()

    def write(self, step_idx, branch, block_idx, block_output):
        block_cpu = block_output.detach().to(device='cpu', dtype=self.storage_dtype)
        payload = self._store_payload(block_cpu)

        if self.storage_backend == 'memory':
            key = (step_idx, branch)
            if key not in self.storage:
                self.storage[key] = [None] * self.num_layers
            self.storage[key][block_idx] = payload
        else:
            torch.save(
                {
                    'mode': self.mode,
                    'payload': payload
                }, self._disk_path(step_idx, branch, block_idx))

    def read(self, step_idx, branch, block_idx, device, dtype):
        if self.storage_backend == 'memory':
            key = (step_idx, branch)
            per_block = self.storage.get(key)
            if per_block is None:
                raise KeyError(
                    f"No cached activations for step {step_idx}, branch {branch}.")
            payload = per_block[block_idx]
        else:
            path = self._disk_path(step_idx, branch, block_idx)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Cache file not found for step {step_idx}, branch {branch}, block {block_idx}"
                )
            payload = torch.load(path)['payload']

        if payload is None:
            return None

        if self.mode == 'background':
            batch_size = len(payload)
            dim = None
            for sample in payload:
                if sample is not None:
                    dim = sample.size(-1)
                    break
            if dim is None:
                return None
            out = torch.zeros(
                batch_size,
                self.seq_len,
                dim,
                device=device,
                dtype=dtype)
            for sample_idx, sample in enumerate(payload):
                if sample is None:
                    continue
                mask = self.background_mask[sample_idx]
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                if idx.numel() > 0:
                    out[sample_idx, idx] = sample.to(device=device, dtype=dtype)
            return out

        return payload.to(device=device, dtype=dtype)


class LatentCacheWriter:

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, tensor, step_idx):
        out = tensor.detach().cpu().contiguous()
        fname = self.root / f"latent_step{step_idx:03d}.pt"
        torch.save(out, fname)
        size_mb = out.numel() * out.element_size() / 1e6
        logging.info(
            "[LatentCacheWriter] saved %s shape=%s dtype=%s size=%.1f MB",
            fname,
            tuple(out.shape),
            out.dtype,
            size_mb,
        )
        return fname


class NoiseCacheWriter:

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, tensor, step_idx):
        out = tensor.detach().cpu().contiguous()
        fname = self.root / f"noise_step{step_idx:03d}.pt"
        torch.save(out, fname)
        size_mb = out.numel() * out.element_size() / 1e6
        logging.info(
            "[NoiseCacheWriter] saved %s shape=%s dtype=%s size=%.1f MB",
            fname,
            tuple(out.shape),
            out.dtype,
            size_mb,
        )
        return fname

def _save_blur_mask_image(blur_mask, path):
    torch.save(blur_mask.cpu(), path + ".pt")
    arr = blur_mask.mean(dim=0)
    arr = arr.clamp_(0.0, 1.0).cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _build_token_mask(self, mask_e, target_shape):
        if mask_e is None:
            return None, None
        if isinstance(mask_e, torch.Tensor):
            mask = mask_e.detach().cpu().float()
        else:
            mask = torch.tensor(mask_e, dtype=torch.float32)
        if mask.dim() != 3:
            raise ValueError("mask_e must have shape [F, H, W].")
        is_latent_sized = (mask.shape[0] == target_shape[1]
                           and mask.shape[1] == target_shape[2]
                           and mask.shape[2] == target_shape[3])
        mask = mask.unsqueeze(0).unsqueeze(0)
        if not is_latent_sized:
            mask = F.interpolate(
                mask,
                size=(target_shape[1], target_shape[2], target_shape[3]),
                mode='nearest')
        f_tokens = target_shape[1] // self.patch_size[0]
        h_tokens = target_shape[2] // self.patch_size[1]
        w_tokens = target_shape[3] // self.patch_size[2]
        latent_mask = mask.squeeze(0).squeeze(0)
        mask = mask.view(1, 1, f_tokens, self.patch_size[0], h_tokens,
                         self.patch_size[1], w_tokens, self.patch_size[2])
        mask = mask.amax(dim=3).amax(dim=4).amax(dim=5)
        mask = mask.reshape(-1)
        return mask > 0.5, latent_mask

    def _build_gaussian_kernel(self, kernel_size, device):
        kernel_size = max(3, int(kernel_size) // 2 * 2 + 1)
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords = coords - (kernel_size - 1) / 2
        grid = coords[:, None]**2 + coords[None, :]**2
        sigma = kernel_size / 3
        kernel = torch.exp(-grid / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    # def _build_gaussian_kernel(self, kernel_size, device):
    #     kernel_size = max(3, int(kernel_size) // 2 * 2 + 1)
    #     kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32, device=device)
    #     kernel = kernel / kernel.numel()  # 每个元素 = 1/(k*k)
    #     return kernel.view(1, 1, kernel_size, kernel_size)

    def _prepare_boundary_weight(self, latent_mask, kernel):
        if latent_mask is None:
            return None, None
        mask = latent_mask.unsqueeze(1)
        blurred = F.conv2d(
            mask,
            kernel,
            padding=kernel.size(-1) // 2).squeeze(1).clamp_(0.0, 1.0)
        boundary = (blurred * (1.0 - blurred)) * 4.0
        boundary = boundary * 0.2
        return boundary.clamp_(0.0, 1.0), blurred.clamp_(0.0, 1.0)

    def _smooth_noise_with_mask(self, noise, boundary_weight, kernel):
        if boundary_weight is None or kernel is None:
            return noise
        c, f, h, w = noise.shape
        reshaped = noise.view(c * f, 1, h, w)
        smoothed = F.conv2d(
            reshaped,
            kernel,
            padding=kernel.size(-1) // 2).view_as(noise)
        weight = boundary_weight.to(noise.device)
        while weight.dim() < noise.dim():
            weight = weight.unsqueeze(0)
        return noise * (1.0 - weight) + smoothed * weight

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 mask_e=None,
                 cache_mode='none',
                 cache_content='background',
                 cache_backend='memory',
                 cache_dir=None,
                 cache_reuse_start_step=0,
                 save_latents=False,
                 latent_cache_dir=None,
                 latent_save_interval=5,
                 save_noise=False,
                 noise_cache_dir=None,
                 noise_save_interval=1,
                 activation_cache=None,
                 return_cache=False,
                 attn_recorder=None,
                 mask_smooth_kernel=0,
                 mask_smooth_mode="gaussian",
                 mask_blend_latent_path=None,
                 blur_mask_image_path=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            mask_e (`Tensor`, *optional*):
                Foreground mask with shape [F, H, W] indicating region E (e.g. dog).
            cache_mode (`str`, *optional*, defaults to 'none'):
                One of {'none', 'record', 'reuse'} controlling activation caching.
            cache_content (`str`, *optional*, defaults to 'background'):
                Either 'background' (only cache reusable background tokens) or 'full' (store entire block outputs).
            cache_backend (`str`, *optional*, defaults to 'memory'):
                Either 'memory' (keep cache in RAM) or 'disk' (stream per-block tensors to cache_dir).
            cache_dir (`str`, *optional*, defaults to None):
                Directory used when `cache_backend == 'disk'`. Required when recording disk caches.
            cache_reuse_start_step (`int`, *optional*, defaults to 0):
                For reuse mode, number of initial diffusion steps to run without loading cached activations.
            save_latents (`bool`, *optional*, defaults to False):
                When True, saves intermediate latents during sampling for inspection/debugging.
            latent_cache_dir (`str`, *optional*, defaults to None):
                Directory to store latent tensors when `save_latents` is enabled.
            latent_save_interval (`int`, *optional*, defaults to 5):
                Save latents every N steps (also saves the very last step).
            activation_cache (`BlockActivationCache`, *optional*):
                Cache object (or cache directory string) used when recording or reusing activations.
            return_cache (`bool`, *optional*, defaults to False):
                When True and cache_mode == 'record', also return the populated cache.
            attn_recorder (`AttentionRecorder`, *optional*):
                Optional recorder used to capture cross-attention weights during sampling.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, BlockActivationCache]:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        token_mask_fg, latent_mask = self._build_token_mask(mask_e, target_shape)
        token_mask_list = [token_mask_fg] if token_mask_fg is not None else None
        background_mask_padded = None
        if token_mask_fg is not None:
            padded_fg = torch.zeros(seq_len, dtype=torch.bool)
            padded_fg[:token_mask_fg.numel()] = token_mask_fg
            background_mask_padded = (~padded_fg).unsqueeze(0)
        boundary_weight = None
        blur_mask = None
        gaussian_kernel = None
        if latent_mask is not None and mask_smooth_kernel is not None and mask_smooth_kernel >= 3:
            kernel_size = int(mask_smooth_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gaussian_kernel = self._build_gaussian_kernel(kernel_size,
                                                          self.device)
            latent_mask = latent_mask.to(self.device)
            boundary_weight, blur_mask = self._prepare_boundary_weight(
                latent_mask, gaussian_kernel)
            if blur_mask_image_path is not None and not getattr(
                    self, "_blur_mask_saved", False):
                _save_blur_mask_image(blur_mask, blur_mask_image_path)
                self._blur_mask_saved = True

        cache_mode = (cache_mode or 'none').lower()
        if cache_mode not in ('none', 'record', 'reuse'):
            raise ValueError(f"Unsupported cache_mode {cache_mode}.")

        cache_content = (cache_content or 'background').lower()
        if cache_content not in ('background', 'full'):
            raise ValueError(f"Unsupported cache_content {cache_content}.")

        cache_backend = (cache_backend or 'memory').lower()
        if cache_backend not in ('memory', 'disk'):
            raise ValueError(f"Unsupported cache_backend {cache_backend}.")

        if isinstance(activation_cache, str):
            cache_obj = BlockActivationCache.load(activation_cache)
        else:
            cache_obj = activation_cache

        if cache_obj is not None:
            cache_content = cache_obj.mode
            cache_backend = cache_obj.storage_backend

        cache_reuse_start_step = int(cache_reuse_start_step)
        if cache_reuse_start_step < 0:
            raise ValueError("cache_reuse_start_step must be >= 0.")

        latent_save_interval = max(1, int(latent_save_interval))
        noise_save_interval = max(1, int(noise_save_interval))

        if cache_mode == 'record' and cache_content == 'background' and token_mask_fg is None:
            raise ValueError(
                "mask_e is required to record background-only caches. Provide mask_e or use cache_content='full'."
            )
        if cache_mode == 'reuse' and token_mask_fg is None:
            raise ValueError(
                "mask_e is required when reusing cached activations to decide which regions to recompute.")

        if cache_mode == 'record' and cache_obj is None:
            if cache_backend == 'disk' and cache_dir is None:
                raise ValueError("cache_dir must be provided when cache_backend='disk'.")
            cache_obj = BlockActivationCache(
                num_layers=len(self.model.blocks),
                seq_len=seq_len,
                mode=cache_content,
                background_mask=background_mask_padded.cpu()
                if background_mask_padded is not None else None,
                storage_backend=cache_backend,
                cache_dir=cache_dir)
        elif cache_mode == 'reuse':
            if cache_obj is None:
                raise ValueError("activation_cache must be provided for reuse mode.")
            cache_content = cache_obj.mode
            if cache_obj.mode == 'background' and background_mask_padded is None:
                raise ValueError(
                    "mask_e must be provided when reusing a background-only cache.")

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        def writer_factory(step_idx, branch):
            if cache_mode != 'record' or cache_obj is None:
                return None
            return lambda block_idx, tensor: cache_obj.write(
                step_idx, branch, block_idx, tensor)

        cache_load_stats = {'cond': {}, 'uncond': {}}

        def reader_factory(step_idx, branch):
            if cache_mode != 'reuse' or cache_obj is None:
                return None
            if step_idx < cache_reuse_start_step:
                return None
            def reader(block_idx, device, dtype):
                start = time.time()
                tensor = cache_obj.read(step_idx, branch, block_idx, device, dtype)
                duration = time.time() - start
                if tensor is not None:
                    cache_load_stats[branch][step_idx] = cache_load_stats[
                        branch].get(step_idx, 0.0) + duration
                return tensor

            return reader

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")
            total_steps = len(timesteps)
            latent_writer = None
            if save_latents and self.rank == 0:
                latent_dir = latent_cache_dir or os.path.join(
                    os.getcwd(), "latent_cache")
                latent_writer = LatentCacheWriter(latent_dir)
            noise_writer = None
            if save_noise and self.rank == 0:
                noise_dir = noise_cache_dir or os.path.join(
                    os.getcwd(), "noise_cache")
                noise_writer = NoiseCacheWriter(noise_dir)

            # sample videos
            latents = noise

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                token_mask_for_step = None
                if token_mask_list is not None and step_idx >= cache_reuse_start_step:
                    token_mask_for_step = token_mask_list
                attn_ctx_cond = None
                if attn_recorder is not None:
                    attn_ctx_cond = attn_recorder.begin_step(
                        step_idx=step_idx, branch='cond')
                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=seq_len,
                    token_mask_fg=token_mask_for_step,
                    cache_writer=writer_factory(step_idx, 'cond'),
                    cache_reader=reader_factory(step_idx, 'cond'),
                    attn_recorder=attn_recorder,
                    attn_context=attn_ctx_cond)[0]
                attn_ctx_uncond = None
                if attn_recorder is not None:
                    attn_ctx_uncond = attn_recorder.begin_step(
                        step_idx=step_idx, branch='uncond')
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    context=context_null,
                    seq_len=seq_len,
                    token_mask_fg=token_mask_for_step,
                    cache_writer=writer_factory(step_idx, 'uncond'),
                    cache_reader=reader_factory(step_idx, 'uncond'),
                    attn_recorder=attn_recorder,
                    attn_context=attn_ctx_uncond)[0]

                for branch in ('cond', 'uncond'):
                    if cache_mode == 'reuse' and cache_obj is not None:
                        total = cache_load_stats[branch].pop(step_idx, 0.0)
                        if total > 0:
                            logging.debug(
                                f"[CacheLoad] step={step_idx} branch={branch} total_load_time={total * 1000:.2f}ms")

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                if noise_writer is not None and self.rank == 0:
                    if (step_idx % noise_save_interval == 0) or (step_idx
                                                                 == total_steps - 1):
                        noise_writer.save(noise_pred, step_idx)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                temp_x0 = temp_x0.squeeze(0)
                if (step_idx >= 0
                        and step_idx < 20
                        and mask_smooth_kernel is not None
                        and mask_smooth_kernel >= 3):
                    if (mask_smooth_mode == "gaussian"
                            and boundary_weight is not None
                            and gaussian_kernel is not None):
                        logging.info(
                            "[MaskSmooth][gaussian][noise] step %d", step_idx)
                        temp_x0 = self._smooth_noise_with_mask(
                            temp_x0, boundary_weight, gaussian_kernel)
                    elif (mask_smooth_mode == "blend_latent"
                          and blur_mask is not None
                          and mask_blend_latent_path is not None):
                        blend_path = mask_blend_latent_path
                        blend_tensor = None
                        if os.path.isdir(blend_path):
                            fname_noise = os.path.join(
                                blend_path, f"noise_step{step_idx:03d}.pt")
                            fname_latent = os.path.join(
                                blend_path, f"latent_step{step_idx:03d}.pt")
                            target_file = fname_noise if os.path.exists(
                                fname_noise) else fname_latent
                            if not os.path.exists(target_file):
                                raise FileNotFoundError(
                                    f"blend tensor not found (noise/latent) for step {step_idx} in {blend_path}"
                                )
                            key = ('dir', step_idx)
                            cache_attr = "_blend_noise_cache"
                            if not hasattr(self, cache_attr):
                                setattr(self, cache_attr, {})
                            cache = getattr(self, cache_attr)
                            if key not in cache:
                                cache[key] = torch.load(
                                    target_file, map_location=self.device)
                            blend_tensor = cache[key]
                            logging.info("Loaded blend tensor %s",
                                         target_file)
                        else:
                            if not hasattr(self, "_blend_noise_cached"):
                                blend_tensor = torch.load(
                                    blend_path, map_location=self.device)
                                self._blend_noise_cached = blend_tensor
                            blend_tensor = self._blend_noise_cached
                        blend_tensor = blend_tensor.to(
                            device=temp_x0.device, dtype=temp_x0.dtype)
                        if blend_tensor.shape != temp_x0.shape:
                            raise ValueError(
                                f"blend tensor shape {blend_tensor.shape} != temp_x0 {temp_x0.shape}"
                            )
                        weight = blur_mask.to(temp_x0.device)
                        while weight.dim() < temp_x0.dim():
                            weight = weight.unsqueeze(0)
                        logging.info(
                            "[MaskSmooth][blend_latent][noise] step %d",
                            step_idx)
                        temp_x0 = weight * temp_x0 + (1.0 -
                                                            weight) * blend_tensor
                        temp_x0 = self._smooth_noise_with_mask(
                            temp_x0, boundary_weight, gaussian_kernel)
                latents = [temp_x0]

                if latent_writer is not None and self.rank == 0:
                    if (step_idx % latent_save_interval == 0) or (step_idx
                                                                 == total_steps - 1):
                        latent_writer.save(latents[0], step_idx)

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        output_video = videos[0] if self.rank == 0 else None
        if return_cache:
            return output_video, cache_obj
        return output_video
