#!/usr/bin/env python3
import argparse
import logging
import os
import torch

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.text2video import WanT2V
from wan.utils.utils import cache_video

logging.basicConfig(level=logging.DEBUG)


def load_mask(path, frame_num, size_hw):
    """
    Load mask tensor in one of the following formats:
        - .pt/.pth torch tensor with shape [F, H, W] or [H, W]
        - image file (png/jpg) interpreted as grayscale 0=recompute, 1=reuse
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext in ('.pt', '.pth'):
        mask = torch.load(path, map_location='cpu')
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError("PT mask must have shape [F, H, W] or [H, W].")
        mask = mask.float()
        if mask.size(0) == 1 and frame_num > 1:
            mask = mask.repeat(frame_num, 1, 1)
        elif mask.size(0) == frame_num:
            return mask
        else:
            # allow latent-level masks (#frames = (frame_num - 1) / vae_stride[0] + 1)
            latent_frames = (frame_num - 1) // 4 + 1  # assume vae_stride[0] == 4
            if mask.size(0) != latent_frames:
                raise ValueError(
                    f"PT mask frame count {mask.size(0)} != {frame_num} or latent frames {latent_frames}.")
        return mask

    from PIL import Image
    h, w = size_hw
    img = Image.open(path).convert("L").resize((w, h))
    mask = torch.tensor(img, dtype=torch.float32) / 255.0
    mask = mask.unsqueeze(0).repeat(frame_num, 1, 1)
    return mask


def run_stage(model,
              prompt,
              mask,
              args,
              cache_mode,
              cache_content,
              activation_cache=None,
              return_cache=False,
              attn_recorder=None):
    return model.generate(
        input_prompt=prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        sampling_steps=args.steps,
        guide_scale=args.cfg_scale,
        seed=args.seed,
        offload_model=args.offload_model,
        mask_e=mask,
        cache_mode=cache_mode,
        cache_content=cache_content,
        cache_backend=args.cache_backend,
        cache_dir=args.cache_dir,
        cache_reuse_start_step=args.cache_reuse_start_step,
        save_latents=args.save_latents,
        latent_cache_dir=args.latent_cache_dir,
        latent_save_interval=args.latent_save_interval,
        save_noise=args.save_noise,
        noise_cache_dir=args.noise_cache_dir,
        noise_save_interval=args.noise_save_interval,
        activation_cache=activation_cache,
        return_cache=return_cache,
        attn_recorder=attn_recorder,
        mask_smooth_kernel=args.mask_smooth_kernel,
        mask_smooth_mode=args.mask_smooth_mode,
        mask_blend_latent_path=args.mask_blend_latent_path,
        blur_mask_image_path=args.blur_mask_image_path,
    )


def main(args):
    cfg = WAN_CONFIGS[args.task]
    t2v = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
    )
    size_hw = SIZE_CONFIGS[args.size][::-1]
    mask = load_mask(args.mask, args.frame_num, size_hw) if args.mask else None

    stage_mode = args.stage.lower()
    if stage_mode not in ('a', 'b', 'both', 'baseline'):
        raise ValueError("stage must be one of {'a', 'b', 'both', 'baseline'}.")
    run_a = stage_mode in ('a', 'both')
    run_b = stage_mode in ('b', 'both')
    run_baseline = stage_mode == 'baseline'
    if run_a and not args.prompt_a:
        raise ValueError("prompt_a is required when running stage A.")
    if run_b and not args.prompt_b:
        raise ValueError("prompt_b is required when running stage B.")
    if run_baseline and not (args.prompt_baseline or args.prompt_b or args.prompt_a):
        raise ValueError("Provide --prompt_baseline (or prompt_b/prompt_a) for baseline stage.")

    if args.cache_backend == 'disk' and args.cache_dir is None and run_a:
        args.cache_dir = os.path.join(args.out_dir, "cache_data")

    cache = None
    os.makedirs(args.out_dir, exist_ok=True)

    if run_baseline:
        baseline_prompt = args.prompt_baseline or args.prompt_b or args.prompt_a
        video_base = run_stage(
            model=t2v,
            prompt=baseline_prompt,
            mask=mask,
            args=args,
            cache_mode='none',
            cache_content=args.cache_content,
        )
        baseline_name = args.output_filename or "baseline.mp4"
        cache_video(video_base.unsqueeze(0),
                    os.path.join(args.out_dir, baseline_name))
        print("Baseline run finished.")
        return

    if run_a:
        video_a, cache = run_stage(
            model=t2v,
            prompt=args.prompt_a,
            mask=mask,
            args=args,
            cache_mode='record',
            cache_content=args.cache_content,
            return_cache=True,
        )
        cache_video(video_a.unsqueeze(0), os.path.join(args.out_dir, "stage_a.mp4"))
        print("Stage A finished.")
        if cache.storage_backend == 'disk':
            print(f"Cache stored on disk at: {args.cache_dir}")

    if run_b:
        activation_cache = cache
        if activation_cache is None:
            activation_cache = args.activation_cache_path or args.cache_dir
        if activation_cache is None:
            raise ValueError(
                "Stage B requires an activation cache. Provide --activation_cache_path or run stage A first."
            )
        video_b = run_stage(
            model=t2v,
            prompt=args.prompt_b,
            mask=mask,
            args=args,
            cache_mode='reuse',
            cache_content=args.cache_content,
            activation_cache=activation_cache,
        )
        out_name = args.output_filename or "stage_b.mp4"
        cache_video(video_b.unsqueeze(0), os.path.join(args.out_dir, out_name))
        print("Stage B finished. Cache reused from:",
              activation_cache if isinstance(activation_cache, str) else "in-memory object")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--out_dir", default="/data02/henry/wan_cache/cache_demo")
    parser.add_argument("--task", default="t2v-14B", choices=WAN_CONFIGS.keys())
    parser.add_argument("--size", default="1280*720", choices=SIZE_CONFIGS.keys())
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--prompt_a", default=None)
    parser.add_argument("--prompt_b", default=None)
    parser.add_argument("--prompt_baseline", default=None,
                        help="Prompt used when --stage baseline (fallback to prompt_b then prompt_a).")
    parser.add_argument(
        "--mask",
        help="Mask path (.pt tensor or image). Black/0 = recompute, white/1 = reuse")
    parser.add_argument("--cache_content", default="full", choices=["full", "background"])
    parser.add_argument("--cache_backend", default="memory", choices=["memory", "disk"])
    parser.add_argument("--cache_dir", default=None, help="Directory for disk cache storage")
    parser.add_argument("--stage", default="both", choices=["a", "b", "both", "baseline"],
                        help="Run only stage 'a', only stage 'b', both, or a single baseline pass without caching.")
    parser.add_argument("--activation_cache_path", default=None,
                        help="Existing cache directory to reuse when running stage B standalone.")
    parser.add_argument("--cache_reuse_start_step", type=int, default=0,
                        help="B 阶段在该步之后才开始读取缓存（reuse 模式）")
    parser.add_argument("--save_latents", action="store_true",
                        help="Enable saving intermediate latents for inspection.")
    parser.add_argument("--latent_cache_dir", default=None,
                        help="Directory to store latent tensors when saving is enabled.")
    parser.add_argument("--latent_save_interval", type=int, default=5,
                        help="Save latents every N steps (also saves final step).")
    parser.add_argument("--output_filename", default=None,
                        help="Override default video filename (baseline/stage B).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--mask_smooth_kernel", type=int, default=0,
                        help="Odd kernel size (>=3) for boundary smoothing; 0 disables smoothing.")
    parser.add_argument("--mask_smooth_mode", default="gaussian",
                        choices=["gaussian", "blend_latent"],
                        help="Smoothing mode applied after cache reuse starts.")
    parser.add_argument("--mask_blend_latent_path", default=None,
                        help="Latent .pt path used when mask_smooth_mode=blend_latent.")
    parser.add_argument("--blur_mask_image_path", default=None,
                        help="Optional path to dump the blur mask as an image.")
    parser.add_argument("--save_noise", action="store_true",
                        help="Save noise prediction tensor for each step (stage A).")
    parser.add_argument("--noise_cache_dir", default=None,
                        help="Directory to store per-step noise tensors.")
    parser.add_argument("--noise_save_interval", type=int, default=1,
                        help="Save noises every N steps (default 1).")
    args = parser.parse_args()
    main(args)
