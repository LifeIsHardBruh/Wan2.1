#!/usr/bin/env python3
import argparse
import logging
import os

from .cache_demo import load_mask, run_stage
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.text2video import WanT2V
from wan.utils.attention_recorder import AttentionRecorder
from wan.utils.utils import cache_video


def parse_int_list(text):
    if text is None or text.lower() in ('', 'none', 'all'):
        return []
    parts = [p.strip() for p in text.split(',')]
    return [int(p) for p in parts if p]


def resolve_layers(arg, num_layers):
    if arg is None:
        return []
    arg = arg.strip().lower()
    if arg in ('last4', 'last-4'):
        start = max(0, num_layers - 4)
        return list(range(start, num_layers))
    if arg == 'all':
        return []
    return parse_int_list(arg)


def main(args):
    cfg = WAN_CONFIGS[args.task]
    t2v = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
    )
    size_hw = SIZE_CONFIGS[args.size][::-1]
    mask = load_mask(args.mask, args.frame_num, size_hw) if args.mask else None

    os.makedirs(args.out_dir, exist_ok=True)
    heatmap_dir = args.heatmap_dir or os.path.join(args.out_dir,
                                                   "attn_heatmaps")
    capture_steps = parse_int_list(args.attn_steps)
    capture_layers = resolve_layers(args.attn_layers, cfg.num_layers)

    cache_obj = None
    if not args.skip_stage_a:
        logging.info("Running stage A (record cache) ...")
        video_a, cache_obj = run_stage(
            model=t2v,
            prompt=args.prompt_a,
            mask=mask,
            args=args,
            cache_mode='record',
            cache_content=args.cache_content,
            return_cache=True)
        cache_video(video_a.unsqueeze(0),
                    os.path.join(args.out_dir, "stage_a.mp4"))
    activation_cache = cache_obj
    if activation_cache is None:
        activation_cache = args.activation_cache_path or args.cache_dir
    if activation_cache is None:
        raise ValueError(
            "No activation cache available for stage B. Provide --activation_cache_path or run stage A."
        )

    recorder = AttentionRecorder(
        tokenizer=t2v.text_encoder.tokenizer,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        save_dir=heatmap_dir,
        capture_steps=capture_steps,
        capture_layers=capture_layers,
        mode=args.attn_mode,
        chunk_size=args.attn_chunk_size,
    )
    if recorder.num_targets == 0:
        logging.warning("No new tokens detected between prompt A and B.")

    logging.info("Running stage B with attention recording ...")
    video_b = run_stage(
        model=t2v,
        prompt=args.prompt_b,
        mask=mask,
        args=args,
        cache_mode='reuse',
        cache_content=args.cache_content,
        activation_cache=activation_cache,
        attn_recorder=recorder)
    out_name = args.output_filename or "stage_b.mp4"
    cache_video(video_b.unsqueeze(0), os.path.join(args.out_dir, out_name))
    recorder.save()
    logging.info("Saved attention heatmaps to %s", heatmap_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--out_dir", default="/data02/henry/wan_cache/attn")
    parser.add_argument("--task", default="t2v-14B", choices=WAN_CONFIGS.keys())
    parser.add_argument("--size", default="1280*720", choices=SIZE_CONFIGS.keys())
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--prompt_a", required=True)
    parser.add_argument("--prompt_b", required=True)
    parser.add_argument("--mask", required=True,
                        help="Mask path (.pt tensor or image). Black/0 = recompute, white/1 = reuse")
    parser.add_argument("--cache_content", default="full", choices=["full", "background"])
    parser.add_argument("--cache_backend", default="memory", choices=["memory", "disk"])
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--activation_cache_path", default=None)
    parser.add_argument("--skip_stage_a", action="store_true")
    parser.add_argument("--cache_reuse_start_step", type=int, default=0)
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--latent_cache_dir", default=None)
    parser.add_argument("--latent_save_interval", type=int, default=5)
    parser.add_argument("--output_filename", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--heatmap_dir", default=None)
    parser.add_argument("--attn_steps", default="10",
                        help="Comma separated diffusion steps to record (empty for all).")
    parser.add_argument("--attn_layers", default="last4",
                        help="Layer ids (comma list) or 'last4'/'all'. 0-indexed.")
    parser.add_argument("--attn_mode",
                        choices=["aggregated", "fullhead"],
                        default="aggregated")
    parser.add_argument("--attn_chunk_size", type=int, default=1024)
    parser.add_argument("--mask_smooth_kernel", type=int, default=0,
                        help="Odd kernel size (>=3) for smoothing mask boundaries when reusing cache.")
    parser.add_argument("--mask_smooth_mode", default="gaussian",
                        choices=["gaussian", "blend_latent"])
    parser.add_argument("--mask_blend_latent_path", default=None,
                        help="Latent .pt path used when mask_smooth_mode=blend_latent.")
    parser.add_argument("--blur_mask_image_path", default=None)
    parser.add_argument("--save_noise", action="store_true")
    parser.add_argument("--noise_cache_dir", default=None)
    parser.add_argument("--noise_save_interval", type=int, default=1)
    args = parser.parse_args()
    main(args)
