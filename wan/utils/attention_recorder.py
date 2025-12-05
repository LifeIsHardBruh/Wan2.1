import logging
import os
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import torch


class _BlockRecorder:

    def __init__(self,
                 num_tokens: int,
                 seq_len: int,
                 num_heads: int,
                 mode: str = 'aggregated'):
        self.mode = mode
        channels = 1 if mode == 'aggregated' else num_heads
        self.storage = torch.zeros(num_tokens,
                                   channels,
                                   seq_len,
                                   dtype=torch.float32)

    def store(self, seq_indices: torch.Tensor, chunk: torch.Tensor):
        """
        Args:
            seq_indices: Tensor[int] with shape [chunk]
            chunk: Tensor[num_tokens, num_heads, chunk] (full-head) or
                   Tensor[num_tokens, num_heads, chunk] before head-mean (aggregated)
        """
        seq_indices = seq_indices.long()
        if self.mode == 'aggregated':
            data = chunk.mean(dim=1, keepdim=True)
        else:
            data = chunk
        self.storage[:, :, seq_indices] = data


class AttentionRecorderRuntime:

    def __init__(self,
                 parent: "AttentionRecorder",
                 context: Dict,
                 seq_lens: torch.Tensor,
                 grid_sizes: torch.Tensor,
                 fg_mask: Optional[torch.Tensor]):
        self.parent = parent
        self.context = context
        self.seq_info = []
        fg_mask_list = None
        if fg_mask is not None:
            fg_mask_list = fg_mask.detach().cpu()
        for idx in range(seq_lens.size(0)):
            info = {
                'seq_len': int(seq_lens[idx].item()),
                'grid_size': tuple(int(x) for x in grid_sizes[idx].tolist()),
                'fg_mask': None if fg_mask_list is None else fg_mask_list[idx]
            }
            self.seq_info.append(info)
        self._target_idx_cache: Dict[int, torch.Tensor] = {}

    def should_capture_block(self, block_idx: int) -> bool:
        if not self.parent.target_indices:
            return False
        if not self.parent.capture_layers:
            return True
        return block_idx in self.parent.capture_layers

    def _target_indices_on_device(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.parent.target_indices:
            return None
        key = device.index if device.type == 'cuda' else -1
        if key not in self._target_idx_cache:
            self._target_idx_cache[key] = torch.tensor(
                self.parent.target_indices,
                device=device,
                dtype=torch.long)
        return self._target_idx_cache[key]

    def capture(self,
                block_index: int,
                q: torch.Tensor,
                k: torch.Tensor,
                lengths: torch.Tensor,
                indices: Optional[List[torch.Tensor]],
                head_dim: int):
        if block_index is None:
            return
        if not self.should_capture_block(block_index):
            return
        target_idx = self._target_indices_on_device(q.device)
        if target_idx is None or target_idx.numel() == 0:
            return
        chunk_size = self.parent.chunk_size
        k_heads = k.permute(0, 2, 3, 1).contiguous()  # [B, heads, dim, Lk]
        q_heads = q.permute(0, 2, 1, 3).contiguous()  # [B, heads, Lq, dim]
        scale = head_dim**-0.5
        batch = q_heads.size(0)
        num_heads = q_heads.size(1)
        for b in range(batch):
            length = int(lengths[b].item())
            if length == 0:
                continue
            if indices is not None:
                active = indices[b]
                if active is None or active.numel() == 0:
                    continue
                seq_mapping = active[:length]
            else:
                seq_mapping = torch.arange(length, device=q.device)
            q_sample = q_heads[b, :, :length, :]  # [heads, L, dim]
            k_sample = k_heads[b]  # [heads, dim, Lk]
            pos = 0
            while pos < length:
                end = min(pos + chunk_size, length)
                q_chunk = q_sample[:, pos:end, :]  # [heads, chunk, dim]
                scores = torch.matmul(q_chunk.float(),
                                      k_sample.float()) * scale  # [heads, chunk, Lk]
                probs = torch.softmax(scores, dim=-1)
                selected = probs[:, :, target_idx]  # [heads, chunk, T]
                selected = selected.permute(2, 0, 1).contiguous()  # [T, heads, chunk]
                seq_idx = seq_mapping[pos:end].detach().cpu()
                chunk_cpu = selected.detach().cpu()
                sample_info = self.seq_info[b]
                self.parent._add_chunk(
                    step=self.context['step'],
                    block_idx=block_index,
                    sample_idx=b,
                    seq_len=sample_info['seq_len'],
                    grid_size=sample_info['grid_size'],
                    seq_indices=seq_idx,
                    chunk=chunk_cpu,
                    num_heads=num_heads,
                    fg_mask=sample_info['fg_mask'])
                pos = end


class AttentionRecorder:

    def __init__(self,
                 tokenizer,
                 prompt_a: str,
                 prompt_b: str,
                 save_dir: str,
                 capture_steps: Optional[List[int]] = None,
                 capture_layers: Optional[List[int]] = None,
                 mode: str = 'aggregated',
                 chunk_size: int = 1024):
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.capture_steps = set(capture_steps or [])
        self.capture_layers = set(capture_layers or [])
        self.mode = mode
        self.chunk_size = max(1, int(chunk_size))
        self.prompt_b_tokens, self.target_indices = self._compute_new_tokens(
            prompt_a, prompt_b)
        self.target_texts = [
            self.tokenizer.tokenizer.convert_ids_to_tokens(
                self.prompt_b_tokens[idx]) for idx in self.target_indices
        ]
        self.records: Dict[Tuple[int, int, int], _BlockRecorder] = {}
        self.meta: Dict[Tuple[int, int, int], Dict] = {}

    @property
    def num_targets(self) -> int:
        return len(self.target_indices)

    def _clean_text(self, text: str) -> str:
        if hasattr(self.tokenizer, '_clean') and self.tokenizer.clean:
            return self.tokenizer._clean(text)
        return text

    def _tokenize(self, text: str) -> List[int]:
        clean_text = self._clean_text(text)
        encoded = self.tokenizer.tokenizer(
            [clean_text],
            return_tensors='pt',
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.seq_len)
        ids = encoded.input_ids[0]
        mask = encoded.attention_mask[0]
        valid = ids[mask.bool()].tolist()
        return valid

    def _compute_new_tokens(self, prompt_a: str,
                            prompt_b: str) -> Tuple[List[int], List[int]]:
        tokens_a = self._tokenize(prompt_a)
        tokens_b = self._tokenize(prompt_b)
        matcher = SequenceMatcher(a=tokens_a, b=tokens_b)
        added: List[int] = []
        for tag, _, _, j1, j2 in matcher.get_opcodes():
            if tag in ('insert', 'replace'):
                added.extend(range(j1, j2))
        added = [idx for idx in added if idx < len(tokens_b)]
        return tokens_b, added

    def begin_step(self, step_idx: int, branch: str) -> Optional[Dict]:
        if self.num_targets == 0:
            return None
        if branch != 'cond':
            return None
        if self.capture_steps and step_idx not in self.capture_steps:
            return None
        logging.info("[AttentionRecorder] capture scheduled at step %d", step_idx)
        return {'step': step_idx, 'branch': branch}

    def prepare_runtime(self,
                        attn_context: Optional[Dict],
                        seq_lens: torch.Tensor,
                        grid_sizes: torch.Tensor,
                        fg_mask: Optional[torch.Tensor]):
        if attn_context is None:
            return None
        return AttentionRecorderRuntime(self, attn_context, seq_lens,
                                        grid_sizes, fg_mask)

    def _add_chunk(self,
                   step: int,
                   block_idx: int,
                   sample_idx: int,
                   seq_len: int,
                   grid_size: Tuple[int, int, int],
                   seq_indices: torch.Tensor,
                   chunk: torch.Tensor,
                   num_heads: int,
                   fg_mask: Optional[torch.Tensor]):
        key = (step, block_idx, sample_idx)
        if key not in self.records:
            self.records[key] = _BlockRecorder(self.num_targets, seq_len,
                                               num_heads, self.mode)
            self.meta[key] = {
                'grid_size': grid_size,
                'fg_mask': fg_mask,
                'seq_len': seq_len,
                'step': step,
                'block': block_idx,
                'sample_idx': sample_idx
            }
        self.records[key].store(seq_indices, chunk)

    def save(self):
        if not self.records:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        for key, recorder in self.records.items():
            meta = self.meta[key]
            tensor = recorder.storage
            grid = meta['grid_size']
            seq_len = grid[0] * grid[1] * grid[2]
            if tensor.size(-1) != seq_len:
                raise ValueError("Sequence length mismatch when saving attention heatmaps.")
            data = tensor.view(self.num_targets, tensor.size(1), *grid)
            payload = {
                'heatmap': data,
                'mode': self.mode,
                'step': meta['step'],
                'block': meta['block'],
                'grid_size': grid,
                'token_indices': self.target_indices,
                'token_ids': [self.prompt_b_tokens[idx] for idx in self.target_indices],
                'token_texts': self.target_texts,
                'fg_mask': meta['fg_mask'],
            }
            path = os.path.join(
                self.save_dir,
                f"step{meta['step']:03d}_block{meta['block']:02d}_sample{meta['sample_idx']}_{self.mode}.pt"
            )
            torch.save(payload, path)
