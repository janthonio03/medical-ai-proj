import math

import numpy as np
import torch
import torch.nn.functional as F


def mask_to_patch_indices(
    mask_hw,
    image_h,
    image_w,
    patch_size,
    threshold=0.5,
    add_cls_offset=True,
):
    if not isinstance(mask_hw, np.ndarray):
        mask_hw = np.asarray(mask_hw)

    if mask_hw.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask_hw.shape}")

    grid_h = image_h // patch_size
    grid_w = image_w // patch_size

    mask = torch.from_numpy(mask_hw.astype(np.float32))[None, None]

    resized = F.interpolate(
        mask,
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    keep = resized > threshold
    idx = torch.nonzero(keep.flatten(), as_tuple=False).flatten()

    if add_cls_offset:
        idx = idx + 1

    return idx.long(), resized


class CLIPVisualHighlighter:
    def __init__(self, vision_tower, beta=2.0):
        if beta <= 0:
            raise ValueError("beta must be > 0")

        self.vision_tower = vision_tower
        self.beta = float(beta)
        self.log_beta = math.log(float(beta))

        self.enabled = False
        self.highlight_idx = None

        self._patched = False
        self._original_forwards = {}

    def set_indices(self, indices):
        self.highlight_idx = indices

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def patch(self):
        if self._patched:
            return

        count = 0

        for _, module in self.vision_tower.named_modules():
            if module.__class__.__name__ != "CLIPAttention":
                continue

            if module in self._original_forwards:
                continue

            self._original_forwards[module] = module.forward
            module.forward = self._make_forward(module)
            count += 1

        if count == 0:
            raise RuntimeError("No CLIPAttention modules found")

        self._patched = True
        print("Patched CLIPAttention layers:", count)

    def unpatch(self):
        if not self._patched:
            return

        for module, forward in self._original_forwards.items():
            module.forward = forward

        self._original_forwards.clear()
        self._patched = False

    def _make_forward(self, module):
        highlighter = self

        def patched_forward(
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        ):
            bsz, tgt_len, embed_dim = hidden_states.size()

            query_states = module.q_proj(hidden_states) * module.scale
            key_states = module._shape(module.k_proj(hidden_states), -1, bsz)
            value_states = module._shape(module.v_proj(hidden_states), -1, bsz)

            proj_shape = (bsz * module.num_heads, -1, module.head_dim)

            query_states = module._shape(
                query_states,
                tgt_len,
                bsz,
            ).view(*proj_shape)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)

            src_len = key_states.size(1)

            attn_weights = torch.bmm(
                query_states,
                key_states.transpose(1, 2),
            )

            expected = (
                bsz * module.num_heads,
                tgt_len,
                src_len,
            )

            if attn_weights.size() != expected:
                raise ValueError(
                    f"unexpected attention shape: {attn_weights.size()} expected {expected}"
                )

            if causal_attention_mask is not None:
                attn_weights = (
                    attn_weights.view(
                        bsz,
                        module.num_heads,
                        tgt_len,
                        src_len,
                    )
                    + causal_attention_mask
                )
                attn_weights = attn_weights.view(
                    bsz * module.num_heads,
                    tgt_len,
                    src_len,
                )

            if attention_mask is not None:
                attn_weights = (
                    attn_weights.view(
                        bsz,
                        module.num_heads,
                        tgt_len,
                        src_len,
                    )
                    + attention_mask
                )
                attn_weights = attn_weights.view(
                    bsz * module.num_heads,
                    tgt_len,
                    src_len,
                )

            if (
                highlighter.enabled
                and highlighter.highlight_idx is not None
                and highlighter.highlight_idx.numel() > 0
            ):
                idx = highlighter.highlight_idx.to(attn_weights.device)
                idx = idx[(idx >= 0) & (idx < src_len)]

                if idx.numel() > 0:
                    attn_weights[..., idx] = (
                        attn_weights[..., idx] + highlighter.log_beta
                    )

            attn_weights = F.softmax(attn_weights, dim=-1)

            if output_attentions:
                attn_weights_reshaped = attn_weights.view(
                    bsz,
                    module.num_heads,
                    tgt_len,
                    src_len,
                )
                attn_weights = attn_weights_reshaped.view(
                    bsz * module.num_heads,
                    tgt_len,
                    src_len,
                )
            else:
                attn_weights_reshaped = None

            attn_probs = F.dropout(
                attn_weights,
                p=module.dropout,
                training=module.training,
            )

            attn_output = torch.bmm(attn_probs, value_states)

            expected_output = (
                bsz * module.num_heads,
                tgt_len,
                module.head_dim,
            )

            if attn_output.size() != expected_output:
                raise ValueError(
                    f"unexpected output shape: {attn_output.size()}"
                )

            attn_output = attn_output.view(
                bsz,
                module.num_heads,
                tgt_len,
                module.head_dim,
            )

            attn_output = (
                attn_output.transpose(1, 2).reshape(
                    bsz,
                    tgt_len,
                    embed_dim,
                )
            )

            attn_output = module.out_proj(attn_output)

            return attn_output, attn_weights_reshaped

        return patched_forward


@torch.inference_mode()
def next_token_logits(model, input_ids, image_tensor):
    out = model(
        input_ids=input_ids,
        images=image_tensor,
        use_cache=False,
        return_dict=True,
    )
    return out.logits[:, -1, :]


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids,
    image_tensor,
    highlighter,
    max_new_tokens=32,
    eos_token_id=None,
):
    highlighter.disable()

    out = input_ids.clone()
    generated = []

    for _ in range(max_new_tokens):
        logits = next_token_logits(model, out, image_tensor)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)

        generated.append(int(next_id.item()))
        out = torch.cat([out, next_id], dim=1)

        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break

    return generated


@torch.inference_mode()
def loba_generate(
    model,
    input_ids,
    image_tensor,
    highlighter,
    alpha=0.3,
    max_new_tokens=32,
    eos_token_id=None,
):
    out = input_ids.clone()
    generated = []
    diagnostics = []

    for step in range(max_new_tokens):
        highlighter.disable()
        logits_base = next_token_logits(
            model,
            out,
            image_tensor,
        )

        highlighter.enable()
        logits_high = next_token_logits(
            model,
            out,
            image_tensor,
        )

        log_p_base = F.log_softmax(logits_base.float(), dim=-1)
        log_p_high = F.log_softmax(logits_high.float(), dim=-1)

        score = (1.0 + alpha) * log_p_high - alpha * log_p_base
        next_id = torch.argmax(score, dim=-1, keepdim=True)

        if step == 0:
            diagnostics.append(
                {
                    "logits_base": logits_base.detach(),
                    "logits_high": logits_high.detach(),
                    "score": score.detach(),
                }
            )

        generated.append(int(next_id.item()))
        out = torch.cat([out, next_id], dim=1)

        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break

    highlighter.disable()
    return generated, diagnostics
