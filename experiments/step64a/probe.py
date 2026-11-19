import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PLAN = Path("results/step60c-plan-full-256-receiptfix/plan.json")
SAMPLE_ID = "slake-en-test-0318"
OUT = Path("results/step64a-official")

parser = argparse.ArgumentParser()
parser.add_argument("--preflight", action="store_true")
args = parser.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
job = os.environ.get("SLURM_JOB_ID", "local")
report_file = OUT / f"step64a-{job}.json"

report = {
    "step": "STEP64A",
    "status": "INCOMPLETE",
    "scope": "Real-image structural audit; not full CAST evaluation",
    "job": job,
    "sample_id": SAMPLE_ID,
    "stages": [],
    "observations": {},
    "checks": {},
    "errors": [],
}

def stage(n, name):
    print(f"STEP64A_STAGE={n}/9 {name}", flush=True)
    report["stages"].append({"stage": n, "name": name})

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def info(mask):
    if mask is None:
        return {"is_none": True}
    x = mask.detach()
    return {
        "is_none": False,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "zeros": int((x == 0).sum().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }

# Same sequence assembly logic as official evaluate_slake.py
def insert_separator(x, separators):
    if len(x) > len(separators):
        separators.append([])
    return [
        element
        for pair in zip(x, separators)
        for element in pair
    ]

def bbox_sequence(mask_crop, n=4, block=12):
    if mask_crop.ndim != 3:
        raise ValueError(
            f"Expected 3D bbox crop, got {mask_crop.shape}"
        )

    b = mask_crop.shape[0]

    local = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(n * block, n * block),
        mode="nearest",
    ).squeeze(1).long()

    local = torch.cat([
        local,
        torch.zeros(
            (b, n * block, 1),
            dtype=torch.long,
            device=local.device,
        )
    ], dim=2).reshape(b, -1)

    global_mask = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(block, block),
        mode="nearest",
    ).squeeze(1).long()

    global_mask = torch.cat([
        global_mask,
        torch.zeros(
            (b, block, 1),
            dtype=torch.long,
            device=global_mask.device,
        )
    ], dim=2).reshape(b, -1)

    return torch.cat([
        local,
        torch.zeros(
            (b, 1),
            dtype=torch.long,
            device=local.device,
        ),
        global_mask,
    ], dim=1)

def prepare_sample(plan):
    samples = plan["samples"]
    item = samples[SAMPLE_ID]

    image_path = Path(item["original_image_file"])
    roi_path = Path(item["strict_roi_file"])

    if not image_path.is_file() or not roi_path.is_file():
        raise FileNotFoundError(
            f"Missing image or ROI: {image_path}, {roi_path}"
        )

    image = Image.open(image_path).convert("RGB")
    roi = np.load(roi_path, allow_pickle=False)

    if roi.ndim != 2:
        raise ValueError(f"Invalid ROI shape: {roi.shape}")

    roi = (roi > 0).astype(np.uint8)

    if roi.shape != (image.height, image.width):
        raise ValueError(
            "ROI/image resolution mismatch: "
            f"{roi.shape} vs {image.size}"
        )

    if roi.sum() == 0:
        raise ValueError("Selected ROI is empty")

    question = item["question"]

    report["observations"]["sample"] = {
        "image_path": str(image_path),
        "roi_path": str(roi_path),
        "image_sha256": sha(image_path),
        "roi_sha256": sha(roi_path),
        "image_size": list(image.size),
        "roi_pixels": int(roi.sum()),
        "question": question,
        "roi_source": "Existing STEP60C strict ROI",
        "official_gt_answer_selection_reexecuted": False,
    }
    return image, roi, question

def load_compatible_processor(model_path):
    # Reuse the processor compatibility loader already used
    # by the earlier CAST evaluation pipeline.
    repro = Path("CAST-step60c-eval/repro").resolve()
    if not repro.is_dir():
        raise FileNotFoundError(repro)

    sys.path.insert(0, str(repro.parent))

    from repro.phase2_common import load_processor

    processor = load_processor(model_path)
    processor.tokenizer.padding_side = "left"

    report["observations"]["processor"] = {
        "class": type(processor).__name__,
        "module": type(processor).__module__,
        "loader": "CAST-step60c-eval/repro/step59_paper.py",
    }
    return processor

def prepare_official_inputs(processor, image, roi, question):
    prompt = processor.tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": f"<|image_1|>\n{question}",
        }],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Official processor kwargs are preserved.
    # Sentinel bbox / empty text highlights avoid
    # introducing dataset annotation leakage.
    inputs = processor(
        text=prompt,
        images=[image],
        images_mask=[roi],
        bboxes=[(0.0, 0.0, 0.0, 0.0)],
        mask_values=[1],
        attribute=["match"],
        qs_highlighted_parts=[],
        return_tensors="pt",
    )

    bbox = bbox_sequence(inputs["bbox_attention_mask"])
    highlights = inputs["highlight_attention_mask"]

    tokens = insert_separator(
        inputs["prompt_chunks"],
        inputs["image_ids_pad"],
    )
    masks = insert_separator(
        highlights,
        bbox.tolist(),
    )

    flat_ids = [
        token for group in tokens for token in group
    ]
    flat_mask = [
        value for group in masks for value in group
    ]

    if len(flat_ids) != len(flat_mask):
        raise RuntimeError(
            "Official sequence/mask alignment mismatch: "
            f"{len(flat_ids)} vs {len(flat_mask)}"
        )

    ids = torch.tensor(
        flat_ids, dtype=torch.long
    ).unsqueeze(0)

    high = torch.tensor(
        flat_mask, dtype=torch.long
    ).unsqueeze(0)

    valid = (ids > -1000000).long()

    # Official two-branch construction, unchanged.
    weighted = high.clone().float()
    weighted[high == 1] = 0.01
    weighted[high == 0] = 3.0

    combined = torch.cat([valid, weighted], dim=0)

    pixels = inputs["pixel_values"]
    sizes = inputs["image_sizes"]

    if pixels.ndim != 5 or sizes.ndim != 2:
        raise RuntimeError(
            f"Unexpected pixel/image shapes: "
            f"{pixels.shape}, {sizes.shape}"
        )

    if (combined == 0).any():
        raise RuntimeError(
            "This audit requires a positive-only mask"
        )

    report["observations"]["multimodal_input"] = {
        "sequence_length": int(ids.shape[1]),
        "negative_image_tokens": int((ids < 0).sum()),
        "highlighted_tokens": int((high == 1).sum()),
        "alpha": 0.01,
        "beta": 3.0,
        "mask": info(combined),
        "pixel_values_shape": list(pixels.shape),
        "image_sizes": sizes.tolist(),
        "bbox": [0.0, 0.0, 0.0, 0.0],
        "text_highlights": [],
    }

    if (ids < 0).sum() == 0 or (high == 1).sum() == 0:
        raise RuntimeError(
            "No image tokens or ROI highlights were generated"
        )

    return ids, combined, pixels, sizes

try:
    stage(1, "load_plan_and_real_slake_image")

    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)

    plan = json.loads(PLAN.read_text())
    report["plan_sha256"] = sha(PLAN)

    image, roi, question = prepare_sample(plan)

    stage(2, "load_compatible_official_processor")
    processor = load_compatible_processor(plan["model_path"])

    stage(3, "construct_official_multimodal_input")
    ids, weighted, pixels, sizes = prepare_official_inputs(
        processor, image, roi, question
    )

    if args.preflight:
        report["status"] = "PREFLIGHT_PASS"
        print("STEP64A_PREFLIGHT=PASS", flush=True)

    else:
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected one visible GPU"
            )

        stage(4, "load_bf16_flashattention2_model")

        model = AutoModelForCausalLM.from_pretrained(
            plan["model_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
            local_files_only=True,
        )

        if plan.get("lora_path"):
            model.load_adapter(plan["lora_path"])

        model = model.to("cuda:0").eval()

        fa2 = [
            m for m in model.modules()
            if type(m).__name__ == "Phi3FlashAttention2"
        ]

        core = [
            m for m in model.modules()
            if type(m).__name__ == "Phi3VModel"
        ]

        actual_backend = (
            core[0]._attn_implementation
            if len(core) == 1 else None
        )

        if len(fa2) != 32 or actual_backend != "flash_attention_2":
            raise RuntimeError(
                "Expected 32 FlashAttention2 layers"
            )

        report["observations"]["runtime"] = {
            "gpu": torch.cuda.get_device_name(0),
            "model_dtype": str(model.dtype),
            "backend": actual_backend,
            "fa2_layers": len(fa2),
        }

        ids = ids.to("cuda:0")
        weighted = weighted.to("cuda:0")
        pixels = pixels.to("cuda:0")
        sizes = sizes.to("cuda:0")

        batch_ids = ids.repeat(2, 1)
        batch_pixels = pixels.repeat(2, 1, 1, 1, 1)
        batch_sizes = sizes.repeat(2, 1)

        stage(5, "compare_generation_position_ids")

        prepared = model.prepare_inputs_for_generation(
            batch_ids,
            attention_mask=weighted,
            pixel_values=batch_pixels,
            image_sizes=batch_sizes,
            use_cache=True,
        )

        positions = prepared["position_ids"]

        baseline_positions = (
            torch.arange(
                batch_ids.shape[1],
                device="cuda:0",
            )
            .unsqueeze(0)
            .repeat(2, 1)
        )

        position_mismatches = [
            int((positions[i] != baseline_positions[i]).sum())
            for i in range(2)
        ]

        report["observations"]["positions"] = {
            "branches_equal": bool(
                torch.equal(positions[0], positions[1])
            ),
            "mismatch_count_vs_binary_baseline":
                position_mismatches,
            "first_positions": positions[:, :12].tolist(),
            "last_positions": positions[:, -12:].tolist(),
        }

        del prepared

        stage(6, "test_original_generate_guard")

        try:
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=batch_ids,
                    pixel_values=batch_pixels,
                    image_sizes=batch_sizes,
                    attention_mask=weighted,
                    max_new_tokens=1,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            report["observations"]["generate"] = {
                "status": "SUCCEEDED",
                "shape": list(generated.shape),
            }
            del generated

        except Exception as e:
            report["observations"]["generate"] = {
                "status": "ERROR",
                "type": type(e).__name__,
                "message": str(e)[:1200],
            }

        print(
            "STEP64A_GENERATE="
            + report["observations"]["generate"]["status"],
            flush=True,
        )

        torch.cuda.empty_cache()

        stage(7, "trace_all_fa2_layers")

        seen = {
            "calls": 0,
            "first_layer_masks": [],
        }

        def layer_hook(module, positional, keyword):
            seen["calls"] += 1
            if module is fa2[0]:
                seen["first_layer_masks"].append(
                    info(keyword.get("attention_mask"))
                )

        handles = [
            module.register_forward_pre_hook(
                layer_hook, with_kwargs=True
            )
            for module in fa2
        ]

        # Fix position IDs to isolate attention-mask effects.
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=batch_ids,
                    pixel_values=batch_pixels,
                    image_sizes=batch_sizes,
                    attention_mask=weighted,
                    position_ids=baseline_positions,
                    use_cache=False,
                    return_dict=True,
                )

            logits = output.logits[:, -1, :].float()
            delta = (logits[0] - logits[1]).abs()

            report["observations"]["direct_forward"] = {
                "max_abs_logit_delta": float(delta.max()),
                "mean_abs_logit_delta": float(delta.mean()),
                "argmax": logits.argmax(-1).tolist(),
                "fa2_layer_calls": seen["calls"],
                "first_layer_masks":
                    seen["first_layer_masks"],
            }

            del output, logits

        finally:
            for handle in handles:
                handle.remove()

        stage(8, "classify_results")

        p = report["observations"]["positions"]
        g = report["observations"]["generate"]
        f = report["observations"]["direct_forward"]

        checks = {
            "multimodal_input_contains_roi": True,
            "all_fa2_layers_executed":
                f["fa2_layer_calls"] == 32,
            "weighted_mask_removed_before_fa2":
                bool(f["first_layer_masks"])
                and all(
                    x["is_none"]
                    for x in f["first_layer_masks"]
                ),
            "same_positions_equal_logits":
                f["max_abs_logit_delta"] <= 1e-4,
            "nonbinary_mask_changes_positions":
                not p["branches_equal"],
            "original_generate_padding_guard_error":
                g["status"] == "ERROR"
                and "padding_side" in g.get("message", ""),
        }

        report["checks"] = checks
        report["status"] = (
            "CONFIRMED"
            if all(checks.values())
            else "PARTIAL_OR_INCONCLUSIVE"
        )

        report["observations"]["gpu_peak_allocated_gib"] = (
            round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            )
        )

except Exception as e:
    report["status"] = "ERROR"
    report["errors"].append({
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc()[-5000:],
    })
    print(f"STEP64A_ERROR={e!r}", flush=True)

finally:
    stage(9, "write_final_report")

    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"STEP64A_REPORT={report_file}", flush=True)
    print(f"STEP64A_STATUS={report['status']}", flush=True)

    for key, passed in report["checks"].items():
        print(f"STEP64A_CHECK {key}={passed}", flush=True)

if report["status"] == "ERROR":
    sys.exit(1)
