import json
import re
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_answer(value):
    if value is None:
        return "unknown"

    text = str(value).strip().lower()

    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    match = re.search(r"\b(yes|no)\b", text)
    return match.group(1) if match else "unknown"


def get_gt(row):
    for key in ("gt", "answer", "label", "ground_truth"):
        if key in row:
            value = norm_answer(row[key])
            if value != "unknown":
                return value
    return "unknown"


def get_prediction(row, method=None):
    method_keys = {
        "Vanilla": ["vanilla_prediction", "prediction", "pred", "parsed_prediction", "parsed"],
        "PH": ["ph_prediction", "method_prediction", "prediction", "pred", "parsed_prediction", "parsed"],
        "VCD": ["vcd_prediction", "method_prediction", "prediction", "pred", "parsed_prediction", "parsed"],
        "CRG": ["crg_prediction", "method_prediction", "prediction", "pred", "parsed_prediction", "parsed"],
        "LoBA": ["loba_prediction", "method_prediction", "prediction", "pred", "parsed_prediction", "parsed"],
    }

    keys = method_keys.get(
        method,
        ["prediction", "pred", "parsed_prediction", "parsed"],
    )

    for key in keys:
        if key in row:
            value = norm_answer(row[key])
            if value != "unknown":
                return value

    raise KeyError(f"prediction key not found; method={method}, keys={list(row.keys())}")


def transition(gt, vanilla, method):
    if gt == "no":
        if vanilla == "yes":
            return "hallucination_rescue" if method == "no" else "failed_hallucination_rescue"
        if vanilla == "no":
            return "induced_hallucination" if method == "yes" else "stable_correct_no"

    if gt == "yes":
        if vanilla == "no":
            return "miss_rescue" if method == "yes" else "failed_miss_rescue"
        if vanilla == "yes":
            return "over_suppression" if method == "no" else "stable_correct_yes"

    return "unknown"


def parse_disease(question):
    text = str(question).strip()
    patterns = [
        r"^Is there (.+?) in (?:the )?.+? of this patient\??$",
        r"^Does .+? suffer from (.+?)\??$",
        r"^Does the patient have (.+?) at .+?\??$",
        r"^Does the patient have (.+?) in .+?\??$",
        r"^Is (.+?) present in .+?\??$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Unknown"
