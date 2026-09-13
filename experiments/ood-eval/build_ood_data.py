"""Download + sample a fixed-seed subset of an external Vietnamese VQA dataset's
OFFICIAL TEST split, for out-of-distribution eval of Vintern-1B-v3_5 (zero-shot)
vs our best bridge+LoRA checkpoint. Neither dataset is AutoViVQA/ViVQA/OpenViVQA
-- both are genuinely unseen by every model we evaluate here.

Datasets (schemas verified by direct download+parse, not guessed):
  vitextvqa -- ViTextVQA (arXiv:2404.10652, UIT, 2024). Scene-text/OCR VQA.
    HF mirror `nhonhoccode/ViTextVQA` (ungated; official `minhquan6203/ViTextVQA`
    is gated). test.json = {"images":[{id,filename}], "annotations":[{id,image_id,
    question,answers}]}. images.zip has internal prefix "images/<filename>".
  vivqax -- ViVQA-X (Springer ICISN 2025, VLAI-AIVN). General free-form VQA +
    NLE explanations, machine-translated from VQA-X. HF `VLAI-AIVN/ViVQA-X`.
    ViVQA-X_test.json = list of {question, image_id, image_name, explanation,
    answer, question_id, question_type, answer_type}. Images NOT bundled --
    fetched individually from the official COCO CDN by filename
    (COCO_{train,val}2014_<id>.jpg encodes which COCO split to hit).

Writes, under --out:
  manifest.json   -- dataset, seed, n requested/actual, source URLs (traceability)
  images/<id>.jpg -- only the sampled images
  internvl.jsonl  -- {"image","conversations":[{"value":"<image>\\n"+question}],
                      "all_answers"} for gen_vintern_base.py (Vintern zero-shot)
  ours.jsonl      -- {"image_name","question","answers","id"} for eval_ours_ood.py

    python build_ood_data.py --dataset vitextvqa --n 1000 --seed 42 --out /kaggle/working/data/vitextvqa
    python build_ood_data.py --dataset vivqax    --n 1000 --seed 42 --out /kaggle/working/data/vivqax
"""
import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import requests

HF = "https://huggingface.co/datasets"
VITEXTVQA_REPO = "nhonhoccode/ViTextVQA"   # ungated mirror; schema-verified to match official minhquan6203/ViTextVQA
VIVQAX_REPO = "VLAI-AIVN/ViVQA-X"
COCO_CDN = "http://images.cocodataset.org"


def _dl(url: str, dst: Path, desc: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {desc} already at {dst}")
        return
    print(f"[dl] {desc} <- {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dst)
    print(f"[dl] done: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def build_vitextvqa(n: int, seed: int, out: Path) -> dict:
    work = out / "_raw"
    test_json = work / "test.json"
    _dl(f"{HF}/{VITEXTVQA_REPO}/resolve/main/test.json", test_json, "ViTextVQA test.json")
    d = json.loads(test_json.read_text())
    id2name = {im["id"]: im["filename"] for im in d["images"]}
    anns = d["annotations"]  # [{id, image_id, question, answers}]

    rng = random.Random(seed)
    n = min(n, len(anns))
    picked = sorted(rng.sample(range(len(anns)), n), key=lambda i: anns[i]["id"])

    needed_files = sorted({id2name[anns[i]["image_id"]] for i in picked})
    zip_path = work / "images.zip"
    _dl(f"{HF}/{VITEXTVQA_REPO}/resolve/main/images.zip", zip_path, "ViTextVQA images.zip (full archive, extracting subset only)")
    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for fn in needed_files:
            member = f"images/{fn}"
            if member not in names:
                print(f"[warn] missing in zip: {member}")
                continue
            with zf.open(member) as src, open(imgs_dir / fn, "wb") as dst:
                dst.write(src.read())
    zip_path.unlink(missing_ok=True)  # reclaim disk once the subset is extracted

    rows = []
    for i in picked:
        a = anns[i]
        fn = id2name[a["image_id"]]
        if not (imgs_dir / fn).exists():
            continue
        rows.append({"id": a["id"], "image_name": fn, "question": a["question"], "answers": list(a["answers"])})
    return {"dataset": "vitextvqa", "source_repo": VITEXTVQA_REPO, "test_split_size": len(anns), "rows": rows}


def build_vivqax(n: int, seed: int, out: Path) -> dict:
    work = out / "_raw"
    test_json = work / "ViVQA-X_test.json"
    _dl(f"{HF}/{VIVQAX_REPO}/resolve/main/ViVQA-X_test.json", test_json, "ViVQA-X test json")
    rows_all = json.loads(test_json.read_text())  # list of dicts

    rng = random.Random(seed)
    n = min(n, len(rows_all))
    picked = sorted(rng.sample(range(len(rows_all)), n), key=lambda i: rows_all[i]["question_id"])

    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in picked:
        r = rows_all[i]
        fn = r["image_name"]  # e.g. COCO_val2014_000000262284.jpg
        split = "train2014" if "train2014" in fn else "val2014"
        dst = imgs_dir / fn
        if not (dst.exists() and dst.stat().st_size > 0):
            try:
                _dl(f"{COCO_CDN}/{split}/{fn}", dst, f"COCO image {fn}")
            except Exception as e:
                print(f"[warn] failed to fetch {fn}: {e}")
                continue
        rows.append({"id": r["question_id"], "image_name": fn, "question": r["question"], "answers": [r["answer"]]})
    return {"dataset": "vivqax", "source_repo": VIVQAX_REPO, "test_split_size": len(rows_all), "rows": rows}


BUILDERS = {"vitextvqa": build_vitextvqa, "vivqax": build_vivqax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BUILDERS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = BUILDERS[a.dataset](a.n, a.seed, out)
    rows = result.pop("rows")

    with open(out / "internvl.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "image": r["image_name"],
                "conversations": [{"from": "human", "value": "<image>\n" + r["question"]}],
                "all_answers": r["answers"],
            }, ensure_ascii=False) + "\n")

    with open(out / "ours.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {**result, "seed": a.seed, "n_requested": a.n, "n_actual": len(rows),
                "sampled_ids": [r["id"] for r in rows]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] {a.dataset}: {len(rows)} rows (seed={a.seed}) -> {out}")


if __name__ == "__main__":
    main()
