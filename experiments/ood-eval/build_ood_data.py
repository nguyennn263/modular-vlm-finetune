"""Download + sample a fixed-seed subset of an external Vietnamese VQA dataset's
OFFICIAL TEST split (or DEV, see openvivqa), for out-of-distribution eval of
Vintern-1B-v3_5 (zero-shot) vs our best bridge+LoRA checkpoint. None of these
4 datasets is AutoViVQA (this project's own training data) -- all genuinely
unseen by every model evaluated here.

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
  openvivqa -- OpenViVQA (Information Fusion 2023, arXiv:2305.04183, UIT).
    Vietnamese street-scene photos, ~44% of QA require reading embedded scene
    text (hybrid of plain VQA + OCR-in-photo). HF `uitnlp/OpenViVQA-dataset`.
    Uses the **dev** split, NOT test: vlsp2023_test_data.json's "answer" field
    is the literal placeholder string "your answer" for every row (verified by
    direct download -- answers are held out for the VLSP leaderboard). dev json
    = {"images": {id: filename}, "annotations": {ann_id: {image_id, question,
    answer}}} (id keys are strings in `images`, ints in `annotations`). Images
    bundled in dev-images.zip, internal prefix "dev-images/<filename>".
  vivqa -- UIT-ViVQA (PACLIC 2021, Tran et al., the ORIGINAL Vietnamese VQA
    dataset -- not to be confused with ViVQA-X or OpenViVQA). COCO-QA-style:
    single-word answers, object/number/color/location types. GitHub
    `kh4nh12/ViVQA`, test.csv columns `,question,answer,img_id,type` (col0 =
    row index). No formal license (paper: "available freely for research
    purposes", no LICENSE file). Images NOT bundled -- every img_id resolves
    against COCO **train2014 only** (verified by HEAD request against both
    train2014/val2014), unlike ViVQA-X which mixes train2014/val2014.

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
OPENVIVQA_REPO = "uitnlp/OpenViVQA-dataset"
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


def build_openvivqa(n: int, seed: int, out: Path) -> dict:
    # dev, NOT test -- vlsp2023_test_data.json's "answer" is the literal
    # placeholder "your answer" for every row (verified by direct download),
    # held out for the VLSP leaderboard. dev is the closest thing to a
    # held-out split with real ground truth.
    work = out / "_raw"
    dev_json = work / "vlsp2023_dev_data.json"
    _dl(f"{HF}/{OPENVIVQA_REPO}/resolve/main/vlsp2023_dev_data.json", dev_json, "OpenViVQA dev json")
    d = json.loads(dev_json.read_text())
    images = d["images"]  # {"<id>": filename}
    anns = d["annotations"]  # {"<ann_id>": {image_id (int), question, answer}}
    ann_items = list(anns.items())

    rng = random.Random(seed)
    n = min(n, len(ann_items))
    picked = sorted(rng.sample(range(len(ann_items)), n), key=lambda i: int(ann_items[i][0]))

    needed_files = sorted({images[str(ann_items[i][1]["image_id"])] for i in picked})
    zip_path = work / "dev-images.zip"
    _dl(f"{HF}/{OPENVIVQA_REPO}/resolve/main/dev-images.zip", zip_path,
        "OpenViVQA dev-images.zip (full archive, extracting subset only)")
    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for fn in needed_files:
            member = f"dev-images/{fn}"
            if member not in names:
                print(f"[warn] missing in zip: {member}")
                continue
            with zf.open(member) as src, open(imgs_dir / fn, "wb") as dst:
                dst.write(src.read())
    zip_path.unlink(missing_ok=True)

    rows = []
    for i in picked:
        ann_id, a = ann_items[i]
        fn = images[str(a["image_id"])]
        if not (imgs_dir / fn).exists():
            continue
        rows.append({"id": ann_id, "image_name": fn, "question": a["question"], "answers": [a["answer"]]})
    return {"dataset": "openvivqa", "source_repo": OPENVIVQA_REPO, "test_split_size": len(ann_items), "rows": rows,
            "note": "uses the DEV split (real answers); vlsp2023_test_data.json's answers are a placeholder"}


def build_vivqa(n: int, seed: int, out: Path) -> dict:
    # UIT-ViVQA (PACLIC 2021, Tran et al.) -- the original Vietnamese VQA
    # dataset (COCO-QA style: single-word answers, object/number/color/
    # location types). CSV via GitHub raw (no HF mirror needed). Every img_id
    # in test.csv resolves against COCO train2014 ONLY (verified by HEAD
    # request against both train2014/val2014) -- not a val2014/train2014 mix
    # like ViVQA-X. No formal license: paper says "available freely for
    # research purposes", no LICENSE file in the repo -- flag if it matters.
    import csv
    work = out / "_raw"
    test_csv = work / "test.csv"
    _dl("https://raw.githubusercontent.com/kh4nh12/ViVQA/main/test.csv", test_csv, "ViVQA test.csv")
    with open(test_csv, encoding="utf-8") as fh:
        rows_all = list(csv.DictReader(fh))

    rng = random.Random(seed)
    n = min(n, len(rows_all))
    picked = sorted(rng.sample(range(len(rows_all)), n), key=lambda i: int(rows_all[i][""]))

    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in picked:
        r = rows_all[i]
        fn = f"COCO_train2014_{int(r['img_id']):012d}.jpg"
        dst = imgs_dir / fn
        if not (dst.exists() and dst.stat().st_size > 0):
            try:
                _dl(f"{COCO_CDN}/train2014/{fn}", dst, f"COCO image {fn}")
            except Exception as e:
                print(f"[warn] failed to fetch {fn}: {e}")
                continue
        rows.append({"id": r[""], "image_name": fn, "question": r["question"], "answers": [r["answer"]]})
    return {"dataset": "vivqa", "source_repo": "kh4nh12/ViVQA", "test_split_size": len(rows_all), "rows": rows,
            "note": "no formal license in the source repo (paper: 'available freely for research purposes')"}


BUILDERS = {"vitextvqa": build_vitextvqa, "vivqax": build_vivqax, "openvivqa": build_openvivqa,
            "vivqa": build_vivqa}


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
