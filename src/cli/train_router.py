"""Train P(r|Q) — the cognitive-prior head (final-plan P4, contribution #2).

    python -m src.cli.train_router                      # full grouped split
    python -m src.cli.train_router --limit 4000 --epochs 2   # quick

Reads data/splits/{train,val}.jsonl, trains PhoBERT + an 8-way head with a
class-balanced loss, and writes checkpoints/router/{best.pt, metrics.json}
(macro-F1, per-class F1, confusion matrix on VAL).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.config.loader import repo_root


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.train_router", description=__doc__)
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=64, dest="max_len")
    p.add_argument("--encoder", default="vinai/phobert-base-v2")
    p.add_argument("--segment", choices=["pyvi", "none"], default="pyvi",
                   help="Vietnamese word segmentation for PhoBERT (pyvi if installed).")
    p.add_argument("--output-dir", default="checkpoints/router", dest="output_dir")
    p.add_argument("--predict-splits", default="train,val", dest="predict_splits",
                   help="After training, dump P(r|Q) for these splits -> outputs/router/prq_<split>.parquet")
    p.add_argument("--from-checkpoint", default=None, dest="from_checkpoint",
                   help="Skip training, load this router checkpoint and only predict.")
    p.add_argument("--dry-run", action="store_true")
    return p


def _rows(split_dir: str, name: str, limit: int | None):
    from src.data.split import load_split  # env-aware; category in metadata

    samples = load_split(name, split_dir)
    if limit:
        samples = samples[:limit]
    return [(s.question, s.metadata["category"]) for s in samples]


def _segmenter(mode: str):
    if mode == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("[warn] pyvi not installed -> no word segmentation (PhoBERT sub-optimal)")
    return lambda x: x


def run(args: argparse.Namespace) -> dict:
    tr_rows = _rows(args.split_dir, "train", args.limit)
    va_rows = _rows(args.split_dir, "val", (args.limit // 5) if args.limit else None)
    print(f"[data] train={len(tr_rows)} val={len(va_rows)}")
    print("[train dist] " + str(Counter(c for _, c in tr_rows)))
    if args.dry_run:
        return {}

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer

    from src.modeling.router import PrQHead
    from src.reasoning_types import CAT2IDX, CATEGORIES

    seg = _segmenter(args.segment)
    tok = AutoTokenizer.from_pretrained(args.encoder)

    def encode(rows):
        texts = [seg(q) for q, _ in rows]
        enc = tok(texts, padding="max_length", truncation=True,
                  max_length=args.max_len, return_tensors="pt")
        y = torch.tensor([CAT2IDX[c] for _, c in rows])
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = repo_root() / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_checkpoint:
        ck = torch.load(args.from_checkpoint, map_location=device, weights_only=False)
        model = PrQHead(ck.get("encoder", args.encoder)).to(device)
        model.load_state_dict(ck["state_dict"])
        print(f"[router] loaded {args.from_checkpoint} (skip training)")
        _dump_predictions(args, model, tok, seg, device)
        return {}

    train_ds, val_ds = encode(tr_rows), encode(va_rows)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    counts = Counter(int(y) for *_, y in train_ds)
    weights = torch.tensor([len(train_ds) / (len(CATEGORIES) * counts.get(i, 1))
                            for i in range(len(CATEGORIES))], dtype=torch.float, device=device)

    model = PrQHead(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best_f1, best_report = -1.0, {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids, mask, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(ids.to(device), mask.to(device)), y.to(device))
            loss.backward()
            opt.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for ids, mask, y in val_dl:
                preds += model(ids.to(device), mask.to(device)).argmax(-1).cpu().tolist()
                gts += y.tolist()
        report = _f1_report(np.array(gts), np.array(preds))
        print(f"[epoch {epoch}] val macro-F1 = {report['macro_f1']:.4f}")
        if report["macro_f1"] > best_f1:
            best_f1, best_report = report["macro_f1"], report
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "categories": CATEGORIES}, out_dir / "best.pt")

    (out_dir / "metrics.json").write_text(json.dumps(best_report, indent=2))
    print(f"[router] best val macro-F1 = {best_f1:.4f} -> {out_dir}")
    print(json.dumps(best_report["per_class_f1"], indent=2, ensure_ascii=False))

    # reload best + dump P(r|Q) predictions for the policy
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    _dump_predictions(args, model, tok, seg, device)
    return best_report


def _dump_predictions(args, model, tok, seg, device) -> None:
    import pandas as pd
    import torch

    from src.data.split import load_split
    from src.reasoning_types import CATEGORIES

    model.eval()
    out = repo_root() / "outputs" / "router"
    out.mkdir(parents=True, exist_ok=True)
    for split in [s.strip() for s in args.predict_splits.split(",") if s.strip()]:
        samples = load_split(split, args.split_dir)
        ids = [f"{(s.metadata or {}).get('image_id')}::{s.question}" for s in samples]
        texts = [seg(s.question) for s in samples]
        probs = []
        for i in range(0, len(texts), 256):
            enc = tok(texts[i:i + 256], padding="max_length", truncation=True,
                      max_length=args.max_len, return_tensors="pt")
            with torch.no_grad():
                p = torch.softmax(model(enc["input_ids"].to(device),
                                        enc["attention_mask"].to(device)), -1)
            probs.append(p.cpu())
        P = torch.cat(probs).numpy()
        df = pd.DataFrame({"sample_id": ids})
        for j, c in enumerate(CATEGORIES):
            df[f"p_{c}"] = P[:, j]
        df.to_parquet(out / f"prq_{split}.parquet", index=False)
        print(f"[router] P(r|Q) for {split}: {len(df)} rows -> {out}/prq_{split}.parquet")


def _f1_report(gts, preds) -> dict:
    import numpy as np

    from src.reasoning_types import CATEGORIES

    n = len(CATEGORIES)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1
    per_class = {}
    f1s = []
    for i, c in enumerate(CATEGORIES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = round(f1, 4)
        f1s.append(f1)
    return {
        "macro_f1": round(float(sum(f1s) / n), 4),
        "accuracy": round(float((gts == preds).mean()), 4),
        "per_class_f1": per_class,
        "confusion_matrix": cm.tolist(),
        "support": {c: int((gts == i).sum()) for i, c in enumerate(CATEGORIES)},
    }


def main(argv: list[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
