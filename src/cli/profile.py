"""Profile FLOPs + wall-clock across n_tiles  (final-plan P1).

    python -m src.cli.profile --n-tiles 1 2 4 6 --samples 64 --bridge tile_attention

Measures, for each tile count: InternViT FLOPs (best-effort, fvcore), single-sample
end-to-end latency, and batched throughput. Writes outputs/profile/pipeline_cost.json.

This tells us whether n_tiles is a real compute lever (final-plan section 5.2 / P1):
if the FLOPs / throughput dynamic range between 1 and max tiles is large, keep
`action = (n_tiles, bridge)`; if it is washed out, the bridge token-count axis
carries the frontier instead.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.config.loader import repo_root


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli.profile", description=__doc__)
    p.add_argument("--n-tiles", type=int, nargs="+", default=[1, 2, 4, 6], dest="n_tiles")
    p.add_argument("--samples", type=int, default=64, help="Images per measurement.")
    p.add_argument("--bridge", default="mini_qformer",
                   help="Bridge to profile — use a cross-attention patch bridge "
                        "(mini_qformer / qformer); tile_attention is O(L^2) and OOMs at high n_tiles.")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size", help="Batch for throughput.")
    p.add_argument("--images-from", default=None, dest="images_from",
                   help="JSONL/parquet with an image_path column (default: data/splits/val.jsonl "
                        "then data/labeled.parquet).")
    p.add_argument("--out", default="outputs/profile/pipeline_cost.json")
    return p


def _sample_image_paths(images_from: str | None, k: int) -> list[str]:
    root = repo_root()
    if images_from:
        path = Path(images_from) if Path(images_from).is_absolute() else root / images_from
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        return [r.get("image_path") or r["image_name"] for r in rows[:k]]

    # Preferred: the grouped split (env-aware image path resolution).
    if (root / "data/splits/val.jsonl").exists():
        from src.data.split import load_split
        return [s.image_path for s in load_split("val")[:k]]

    # Fallback: the env-aware raw loader (works on Kaggle without phase 0).
    from src.utils.data_loader_helper import AblationDataLoader
    return [s.image_path for s in AblationDataLoader(str(root)).load_raw_data(max_samples=k)[:k]]


def _flops_internvit(vision_model, pixel_values) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(vision_model, pixel_values)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total())
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    import torch
    from transformers import AutoModel

    from src.data.tiling import load_image_tiles
    from src.training import create_finetune_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "5CD-AI/Vintern-1B-v3_5"
    base = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                    low_cpu_mem_usage=False, trust_remote_code=True).eval().to(device)
    model = create_finetune_model(base, bridge_type=args.bridge, bridge_config={}).eval().to(device)
    dtype = next(model.vision_model.parameters()).dtype
    model.bridge = model.bridge.to(device=device, dtype=dtype)  # match the frozen stack (trainer does this)

    img_paths = _sample_image_paths(args.images_from, args.samples)
    text_emb = model.language_model.model.embed_tokens(
        torch.randint(0, 1000, (1, 32), device=device)
    )  # stand-in prompt for the prefill cost

    @torch.no_grad()
    def forward_once(pv):  # pv: (B, T, 3, S, S)
        b, t = pv.shape[:2]
        vis = model.vision_model(pv.flatten(0, 1).to(dtype))
        if hasattr(vis, "last_hidden_state"):
            hs = vis.last_hidden_state
        elif isinstance(vis, (tuple, list)):
            hs = vis[0]
        else:
            hs = vis
        hs = hs.reshape(b, t * hs.shape[1], hs.shape[2])
        bridge_out = model.bridge(hs)
        if bridge_out.dim() == 2:
            bridge_out = bridge_out.unsqueeze(1)
        combined = torch.cat([bridge_out, text_emb.expand(b, -1, -1).to(bridge_out.dtype)], dim=1)
        model.language_model(inputs_embeds=combined)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def _timed(pv) -> float:
        _sync()
        s = time.perf_counter()
        forward_once(pv)
        _sync()
        return time.perf_counter() - s

    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bridge": args.bridge, "samples": len(img_paths), "rows": [],
    }

    for n in args.n_tiles:
        row = {"n_tiles": n}
        try:
            tiles = [load_image_tiles(p, n_tiles=n).to(device) for p in img_paths]
            flops = _flops_internvit(model.vision_model, tiles[0].to(dtype))
            row["internvit_gflops"] = round(flops / 1e9, 2) if flops else None

            for _ in range(3):
                forward_once(tiles[0].unsqueeze(0))
            lat = [_timed(t.unsqueeze(0)) for t in tiles]
            row["latency_ms_median"] = round(1000 * statistics.median(lat), 1)
            row["latency_ms_p90"] = round(1000 * statistics.quantiles(lat, n=10)[-1], 1)

            bs = args.batch_size
            while bs >= 1:
                try:
                    batched = torch.stack(tiles[:bs])
                    for _ in range(2):
                        forward_once(batched)
                    row["throughput_img_per_s"] = round(bs / _timed(batched), 2)
                    row["throughput_batch"] = bs
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bs //= 2
        except Exception as exc:  # noqa: BLE001 - record and continue to the next n
            row["error"] = repr(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary["rows"].append(row)
        print(row)
        out.write_text(json.dumps(summary, indent=2))  # incremental — survive a later OOM

    ok = [r for r in summary["rows"] if "latency_ms_median" in r]
    if len(ok) >= 2:
        b, e = ok[0], ok[-1]
        summary["dynamic_range"] = {
            "n_tiles": [b["n_tiles"], e["n_tiles"]],
            "flops_x": (round(e["internvit_gflops"] / b["internvit_gflops"], 2)
                        if b.get("internvit_gflops") and e.get("internvit_gflops") else None),
            "latency_x": round(e["latency_ms_median"] / b["latency_ms_median"], 2),
            "throughput_x": (round(b["throughput_img_per_s"] / e["throughput_img_per_s"], 2)
                             if b.get("throughput_img_per_s") and e.get("throughput_img_per_s") else None),
        }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[profile] {out}\n" + json.dumps(summary.get("dynamic_range", "n<2 rows ok"), indent=2))


if __name__ == "__main__":
    main()
