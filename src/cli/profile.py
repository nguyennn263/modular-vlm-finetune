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
    p.add_argument("--bridge", default="tile_attention", help="Bridge to profile (use a patch bridge).")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size", help="Batch for throughput.")
    p.add_argument("--images-from", default=None, dest="images_from",
                   help="JSONL/parquet with an image_path column (default: data/splits/val.jsonl "
                        "then data/labeled.parquet).")
    p.add_argument("--out", default="outputs/profile/pipeline_cost.json")
    return p


def _sample_image_paths(images_from: str | None, k: int) -> list[str]:
    root = repo_root()
    candidates = [images_from] if images_from else ["data/splits/val.jsonl", "data/labeled.parquet"]
    for rel in candidates:
        path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(path)["image_path"].head(k).tolist()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r["image_path"] for r in rows[:k]]

    # Fallback: the env-aware raw loader (works on Kaggle without phase 0).
    from src.utils.data_loader_helper import AblationDataLoader
    samples = AblationDataLoader(str(root)).load_raw_data(max_samples=k)
    return [s.image_path for s in samples[:k]]


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

    results = []
    for n in args.n_tiles:
        tiles = [load_image_tiles(p, n_tiles=n) for p in img_paths]  # list of (n,3,S,S)

        # FLOPs: InternViT on one image's tiles (the dominant, tile-scaling term).
        flops = _flops_internvit(model.vision_model, tiles[0].to(device=device, dtype=dtype))

        # Warmup + single-sample latency.
        for _ in range(3):
            forward_once(tiles[0].unsqueeze(0).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for t in tiles:
            s = time.perf_counter()
            forward_once(t.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - s)

        # Batched throughput.
        bs = args.batch_size
        batched = torch.stack(tiles[:bs]).to(device)
        for _ in range(2):
            forward_once(batched)
        if device.type == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        forward_once(batched)
        if device.type == "cuda":
            torch.cuda.synchronize()
        thr = bs / (time.perf_counter() - s)

        row = {
            "n_tiles": n,
            "internvit_gflops": round(flops / 1e9, 2) if flops else None,
            "latency_ms_median": round(1000 * statistics.median(lat), 1),
            "latency_ms_p90": round(1000 * statistics.quantiles(lat, n=10)[-1], 1),
            "throughput_img_per_s": round(thr, 2),
        }
        results.append(row)
        print(row)

    base_row = results[0]
    summary = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "bridge": args.bridge,
        "samples": len(img_paths),
        "rows": results,
        "dynamic_range": {
            "flops_x": (round(results[-1]["internvit_gflops"] / base_row["internvit_gflops"], 2)
                        if base_row["internvit_gflops"] else None),
            "latency_x": round(results[-1]["latency_ms_median"] / base_row["latency_ms_median"], 2),
            "throughput_x": round(base_row["throughput_img_per_s"] / results[-1]["throughput_img_per_s"], 2),
        },
    }
    out = repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[profile] {out}\n" + json.dumps(summary["dynamic_range"], indent=2))


if __name__ == "__main__":
    main()
