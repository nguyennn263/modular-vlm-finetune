# Paper figures

Regenerate: `python scripts/figures/make_figures.py` (needs matplotlib).
Numbers hard-coded from the canonical 2-epoch / 3-seed set — see
`plans/SESSION-STATE.md §1`. Update the script if those change.

- `fig_bridge_equalizing` — 5 bridges, corpus CIDEr-D, bridge-only vs +decoder
  attention LoRA. Shows the 79–92 spread collapsing into a ~101–103 band.
- `fig_tile_collapse` — multi-token bridge trained at 1 tile, evaluated at
  {1,3,6} tiles: token-F1 (left) and validation loss (right).
- `fig_method` — the frozen-backbone architecture; green = the ~1 % that trains.

PDF for LaTeX `\includegraphics`; PNG for quick preview.
