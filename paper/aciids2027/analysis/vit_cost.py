"""Vision-encoder compute of InternViT-300M per image view (Sec. 4.4, 'Computational cost').

Analytic multiply-accumulate count (MACs), the unit fvcore reports as "flops": per layer and
token, QKV + output projection (4 d^2) and MLP (2 d d_ff); attention scores and weighted sum
(2 n^2 d per layer); plus the patch embedding. For a 448x448 view this gives 361.8 GMACs,
matching the 362 "GFLOPs" measured with fvcore in plans/final-plan.md (P1 profiling,
P100). Run: python3 paper/aciids2027/analysis/vit_cost.py
"""
D, FF, L, P = 1024, 4096, 24, 14


def gmacs(side: int) -> float:
    patches = (side // P) ** 2
    n = patches + 1                                   # + [CLS]
    linear = L * n * (4 * D * D + 2 * D * FF)
    attn = L * 2 * n * n * D
    embed = patches * (P * P * 3 * D)
    return (linear + attn + embed) / 1e9


if __name__ == "__main__":
    v336, v448 = gmacs(336), gmacs(448)
    print(f"336x336 view: {v336:.1f} GMACs ({(336 // P) ** 2 + 1} tokens)")
    print(f"448x448 view: {v448:.1f} GMACs ({(448 // P) ** 2 + 1} tokens)")
    print(f"6 tiles + thumbnail (7 views): {7 * v448:.1f} GMACs = {7 * v448 / v336:.1f}x one 336 view")
