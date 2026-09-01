"""The 8 canonical reasoning-type categories — single source of truth, zero deps."""

CATEGORIES = [
    "relational", "recognition", "spatial", "causal",
    "action", "counting", "context", "yesno",
]
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for c, i in CAT2IDX.items()}
