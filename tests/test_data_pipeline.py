"""Unit tests for the data-prep helpers (no dataset download needed)."""
import pandas as pd
import pytest

from src.data.labeled_table import CANONICAL, _norm_label, _norm_q, _parse_answers
from src.data.split import assign


def test_label_normalisation_maps_to_8_codes():
    codes = {CANONICAL[_norm_label(k)] for k in CANONICAL}
    assert codes == {
        "relational", "recognition", "spatial", "causal",
        "action", "counting", "context", "yesno",
    }


def test_norm_q_collapses_whitespace():
    assert _norm_q("  a   b\n c ") == "a b c"


@pytest.mark.parametrize("raw,expected", [
    (["x", "y"], ["x", "y"]),
    ("['a', 'b']", ["a", "b"]),
    ("plain", ["plain"]),
])
def test_parse_answers(raw, expected):
    assert _parse_answers(raw) == expected


def test_load_image_tiles_shape(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from src.data.tiling import load_image_tiles

    img = tmp_path / "x.jpg"
    Image.new("RGB", (900, 300), "red").save(img)

    assert load_image_tiles(str(img), n_tiles=1).shape == (1, 3, 448, 448)
    for n in (2, 4, 6):
        assert load_image_tiles(str(img), n_tiles=n).shape == (n, 3, 448, 448)


def test_collator_multitile_shape(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from src.data.collator import custom_collate_fn
    from src.schema.data_schema import OneSample

    for i in range(3):
        Image.new("RGB", (600, 400), "blue").save(tmp_path / f"{i}.jpg")
    batch = [OneSample(image_path=str(tmp_path / f"{i}.jpg"), question="q?", answers=["a"])
             for i in range(3)]

    assert custom_collate_fn(batch, n_tiles=1)["pixel_values"].shape == (3, 3, 336, 336)
    assert custom_collate_fn(batch, n_tiles=4)["pixel_values"].shape == (3, 4, 3, 448, 448)
    # batch-level augmentation: T is one of the choices, same for the whole batch
    pv = custom_collate_fn(batch, tile_choices=[2, 6])["pixel_values"]
    assert pv.shape[0] == 3 and pv.shape[1] in (2, 6)


def test_split_is_grouped_and_deterministic():
    rows = []
    for img in range(200):
        cat = ["relational", "recognition", "spatial", "causal"][img % 4]
        for _ in range(3):
            rows.append({"image_id": img, "category": cat})
    df = pd.DataFrame(rows)

    a = assign(df, (0.7, 0.15, 0.15), seed=42)
    b = assign(df, (0.7, 0.15, 0.15), seed=42)
    assert a["split"].tolist() == b["split"].tolist()          # deterministic
    assert (a.groupby("image_id")["split"].nunique() == 1).all()  # no leak
    frac = a.drop_duplicates("image_id")["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.05
