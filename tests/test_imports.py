"""Import smoke tests.

Two tiers: modules that must import with no heavy ML deps (tooling, CLI parsing,
data prep) and modules that legitimately need torch/transformers.
"""
import importlib
import sys

import pytest

LIGHT_MODULES = [
    "src",
    "src.cli.train",
    "src.cli.evaluate",
    "src.cli.download",
    "src.cli.profile",
    "src.cli.train_router",
    "src.cli.train_policy",
    "src.cli.oracle",
    "src.cli.build_fiq",
    "src.reasoning_types",
    "src.analysis.stats",
    "src.analysis.expB",
    "src.analysis.oracle",
    "src.config.loader",
    "src.utils.logging",
    "src.utils.paths",
    "src.utils.data_loader_helper",
    "src.data.environment",
    "src.data.data_actions",
    "src.data.labeled_table",
    "src.data.split",
    "src.schema.data_schema",
]

HEAVY_MODULES = [
    "src.modeling.bridge_modules",
    "src.modeling.router",
    "src.modeling.policy",
    "src.training",
    "src.training.setup",
    "src.training.trainer",
    "src.data.loader",
    "src.data.collator",
    "src.data.tiling",
    "metrics.vqa_metrics",
]


@pytest.mark.parametrize("name", LIGHT_MODULES)
def test_light_import(name):
    importlib.import_module(name)


def test_import_src_is_torch_free():
    for mod in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        del sys.modules[mod]
    importlib.reload(importlib.import_module("src"))
    assert "torch" not in sys.modules, "importing `src` must not pull torch"


@pytest.mark.parametrize("name", HEAVY_MODULES)
def test_heavy_import(name):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    importlib.import_module(name)
