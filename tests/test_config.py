from __future__ import annotations

import pytest

from nanopa_twin.caliber.presets import available_presets, from_mapping, load_preset
from nanopa_twin.caliber.specs import ModelConfig
from nanopa_twin.movement import build_model


def test_main_nhanes_matches_paper_defaults() -> None:
    config = load_preset("main_nhanes")
    assert config.model.embed_dim == 128
    assert config.model.hidden_dim == 256
    assert config.optim.lr == pytest.approx(3e-4)
    assert config.optim.batch_size == 32
    assert config.optim.max_epochs == 200
    assert config.seeds == (42, 123, 456, 789, 2024)
    assert config.loss.lambda_spec == pytest.approx(0.3)


def test_model_config_defaults_equal_main() -> None:
    defaults = ModelConfig()
    assert defaults.embed_dim == 128
    assert defaults.hidden_dim == 256
    assert defaults.n_class == 3


@pytest.mark.parametrize("name", available_presets())
def test_every_preset_builds_a_model(name: str) -> None:
    config = load_preset(name)
    model = build_model(config)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(KeyError):
        from_mapping({"nonsense": 1})


def test_ablation_toggles_change_architecture() -> None:
    no_pae = load_preset("ablation_no_pae")
    assert no_pae.model.use_photoacoustic is False
    assert build_model(no_pae).encoder is None
    transformer = load_preset("ablation_transformer_core")
    assert transformer.model.temporal_core == "transformer"
