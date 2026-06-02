from __future__ import annotations

import torch
from torch import Tensor, nn

from nanopa_twin.balance.state_space import Projector, build_projector
from nanopa_twin.barrel.dataset import Batch
from nanopa_twin.caliber.specs import ExperimentConfig, ModelConfig
from nanopa_twin.complications.heads import DensityHead, ProgressionHead, StagingHead
from nanopa_twin.complications.outputs import ModelOutputs
from nanopa_twin.escapement.synchroniser import CrossModalSynchroniser
from nanopa_twin.train_wheels.intervals import IntervalEmbedding
from nanopa_twin.train_wheels.photoacoustic import PhotoacousticEncoder
from nanopa_twin.train_wheels.tabular import TabularEmbedding


def _synchroniser_mode(model: ModelConfig) -> str:
    if model.temporal_core == "transformer":
        return "transformer"
    return "gated" if model.use_gated_fusion else "concat"


class NanoPATwin(nn.Module):
    deltas: Tensor

    def __init__(self, model: ModelConfig, horizons: tuple[int, ...]) -> None:
        super().__init__()
        self.model = model
        self.tabular = TabularEmbedding(model.n_biomarker, model.embed_dim)
        self.intervals = IntervalEmbedding(model.interval_dim)
        self.encoder: PhotoacousticEncoder | None = (
            PhotoacousticEncoder(
                model.n_wavelength,
                model.n_sensor,
                model.embed_dim,
                model.encoder_channels,
                model.encoder_blocks,
            )
            if model.use_photoacoustic
            else None
        )
        self.synchroniser = CrossModalSynchroniser(
            obs_dim=2 * model.embed_dim,
            state_dim=model.state_dim,
            interval_dim=model.interval_dim,
            mode=_synchroniser_mode(model),
            transformer_heads=model.transformer_heads,
            transformer_layers=model.transformer_layers,
        )
        self.projector: Projector = build_projector(
            model.projection, model.state_dim, model.hidden_dim, model.proj_dim
        )
        self.staging = StagingHead(model.proj_dim, model.n_class)
        self.density = DensityHead(model.proj_dim)
        self.progression = ProgressionHead(model.proj_dim, model.n_class)
        self.register_buffer(
            "deltas", torch.tensor([h / 12.0 for h in horizons], dtype=torch.float32)
        )

    def _photoacoustic(
        self, pa: Tensor, pa_mask: Tensor, concentrations: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length = pa_mask.shape
        empty = pa.new_zeros(0, 3)
        if self.encoder is None:
            e_pa = pa.new_zeros(batch, length, self.model.embed_dim)
            return e_pa, empty, empty
        signals = pa.reshape(batch * length, *pa.shape[2:])
        embedding = self.encoder(signals)
        flat_mask = pa_mask.reshape(-1, 1)
        e_pa = (embedding * flat_mask).reshape(batch, length, -1)
        index = torch.nonzero(pa_mask.reshape(-1) > 0, as_tuple=False).squeeze(-1)
        predicted = self.encoder.concentrations(embedding[index])
        target = concentrations.reshape(batch * length, 3)[index]
        return e_pa, predicted, target

    def forward(self, batch: Batch) -> ModelOutputs:
        e_tab = self.tabular(batch["biomarkers"])
        e_pa, predicted, target = self._photoacoustic(
            batch["pa"], batch["pa_mask"], batch["concentrations"]
        )
        observation = torch.cat([e_tab, e_pa], dim=-1)
        if self.model.use_time_embedding:
            dt_emb = self.intervals(batch["delta_t"])
        else:
            dt_emb = e_tab.new_zeros(e_tab.shape[0], e_tab.shape[1], self.model.interval_dim)
        state = self.synchroniser(observation, dt_emb, batch["seq_mask"])
        z_current, z_future = self.projector.project(state, self.deltas)
        return ModelOutputs(
            stage_logits=self.staging(z_current),
            bmd=self.density(z_current),
            progression_logits=self.progression(z_future),
            pred_concentrations=predicted,
            target_concentrations=target,
        )

    @torch.no_grad()
    def infer_subject(self, batch: Batch) -> dict[str, Tensor]:
        self.eval()
        outputs = self.forward(batch)
        return {
            "stage": torch.softmax(outputs.stage_logits, dim=-1),
            "bmd": outputs.bmd,
            "progression": torch.softmax(outputs.progression_logits, dim=-1),
        }


def build_model(config: ExperimentConfig) -> NanoPATwin:
    return NanoPATwin(config.model, config.data.horizons)
