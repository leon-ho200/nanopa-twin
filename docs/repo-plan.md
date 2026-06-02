# Repository Plan

## Directory tree

```
nanopa_twin/
  caliber/          frozen-dataclass config schema + YAML preset loader (tyro-facing)
    presets/        main_*, ablation_*, supplementary_*, _smoke
  mainspring/       spectra (Eq. 1), phantoms (waveform surrogate), cohort generator
  barrel/           dataset, variable-length collate, loaders, device move
  train_wheels/     photoacoustic encoder, tabular embedding, interval embedding
  escapement/       cross-modal temporal synchroniser (Eq. 3-6; gated / concat / transformer)
  balance/          HiPPO-LegS matrix, metabolic state-space projector (Eq. 7-8)
  complications/    staging / density / progression heads, composite objective (Eq. 2, 9)
  regulator/        seeding, trainer, two-phase loop, cross-validation
  timegrapher/      classification, regression, forecasting, calibration, significance, resampling, profiling, report
  dial/             tyro command surface (fit / evaluate / forecast / encode / export)
  movement.py       assembled NanoPATwin model + single-subject inference
tests/              physics, shapes, state-space, synchroniser, objectives, metrics, gradients, overfit, determinism, pipeline, config, style-guard
docs/               project-context, implementation-map, deviations, repo-plan
scripts/            shell entry points for training, evaluation, data preparation
```

## Module responsibilities
- `caliber` is the only place configuration is defined; everything downstream receives typed dataclasses.
- `mainspring` is the single source of synthetic data; nothing else fabricates signals.
- `train_wheels` + `escapement` + `balance` + `complications` are pure modelling; they hold no I/O.
- `regulator` owns optimisation, checkpointing, and the PAE pre-train then fine-tune schedule.
- `timegrapher` is read-only evaluation and never mutates the model.
- `dial` is the command-line boundary; I/O and argument handling live here.

## Pinned dependencies
torch >= 2.1, numpy >= 1.24, scipy >= 1.10, scikit-learn >= 1.3, pyyaml >= 6.0, tyro >= 0.8. Development: ruff, black, isort, mypy, pytest.

## Test coverage
- shape: encoder / synchroniser / state-space / model output dimensions.
- physics-invariant: nonnegative absorption and concentrations, gate in [0, 1], HiPPO negative eigenvalues, interval embedding at zero gap.
- overfit-single-batch: loss drops on a fixed batch.
- metric-correctness: AUROC against scikit-learn, concordance against known orderings, regression closed forms, DeLong p-value range, bootstrap ordering.
- gradient-flow: every parameter receives a finite gradient.
- numerical-regression: identical losses under a fixed seed; reproducible cohort generation.
- integration: `run_experiment` end-to-end and checkpoint round-trip.
- config: presets load and build; unknown keys rejected; defaults equal the main setting.
- style-guard: no comments or docstrings in sources; no forbidden phrases or emoji in README / Makefile.
