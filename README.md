# NanoPA-Twin

A decision log for the NanoPA-Twin code release: a NIR-II photoacoustic digital-twin model that
fuses simulated photoacoustic waveforms with longitudinal tabular biomarkers to stage osteoporosis,
regress bone-mineral density, and forecast 12- and 24-month progression (IEEE Journal of Biomedical
and Health Informatics).

This README is written as a series of Architecture Decision Records (ADRs). Each record states the
context, the decision the authoring team took, and the consequences. Operational material
(installation, commands, data, compute, ethics, citation) follows the records.

| ADR | Title | Status |
|-----|-------|--------|
| 0001 | What this release contains | Accepted |
| 0002 | Three data sources and an offline generator | Accepted |
| 0003 | A three-stage twin: encoder, synchroniser, state-space head | Accepted |
| 0004 | Training schedule and hyperparameters | Accepted |
| 0005 | Evaluation endpoints and expected values | Accepted |
| 0006 | Compute budget | Accepted |
| 0007 | Watch-movement layout and tyro configuration | Accepted |

---

## ADR-0001 — What this release contains

**Context.** The manuscript proposes three coupled components: a physics-informed photoacoustic
encoder (PAE), a cross-modal temporal synchroniser (CMTS) with a modality-aware gate, and a metabolic
state-space prediction head (MSSP) built on a HiPPO-LegS state matrix.

**Decision.** Ship one package, `nanopa_twin`, that implements all three components, the composite
training objective of Eq. (9), the evaluation suite behind Tables I-IX, and a command surface to
train, evaluate, forecast, pre-train the encoder, and export weights.

**Consequences.** A single `pip install -e .` exposes the `nanopa-twin` command and the importable
model `nanopa_twin.NanoPATwin`. The model runs on CPU for development and on a GPU for full-scale
settings.

---

## ADR-0002 — Three data sources and an offline generator

**Context.** Results are reported on the Harvard BMD dataset, the NHANES 2017-2020 cohort, and a
k-Wave photoacoustic simulation (PA-Bone-Sim). k-Wave is a MATLAB toolbox, and the two clinical
cohorts require external download.

**Decision.** Keep dataset provenance explicit and provide an offline generator (`mainspring`) that
produces signals from the Eq. (1) spectral decomposition so the pipeline runs end-to-end without the
external toolchain. See `docs/deviations.md` (D1).

**Consequences.** The headline numbers in ADR-0005 are the manuscript values on the real datasets;
the offline generator validates the pipeline, it does not recreate those exact numbers.

| Dataset | Version | License | Preprocessing | On-disk |
|---|---|---|---|---|
| Harvard BMD | Dataverse `doi:10.7910/DVN/UDZIJS` | confirm at Dataverse record (Dataverse default CC0) | DXA femoral-neck T-score to WHO three-class label | ~10 MB tabular |
| NHANES 2017-2020 | DEMO_J / BMX_J / DXXFEM_J / lab | U.S. public domain | join cycles, age >= 50, keep full DXA + lab | ~50 MB tabular |
| PA-Bone-Sim | k-Wave from calcaneus micro-CT | derived from public micro-CT (per source) | Otsu trabecular mask, NIR-II at 1064/1300/1550 nm | 2048 x 128 x 3 per scan |

The PA-to-NHANES link is a synthetic match by T-score decile; the manuscript notes (Section IV-A and
the Discussion) that the resulting `+1.8%` AUROC contribution from photoacoustic data may carry
T-score leakage rather than independent imaging signal. The release surfaces this rather than hiding
it.

---

## ADR-0003 — A three-stage twin: encoder, synchroniser, state-space head

**Context.** The twin must turn variable-length, irregularly sampled, sometimes-missing photoacoustic
and tabular streams into a single evolving state, then project that state forward in continuous time.

**Decision.**
- `train_wheels/photoacoustic.py` implements the PAE: four residual blocks of two `3x1` convolutions
  with batch normalisation and GeLU, global average pooling to a 128-dimensional embedding, and a
  softplus head predicting the three chromophore concentrations (Eq. 1-2).
- `escapement/synchroniser.py` implements CMTS: the modality-aware gate of Eq. (4) and the convex
  state update of Eq. (5)-(6), with a learnable interval embedding for the elapsed time. Concat-fusion
  and a transformer core are provided for the ablations.
- `balance/state_space.py` implements MSSP: a HiPPO-LegS state matrix and the discretisation
  `Ā = exp(AΔ)`, `B̄ = A⁻¹(Ā − I)B` of Eq. (7)-(8), feeding the staging, density, and progression heads.

**Consequences.** Each component can be switched off through `caliber` flags, which is how the
Table IV ablations are expressed. The state-matrix discretisation uses a dense matrix exponential
(`docs/deviations.md`, D2) because the head projects a single state vector rather than convolving a
sequence.

---

## ADR-0004 — Training schedule and hyperparameters

**Context.** The reported setting pre-trains the encoder on PA-Bone-Sim, then fine-tunes end-to-end,
with class imbalance handled by a focal term.

**Decision.** `regulator` runs AdamW at learning rate `3e-4`, weight decay `1e-4`, batch size 32, up
to 200 epochs with early stopping (patience 20) on validation loss, five-fold stratified
cross-validation, and seeds `(42, 123, 456, 789, 2024)`. The encoder is pre-trained for 50 epochs on
the spectral objective before fine-tuning. Loss weights are `λ = (1.0, 0.5, 1.0, 0.3)`. Checkpoints
are written atomically and store the seed.

**Consequences.** `configs` live as YAML presets under `nanopa_twin/caliber/presets/`. The defaults of
`ModelConfig` equal the NHANES main setting; `_smoke.yaml` shrinks every dimension for tests only and
must not be used for reporting.

---

## ADR-0005 — Evaluation endpoints and expected values

**Context.** Reviewers need one command per reported endpoint and the value to expect.

**Decision.** Expose evaluation through the `dial` command. Expected values below are the manuscript
results on the real datasets (mean over five seeds).

| Endpoint | Command | Expected (manuscript) |
|---|---|---|
| NHANES staging | `nanopa-twin evaluate --preset main_nhanes --checkpoint <ckpt>` | AUROC 0.921 ± 0.014, accuracy 0.847, macro-F1 0.801, κ 0.743 |
| Harvard staging | `nanopa-twin evaluate --preset main_harvard --checkpoint <ckpt>` | AUROC 0.893 ± 0.016, accuracy 0.808, macro-F1 0.763, κ 0.685 |
| BMD regression | `nanopa-twin evaluate --preset main_nhanes --checkpoint <ckpt>` | RMSE 0.038 ± 0.003, MAE 0.029 ± 0.002 g/cm² |
| Progression 12 mo | `nanopa-twin forecast --preset main_nhanes --checkpoint <ckpt>` | c-index 0.841 ± 0.018, AUROC 0.876 ± 0.016 |
| Progression 24 mo | `nanopa-twin forecast --preset main_nhanes --checkpoint <ckpt>` | c-index 0.813 ± 0.021, AUROC 0.847 ± 0.019 |
| PA availability sweep | `nanopa-twin evaluate --preset supplementary_missing_modality --checkpoint <ckpt>` | AUROC 0.903 (0%) rising to 0.921 (100%) |

Ablation presets (`ablation_*`) reproduce the Table IV deltas: removing MSSP costs the most
(−3.5% AUROC), then the encoder (−1.8%), with the spectral prior contributing a small `+0.7%`.

**Consequences.** On the offline generator these commands run to completion and report the same
metric structure; the absolute values track the synthetic signal, not the clinical cohorts.

---

## ADR-0006 — Compute budget

**Context.** Honest cost reporting is required.

**Decision.** State the manuscript budget without softening it.

| Item | Value |
|---|---|
| Hardware | 1 x NVIDIA A100 40 GB |
| Parameters | 2.8 M |
| Training | 28.4 s/epoch, up to 200 epochs |
| Inference | 2.7 ms/subject |
| Peak memory | 3.5 GB |

**Consequences.** The `_smoke` setting runs in seconds on CPU; the main settings target the A100 above.

---

## ADR-0007 — Watch-movement layout and tyro configuration

**Context.** The release must be organised distinctly while staying readable.

**Decision.** Lay the package out as a watch movement: `caliber` (the movement specification, i.e.
config), `mainspring` (the energy store, i.e. data sources), `barrel` (batching), `train_wheels`
(the gear train of encoders), `escapement` (temporal synchronisation), `balance` (the state-space
oscillator), `complications` (the added output functions), `regulator` (training rate control),
`timegrapher` (accuracy measurement, i.e. evaluation), and `dial` (the face, i.e. the CLI). Configure
through frozen dataclasses driven by `tyro`.

**Consequences.** The mapping is recorded in `docs/repo-plan.md`; behaviour matches the equations
regardless of the naming (`docs/deviations.md`, D4).

---

## Operational notes

### Install

```bash
pip install -e .                      # from a clone
conda env create -f environment.yml   # conda
docker build -t nanopa-twin .         # container
```

### Run

```bash
nanopa-twin fit --preset _smoke --out runs           # CPU smoke run
nanopa-twin fit --preset main_nhanes --device cuda   # main setting
nanopa-twin evaluate --preset main_nhanes --checkpoint runs/main_nhanes.pt
nanopa-twin forecast --preset main_nhanes --checkpoint runs/main_nhanes.pt
```

`make help` lists the supporting targets (`install`, `dev`, `lint`, `type`, `test`, `smoke`,
`docker`, `clean`).

### Checkpoints

None are distributed. Train with the commands above, or request weights from the authors.

### Code availability (verbatim from the manuscript)

> Full details on the training script, data preprocessing pipeline, and configuration file(s) are
> provided at https://github.com/[ANONYMIZED]/NanoPA-Twin.

### Ethics

The study collected no primary data and involved no human or animal subjects. All datasets (Harvard
BMD, NHANES 2017-2020, and the k-Wave photoacoustic simulation) are publicly available. Downstream use
requires domain-specific validation.

### Citation

```bibtex
@article{zhao_nanopa_twin,
  title   = {NanoPA-Twin: NIR-II Photoacoustic Nanoprobe-Enabled Digital Twins for Real-Time Bone
             Metabolism Monitoring and Osteoporosis Progression Prediction},
  author  = {Zhao, Jiaxin and Xiang, Zihan and She, Jiang},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025}
}
```
