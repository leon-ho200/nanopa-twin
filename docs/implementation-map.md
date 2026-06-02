# Implementation Map

Source files carry no comments and no docstrings by project convention; every paper provenance link lives here. Module names follow a mechanical-watch-movement layout: `caliber` (config), `mainspring` (data sources), `barrel` (batching), `train_wheels` (encoders), `escapement` (cross-modal synchronisation), `balance` (state-space core), `complications` (heads + losses), `regulator` (training), `timegrapher` (evaluation), `dial` (CLI). The assembled model lives in `nanopa_twin/movement.py`.

| paper item | equation / figure / table | file | symbol / object | notes |
|---|---|---|---|---|
| Spectral absorption decomposition | Eq. (1) | `nanopa_twin/mainspring/spectra.py` | `decompose_absorption`, `MOLAR_EXTINCTION` | μ_a(λ) over HbO₂, Hb, HA at 1064/1300/1550 nm |
| Spectral prior loss | Eq. (2) | `nanopa_twin/complications/objectives.py` | `spectral_prior_loss` | mean squared error on three predicted concentrations |
| PA waveform generator (k-Wave surrogate) | §IV.A, Fig. 1 | `nanopa_twin/mainspring/phantoms.py`, `nanopa_twin/mainspring/cohort.py` | `simulate_pa_signal`, `SyntheticCohort` | deterministic physics-prior stand-in; see `deviations.md` |
| Physics-informed PA encoder (PAE) | §III.C, Fig. 1 | `nanopa_twin/train_wheels/photoacoustic.py` | `PhotoacousticEncoder`, `ResidualBlock` | four residual blocks, 2× conv(3) + BN + GeLU, global average pooling → `d_e` |
| Spectral prior head | §III.C | `nanopa_twin/train_wheels/photoacoustic.py` | `PhotoacousticEncoder.concentrations` | softplus head emitting three nonnegative concentrations |
| Tabular MLP embedding | §III.D | `nanopa_twin/train_wheels/tabular.py` | `TabularEmbedding` | two-layer residual MLP `e^tab = MLP(b_i)` |
| Learnable time embedding | §III.D, Eq. (4) | `nanopa_twin/train_wheels/intervals.py` | `IntervalEmbedding` | sinusoidal encoding of Δt with a learned projection |
| Multimodal observation | Eq. (3) | `nanopa_twin/escapement/synchroniser.py` | `CrossModalSynchroniser.observe` | `[e^tab; e^pa]` if PA present else `[e^tab; 0]` |
| Modality-aware gate | Eq. (4) | `nanopa_twin/escapement/synchroniser.py` | `CrossModalSynchroniser.gate` | g = σ(W_g[o; s_{i-1}; Δt]) |
| Gated state update | Eq. (5)–(6) | `nanopa_twin/escapement/synchroniser.py` | `CrossModalSynchroniser.step` | candidate tanh update, convex blend |
| HiPPO-LegS state matrix | §III.E, ref [16] | `nanopa_twin/balance/hippo.py` | `hippo_legs_matrix` | strictly negative real eigenvalues |
| State-space projection | Eq. (7)–(8) | `nanopa_twin/balance/state_space.py` | `MetabolicStateSpace.project` | Ā = exp(AΔ), B̄ = A⁻¹(Ā − I)B via dense matrix exponential |
| Classification head | §III.E | `nanopa_twin/complications/heads.py` | `StagingHead` | softmax over three stages |
| BMD regression head | §III.E | `nanopa_twin/complications/heads.py` | `DensityHead` | affine readout `w_d·z + b_d` |
| Progression head | §III.E | `nanopa_twin/complications/heads.py` | `ProgressionHead` | softmax over future state at horizon Δ |
| Focal loss | §III.F, ref [17] | `nanopa_twin/complications/objectives.py` | `focal_loss` | class-imbalance term for staging |
| Total training objective | Eq. (9) | `nanopa_twin/complications/objectives.py` | `CompositeObjective` | λ = (1.0, 0.5, 1.0, 0.3) |
| Single-subject inference | Algorithm 1 | `nanopa_twin/movement.py` | `NanoPATwin.infer_subject` | per-visit accumulation then projection |
| Complexity O(N_s·d_e), O(d_e²) | §III.G | `nanopa_twin/escapement/synchroniser.py` | — | linear in sequence length |
| Three-class staging results | Table I | `nanopa_twin/timegrapher/classification.py` | `staging_report` | AUROC, accuracy, macro-F1, Cohen κ |
| BMD regression results | Table II | `nanopa_twin/timegrapher/regression.py` | `regression_report` | RMSE, MAE |
| Progression forecasting | Table III | `nanopa_twin/timegrapher/forecasting.py` | `concordance_index`, `forecast_report` | 12- / 24-month c-index and AUROC |
| Ablation study | Table IV | `nanopa_twin/caliber/presets/ablation_*.yaml` | `ModelConfig` toggles | component switches mirror Table IV rows |
| Per-class precision/recall/F1 | Table V | `nanopa_twin/timegrapher/classification.py` | `per_class_report` | |
| Hyperparameter sensitivity | Table VI | `nanopa_twin/caliber/presets/supplementary_sensitivity_*.yaml` | `d_e`, `d_h` sweeps | |
| Paired significance tests | Table VII | `nanopa_twin/timegrapher/significance.py` | `delong_test`, `paired_pvalue` | DeLong variance for AUROC |
| Computational cost | Table VIII | `nanopa_twin/timegrapher/profiling.py` | `count_parameters`, `time_inference` | params / throughput |
| PA-availability robustness | Table IX | `nanopa_twin/caliber/presets/supplementary_missing_modality.yaml` | `pa_availability` | gated vs. concat fusion under 0–100% PA |
| Calibration (ECE) | Fig. 4 | `nanopa_twin/timegrapher/calibration.py` | `expected_calibration_error` | reliability of staging confidence |
| Training / optimisation | §IV.C | `nanopa_twin/regulator/trainer.py`, `nanopa_twin/regulator/loop.py` | `Trainer`, `cross_validate` | AdamW, early stopping, two-phase PAE pre-train, atomic checkpoints |
| Bootstrap confidence intervals | §IV (results ±) | `nanopa_twin/timegrapher/resampling.py` | `bootstrap_ci` | |
