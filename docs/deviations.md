# Deviations

Each entry records a point where this implementation departs from a literal reading of the manuscript, with the paper anchor and the reason.

## D1 — Photoacoustic simulation source
- Paper anchor: §IV.A, Fig. 1, refs [3], [9].
- Manuscript: PA-Bone-Sim signals come from k-Wave (a MATLAB toolbox) applied to calcaneus micro-CT phantoms.
- Here: `nanopa_twin/mainspring/{spectra,phantoms,cohort}.py` generate signals from a deterministic physics-prior model that maps chromophore concentrations through the Eq. (1) spectral decomposition into a damped, structure-modulated waveform. k-Wave and the credentialed NHANES / Harvard downloads are external to this repository.
- Reason: keeps the release runnable end-to-end offline while preserving the spectral-absorption-to-waveform relationship that the encoder and the spectral prior loss depend on. Reported numbers in the README come from the manuscript, not from this generator.

## D2 — State-matrix discretisation
- Paper anchor: §III.E, Eq. (7)-(8), refs [15], [16].
- Manuscript: the HiPPO-LegS matrix is diagonalised through the Normal-Plus-Low-Rank (NPLR) decomposition for an efficient matrix exponential.
- Here: `nanopa_twin/balance/state_space.py` computes `exp(AΔ)` with a dense matrix exponential.
- Reason: the MSSP projects a single state vector to one (or a few) future horizons rather than convolving over a long sequence, so the NPLR speedup is not needed; the dense exponential yields the identical discrete operator `Ā_Δ`.

## D3 — Reduced-head ablations
- Paper anchor: Table IV ("− MSSP (linear head)" and "S4 → MLP head").
- Here: both presets (`ablation_no_mssp.yaml`, `ablation_s4_to_mlp.yaml`) select the single `projection: mlp` head.
- Reason: the manuscript does not separately specify the internal structure of the two reduced heads; both replace the state-space projection with a feed-forward map.

## D4 — Repository layout and configuration stack
- Paper anchor: not applicable (engineering choice).
- Manuscript / kickoff template: a `src/<package>` tree with a Hydra configuration stack.
- Here: a flat watch-movement layout (`caliber`, `mainspring`, `barrel`, `train_wheels`, `escapement`, `balance`, `complications`, `regulator`, `timegrapher`, `dial`) with frozen-dataclass configs driven by `tyro`.
- Reason: release-distinctness requirement. Behaviour, equations, and reported quantities are unaffected.
