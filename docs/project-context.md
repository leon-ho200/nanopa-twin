# Project Context — NanoPA-Twin

| field | value | confidence |
|---|---|---|
| project_name | `nanopa_twin` | HIGH |
| domain | Multimodal temporal bone-metabolism modelling (photoacoustic waveforms + tabular biomarkers) for osteoporosis staging, BMD regression, and progression forecasting | HIGH |
| framework | PyTorch 2.x with a hand-written S4 / HiPPO-LegS state-space projection (`torch.nn`) | inferred from operator fingerprints, confirmed |
| venue | IEEE Journal of Biomedical and Health Informatics (JBHI) | HIGH |
| primary_datasets | 3 (Harvard BMD; NHANES 2017–2020; PA-Bone-Sim synthetic) | HIGH (names) |
| compute_target | 1× NVIDIA A100 40 GB; 2.8 M params; 28.4 s/epoch; 2.7 ms/subject inference; 3.5 GB | HIGH |
| hparams_reference | Methods §IV.C + Tables VI and I–IX | HIGH |
| supp_path | none on disk (text references Supplementary Tables S1, S3) | HIGH |

NEEDS_USER_DECISION: 0 (framework resolved to PyTorch 2.x).

## 1. project_name
`nanopa_twin` — title stopwords removed, two content words retained. HIGH.

## 2. supp_path
none. The main text references Supplementary Table S1 (independent effect of the spectral prior loss on PA-Bone-Sim) and Supplementary Table S3 (gated recurrence vs. self-attention). No supplementary file ships beside the manuscript; both are recorded as targets in `implementation-map.md` and approximated by `supplementary_*` presets. HIGH.

## 3. domain
Photoacoustic imaging combined with bone-metabolism modelling. Three concurrent objectives (Methods §III.A): three-class osteoporosis staging (Normal / Osteopenia / Osteoporosis), bone-mineral-density regression, and 12- / 24-month progression forecasting. HIGH.

## 4. framework
PyTorch 2.x with plain `torch.nn`. The Metabolic State-Space Prediction head (§III.E) uses a HiPPO-LegS initialised state matrix, NPLR-style parameterisation, and the discretisation `Ā = exp(AΔ)`, `B̄ = A⁻¹(Ā − I)B`. Operator fingerprints across Methods: 1-D convolutions, batch normalisation, GeLU, global average pooling, focal loss, AdamW. The manuscript does not name a framework explicitly; PyTorch is the resolved value.

## 5. venue
IEEE Journal of Biomedical and Health Informatics. Every page header carries the journal name; reference and figure styling match the IEEE template. HIGH.

## 6. primary_datasets
1. **Harvard BMD** — 1,128 records located, 1,043 retained after excluding incomplete subjects. DXA-derived femoral-neck BMD is the primary outcome. WHO T-score thresholds: Normal `T > −1.0`, Osteopenia `−2.5 < T ≤ −1.0`, Osteoporosis `T ≤ −2.5`. Source: Harvard Dataverse, persistent id `doi:10.7910/DVN/UDZIJS` (§IV.A, ref [7]). License: confirm against the Dataverse record at preparation time (Dataverse defaults to CC0 unless a waiver is attached).
2. **NHANES 2017–2020** — DEMO_J, BMX_J, DXXFEM_J, and the standard biochemistry lab files. 2,847 subjects with full DXA and lab data at age ≥ 50; longitudinal subset of 1,461 subjects with ≥ 2 observations (median T = 2, maximum T = 2). Source: CDC NHANES (§IV.A, ref [8]); U.S. public-domain.
3. **PA-Bone-Sim (synthetic)** — k-Wave acoustic simulation from human calcaneus micro-CT phantoms. 128 sensors, 2.25 MHz centre frequency, 50 µm resolution, NIR-II illumination at 1064 / 1300 / 1550 nm, 3,200 parameter combinations, signal tensor `2048 × 128 × 3` (§IV.A, Fig. 1; refs [3], [9]). k-Wave is a MATLAB toolbox and is not runnable here; this release ships a deterministic physics-prior waveform generator driven by the Eq. (1) spectral decomposition in place of k-Wave (logged in `deviations.md`).

## 7. compute_target
Single NVIDIA A100 40 GB. 2.8 M parameters, 28.4 s/epoch, 2.7 ms/subject inference, 3.5 GB memory (Table VIII, §IV.C). Up to 200 epochs with early stopping (patience 20) on validation loss; the PAE is pre-trained 50 epochs on PA-Bone-Sim before end-to-end fine-tuning.

## 8. hparams_reference
Methods §IV.C and Tables VI, I–IX. AdamW; learning rate `3e-4`; weight decay `1e-4`; batch size 32; ≤ 200 epochs; early-stopping patience 20 on validation loss; five-fold stratified cross-validation; seeds `(42, 123, 456, 789, 2024)`. Embedding dimension `d_e = 128`; state-space hidden dimension `d_h = 256` (NHANES) / `128` (Harvard BMD). Loss weights `λ = (1.0, 0.5, 1.0, 0.3)` for classification / regression / progression / spectral.

## 9. extra_signals
- Verbatim code-availability statement appears twice, each pointing to an anonymised repository URL.
- One algorithm box (Algorithm 1, single-subject inference).
- Supplementary experiments S1 and S3 are referenced; no supplementary file is present on disk.
- Honesty caveat (§IV.A, §V): the PA-to-NHANES link is synthetic, matched by T-score decile, so the reported `+1.8%` AUROC contribution from PA may include T-score leakage rather than independent imaging signal.
- Ethical statement: no primary data collection, no human or animal subjects, public datasets only.
- No released checkpoints.
- A Mamba-based projection variant is mentioned as a preliminary result (AUROC 0.918, higher variance).
