# NEURO 120 Final Project — Song vs Speech vs Instrumental Music in Human ECoG

This project analyses two complementary human ECoG datasets to ask whether
song-selective information in auditory cortex is uniquely concentrated in
song-selective electrodes (Norman-Haignère et al. 2022) and whether the same
vocal-vs-instrumental contrast transfers to a naturalistic 190-second song
(Bellier et al. 2023).

## Project layout

```
neuro120-project/
├── data/                       # all raw data (see data/README.md)
│   ├── bellier_2023/
│   │   ├── hfa/                #   P*_HFA_data.mat + P*_MNI_*.mat, 29 patients
│   │   ├── audio/              #   thewall1.wav + stim MATs + vocal_segments.csv
│   │   └── reference_code/     #   original MATLAB pipeline (PF_HFAdecoding-main)
│   └── norman_haignere_2022/
│       ├── individual_electrodes/  # 33 response .mat files (7 song / 15 speech / 11 music)
│       ├── ...                  # components / fMRI / anat files
│       └── reference_code/      # original MATLAB analyses
├── papers/                      # Bellier.pdf, Norman-Heignere.pdf
├── results/
│   ├── figures/                 # fig1.png … fig10.png (generated)
│   ├── tables/                  # *.csv (generated)
│   └── cache/                   # supergrid + intermediate artifacts
├── logic/                       # every analysis module lives here
│   ├── __init__.py
│   ├── config.py                #   ALL paths & hyperparameters (single source of truth)
│   ├── data_utils.py            #   Norman-Haignère loader + dataset assembly
│   ├── bellier_data.py          #   Bellier loader + 29-patient supergrid
│   ├── bellier_decoder.py       #   blocked-CV vocal-vs-instrumental decoder (+ tiny CNN)
│   ├── temporal_profile.py      #   event-locked profiles & cross-dataset features
│   ├── subsets.py               #   electrode subsets + matched random-subset control
│   ├── decoding.py              #   grouped CV decoders (baseline + time-resolved)
│   ├── rdm.py                   #   representational dissimilarity matrices
│   ├── stats.py                 #   bootstrap, permutation, empirical p-values
│   ├── plots.py                 #   all figures
│   ├── nonlinear.py             #   supplementary nonlinear baselines
│   ├── analyses.py              #   high-level analysis drivers
│   └── pipeline.py              #   end-to-end orchestrator (`run_all()`)
├── project.ipynb                # final deliverable notebook (11 sections)
├── writeup.tex                  # final paper
├── requirements.txt
└── README.md
```

## Quick start

```bash
# (one-time) create a virtual env and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the whole analysis
jupyter nbconvert --to notebook --execute project.ipynb --output project.ipynb

# or run the full pipeline from the shell
python logic/pipeline.py

# or from Python (note the `logic/` path prepend)
python - <<'PY'
import sys
sys.path.insert(0, "logic")
import pipeline
pipeline.run_all()
PY
```

Results land in `results/figures/` (PNG) and `results/tables/` (CSV). The
supergrid cache (large) lives in `results/cache/`.

## Design choices

- Every CV split is a stimulus-grouped `StratifiedGroupKFold`; no stimulus
  ever crosses train/test folds.
- Every feature scaler is fit inside the training fold only.
- Every resampling procedure uses `config.RANDOM_STATE` (42) for full
  reproducibility.
- Negative results (matched random-subset control, nonlinear baselines) are
  reported alongside the positive ones.
- The Bellier extension adds an independent naturalistic-song test that
  converges on or diverges from the Norman-Haignère conclusions — no
  p-hacking, no claim is strengthened by hand-waving.

## Notebook structure

1. **Setup** — imports, config, directory bootstrap.
2. **Baseline 3-class decoder** (Figure 2).
3. **Time-resolved song-vs-music decoder** (Figure 5).
4. **Formalised divergence with bootstrap and permutation** (Figure 3).
5. **Matched random-subset control** (Figure 4 — headline result).
6. **Cross-temporal generalisation** (Figure 6).
7. **Leave-one-electrode-out contribution** (Figure 7).
8. **Supplementary nonlinear baselines** (negative result).
9. **Bellier extension 1 — vocal/instrumental decoder** (Figure 8), with
   stimulus-alignment + vocal-mask sanity check.
10. **Bellier extension 2 — cross-dataset temporal profile** (Figures 9–10).
11. **Final takeaways.**
