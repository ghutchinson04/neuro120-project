# Datasets

Two ECoG datasets live here, one per source paper. Each subfolder is
self-contained: its `README` and `code/` describe the upstream release.
Both datasets are now used by the analysis pipeline: Norman-Haignere
drives the core paper, and Bellier drives the two extension analyses
(vocal-vs-instrumental decoder and cross-dataset temporal profiles).

## `norman_haignere_2022/` - primary dataset

Source: Norman-Haignere SV, Feather J, Boebinger D, Brunner P, Ritaccio
A, McDermott JH, Schalk G, Kanwisher N. "A neural population selective
for song in human auditory cortex." *Current Biology* (2022).

Corresponding paper PDF: `../papers/Norman-Haignere .pdf`.

Layout:

- `README.txt` - upstream release notes (figure-script mapping).
- `code/` - the original MATLAB scripts from the paper's public release
  (reference only; our Python pipeline does not use them).
- `individual_electrodes/` - the `.mat` files loaded by `data_utils.py`.
  Contains 7 song, 15 speech, and 11 music electrode files, each with
  HFA responses to the 165 natural-sound stimuli (49 of which are song /
  speech / music in the coarse labeling we use).
- `ecog_component_*.mat`, `acoustic_features.mat`,
  `fmri-ecog-correlations.mat` - additional population- and
  component-level data referenced by the paper's figures but not loaded
  by our pipeline.
- `ecog_component_weights_electrodesurf/`,
  `ecog_component_weights_fMRIsurf/` - anatomical weight maps on the
  FreeSurfer fsaverage template.
- `fMRI-component-predictions/`, `hypothesis_driven_components/` -
  fMRI-derived companion data used by the paper's later figures.

Our analysis loads `individual_electrodes/` via `config.DATA_DIR`. The
rest is kept as reference so results can be cross-checked against the
original publication without re-downloading.

## `bellier_2023/` - extension dataset

Source: Bellier L et al., "Music can be reconstructed from human
auditory cortex activity using nonlinear decoding models." *PLoS
Biology* (2023).

Corresponding paper PDF: `../papers/Bellier.pdf`.

Layout:

- `README.md` - upstream README from the `PF_HFAdecoding` repo.
- `code/` - the `PF_HFAdecoding` MATLAB + Python release.
- `hfa/` - per-patient high-frequency activity `.mat` files
  (`P*_HFA_data.mat`) and MNI electrode coordinates
  (`P*_MNI_electrode_coordinates.mat`). Each HFA matrix has
  `T = 19,072` samples at 100 Hz (190.72 s of *Another Brick in
  the Wall, Part 1*). All 29 patients are pooled into a supergrid
  by `bellier_data.build_supergrid()`.
- `audio/thewall1.wav` - the original audio clip the patients heard
  during the ECoG recording. 190.72 s at 44.1 kHz, which matches the
  neural time axis exactly after downsampling to 100 Hz (no
  time-stretching is applied anywhere in this pipeline).
- `audio/thewall1_stim32.mat`, `audio/thewall1_stim128.mat` -
  Bellier's auditory stimulus features (32 and 128 mel bands,
  shape `(19072, n_bands)` at 100 Hz). Not currently loaded by
  the pipeline, staged for possible STRF extensions.
- `audio/vocal_segments.csv` - manual annotation of when the lead
  vocal is present. Columns `start_s,end_s` in plain seconds
  (relative to the start of `thewall1.wav`). Rows may overlap; the
  parser union-merges overlapping intervals when building the mask.
  `bellier_data.load_vocal_segments()` converts this file into a
  100-Hz binary vocal mask of length `T`.

The Bellier extension is wired behind the `include_bellier` flag of
`pipeline.run_all`. If `vocal_segments.csv` is missing the extension
steps are skipped cleanly and the Norman-Haignere pipeline still
produces all of Figures 1-7 and Tables 1-4.
