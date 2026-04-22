"""Central configuration for the song-vs-music ECoG analysis.

Every hard-coded path, hyperparameter, and resampling budget used by the
rest of the project is defined exactly once in this file and imported
from here. Do **not** redefine these constants elsewhere; change them
here and everything downstream will pick up the new value.

Design rationale
----------------
* A single source of truth makes the whole pipeline reproducible: a
  grader or classmate can inspect this file to see every analysis
  choice (seed, CV folds, permutation budget, time window).
* Keeping ``RANDOM_STATE`` global means every resampling/CV step in
  every module seeds the same way, so re-running reproduces all
  figures bit-for-bit.
"""
from __future__ import annotations

from pathlib import Path


# reproducibility seed shared by every resampling and cv step
RANDOM_STATE = 42

# time window in seconds relative to stimulus onset
TIME_MIN = 0.0
TIME_MAX = 2.0

# sliding window used by time resolved decoders and divergence curves
WINDOW_SEC = 0.15
STEP_SEC = 0.05

# outer cross validation folds and repeats
N_SPLITS = 5
N_REPEATS = 20

# resampling budgets for bootstrap, permutation, and random subset draws
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05
PERM_N = 1000
RANDOM_SUBSETS_N = 1000

# size for the matched random subset control, set from the song only
# electrode count (7), callers may override via arguments
SUBSET_SIZE = 7

# early window used for summary statistics on the time resolved curves
EARLY_WINDOW = (0.2, 0.6)

# label groupings mapping fine labels to coarse labels
SPEECH_FINE = {"EngSpeech", "ForSpeech"}
SONG_FINE = {"Song"}
MUSIC_FINE = {"Music"}
MAJOR_CLASSES = ("song", "speech", "music")

# config lives at <project_root>/logic/config.py so two parent hops
# land on the project root that holds data/ and results/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
NORMAN_HAIGNERE_DIR = DATA_ROOT / "norman_haignere_2022"
BELLIER_DIR = DATA_ROOT / "bellier_2023"

# primary dataset used for the main analyses
DATA_DIR = NORMAN_HAIGNERE_DIR / "individual_electrodes"

# norman haignere auxiliary files used by the acoustic partialled
# divergence analysis, ecog_component_responses,mat carries the
# canonical 165 stimulus ordering that acoustic_features,mat is
# aligned to (see reference_code/ecog_acoustic_corr,m)
NORMAN_ACOUSTIC_MAT = NORMAN_HAIGNERE_DIR / "acoustic_features.mat"
NORMAN_COMPONENT_RESP_MAT = NORMAN_HAIGNERE_DIR / "ecog_component_responses.mat"

RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"

# bellier 2023 extension (long timescale vocal coding)
BELLIER_HFA_DIR = BELLIER_DIR / "hfa"
BELLIER_STIM_DIR = BELLIER_DIR / "audio"
BELLIER_AUDIO_PATH = BELLIER_STIM_DIR / "thewall1.wav"      # bellier provided wav
BELLIER_VOCAL_CSV = BELLIER_STIM_DIR / "vocal_segments.csv" # user supplied annotation

BELLIER_FS = 100                # hfa sampling rate in hz
BELLIER_T = 19072               # total samples, roughly 190,72 seconds
BELLIER_CV_FOLDS = 5            # blocked time folds
BELLIER_WIN_SEC = 0.5           # decoder feature window
BELLIER_WIN_STEP_S = 0.1        # step between windows
BELLIER_BOOT_N = 500            # bootstrap replicates for confidence intervals

# temporal cnn hyperparameters, only runs if logreg beats chance
CNN_EPOCHS = 40
CNN_BATCH = 64
CNN_LR = 1e-3
CNN_MIN_BACC_OVER_CHANCE = 0.05

# event locked profile window around each onset
PROFILE_PRE_SEC = 0.5
PROFILE_POST_SEC = 1.5


def ensure_dirs() -> None:
    """Create the results subdirectories if they do not exist.

    Idempotent: safe to call at the start of any pipeline entry point.
    Creates ``results/``, ``results/figures/``, ``results/tables/``, and
    ``results/cache/`` under the project root.
    """
    for d in (RESULTS_DIR, FIG_DIR, TAB_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
