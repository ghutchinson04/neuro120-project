# NEURO 120 Final Project — Song vs Speech vs Instrumental Music in Human ECoG

A re-analysis of two human electrocorticography (ECoG) datasets, asking whether
the brain regions that respond most strongly to song also carry the most
information a decoder could use to tell song apart from instrumental music.

## The two papers, in plain words

If you are new to this literature, here is enough context to read the writeup.

**Norman-Haignere et al. (2022) — "A neural population selective for song in
human auditory cortex"** is the paper that kicked the question off. They played
165 natural sounds (songs, speech, instrumental music, and everyday sounds) to
patients with ECoG electrodes on the surface of their auditory cortex and
recorded the neural response to each sound. Using a hypothesis-free statistical
decomposition, they found that some electrodes respond much more strongly to
song than to any other category. Seven electrodes were flagged as
"song-selective" on that basis. Their main claim is that a distinct neural
population tracks song specifically, separate from speech and instrumental
music.

**Bellier et al. (2023) — "Music can be reconstructed from human auditory
cortex activity using nonlinear decoding models"** runs a different experiment.
Twenty-nine ECoG patients listened to a continuous 191-second excerpt of Pink
Floyd's *Another Brick in the Wall, Part 1*, and the authors showed that the
song's acoustics can be reconstructed from the neural recordings. They also
report that vocal content drives stronger responses in right STG than left STG.
For us, this is the "continuous, naturalistic music" check on whatever we find
in the short, repeated clips of Norman-Haignere.

**Our re-analysis** asks one question across both datasets: are the
song-selective electrodes actually more *informative* for a decoder than their
neighbours, or do they just fire more loudly? See `writeup.pdf` for the full
answer. The short version is: informative, yes; privileged, no.

## How to run it

**1. Install dependencies** (one-time, Python 3.10+ recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Put the data in `data/`** following `data/README.md`. Both datasets are
publicly released with their respective papers. The code expects them in the
layout `data/norman_haignere_2022/` and `data/bellier_2023/`.

**3. Run the full analysis.** Pick either option; both produce the same
figures and tables.

Option A — notebook (recommended if you want to see the intermediate
results):

```bash
jupyter lab project.ipynb
```

Then select the Python kernel for your virtual environment and run all cells.

Option B — from the command line (one command, headless):

```bash
python logic/pipeline.py
```

**4. Look at the output.** Everything lands in `results/`:

- `results/figures/` — every figure in `writeup.pdf` as a PDF.
- `results/tables/` — every table as a CSV.
- `results/cache/` — intermediate artifacts (including the Bellier supergrid,
  which is large and takes the longest to build on a cold run).

**5. Rebuild the writeup** if you edit `writeup.tex`:

```bash
latexmk -pdf writeup.tex
```