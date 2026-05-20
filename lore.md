# Lore Log

## 2026-05-17

1. Verified that `lore.md` did not exist in the repo.
Why: the user asked for a running record of every action and the reason for it, so the log had to be created before further work.

2. Confirmed the current working branch and worktree state with `git status`.
Why: I needed to know where new planning and code changes would land, and whether there were existing untracked research artifacts to preserve.

3. Read `research-outline.md` and `IDEAS.md`.
Why: these are the current source-of-truth planning documents for PRISM Part 1 and the side angles, so they define what both parallel tracks should optimize for.

4. Decided to work in parallel on two active directions: the performance-led core SRG paper track and the mechanism-led synthetic/diagnostic track.
Why: the user explicitly asked to pursue both directions in parallel rather than choose one first.

5. Set the immediate execution order to: map code paths, create parallel-track planning artifacts, then prepare code/research scaffolding.
Why: the repo already contains many relevant components, so the first practical step is to find the exact extension points before adding new artifacts or code.

6. Mapped the current implementation in `src/drf_experiment` and confirmed that SRG already exists as `spectral_response`, along with STFT routing, top-k routing, stochastic channel gates, and real-data shortlists.
Why: this changed the plan from “implement SRG from scratch” to “build paper-facing datasets, suites, and diagnostics around an existing SRG implementation.”

7. Checked the local Python runtime dependencies with `python3` imports for `torch`, `spikingjelly`, `pandas`, `seaborn`, and `h5py`.
Why: before scheduling experiments, I needed to know whether this machine could already run the synthetic and SHD benchmark paths.

8. Found that `torch`, `spikingjelly`, `pandas`, `seaborn`, and `h5py` are all missing from the default `python3` environment.
Why: this is a blocking resource gap for actual runs, so it had to be recorded early and surfaced in the runbook.

9. Added new synthetic datasets in `src/drf_experiment/datasets.py`: `multi_sine`, `band_switch`, and `spectral_noise`.
Why: the mechanism-led paper track needed paper-specific tasks that test mixture routing, time-local routing, and robustness to noise instead of relying only on the existing toy tasks.

10. Added new paper-focused suites in `src/drf_experiment/suites.py`: `paper_synthetic_mechanism`, `paper_synthetic_ablation`, and `paper_real_shortlist`.
Why: the performance-led and mechanism-led tracks needed explicit shortlists that can be run directly without rebuilding experiment selection each time.

11. Created `paper-runbook.md` with the two-track paper strategy, exact suite/dataset commands, success gates, and the missing resource list.
Why: the user asked to start building the paper, and the fastest useful artifact was a single runbook that ties together the research plan, code hooks, and execution order.

12. Added a dataset-only spectral diagnostic path in `src/drf_experiment/analysis.py` and `src/drf_experiment/cli.py`.
Why: the mechanism-led track needed a no-training tool to test whether datasets are spectrally class-separable and time-local enough to justify SRG or STFT-SRG before spending compute on full runs.

13. Updated `paper-runbook.md` to include the new dataset-diagnostic command.
Why: the diagnostic is only useful if it is immediately runnable from the paper workflow, so the runbook needed the exact command as soon as the code landed.

14. Ran `python3 -m py_compile` over the edited files in `src/drf_experiment`.
Why: the local Python environment cannot run the experiments yet, so syntax checking was the minimum safe verification step to confirm that the scaffolding changes are internally consistent.

15. Confirmed that the syntax check passed for the updated dataset, suite, analysis, and CLI files.
Why: before pausing or asking for environment resources, I needed to verify that the codebase remained parseable after the new paper-facing additions.

16. Inspected the repo for packaging and environment files and confirmed that there was no `pyproject.toml`, `requirements.txt`, or modern project environment definition.
Why: the user asked to convert the repo to a `uv`-based environment, so I first needed to verify whether a project configuration already existed.

17. Confirmed that `uv` is installed locally and that `uv` already has Python `3.11.14` available on this machine.
Why: before creating a new environment workflow, I needed to verify that the intended tool and a compatible Python version were actually present.

18. Added `pyproject.toml` and `.python-version` to define a `uv`-managed Python `3.11.14` environment for this repo.
Why: the repo needed a real project definition so `uv sync` could create a reproducible local `.venv` instead of depending on Faiz's old conda environment.

19. Updated `README.md` to replace the old conda-specific environment instructions with `uv` setup commands and to document the run artifact and runtime metric meanings.
Why: the user asked what the runs, time metrics, and related outputs actually mean, so the repo needed first-class documentation rather than leaving that buried in code.

20. Updated `paper-runbook.md` to use `uv run` commands and to include a short run artifact glossary.
Why: the paper workflow should match the new environment manager and use the same terminology for outputs and runtime metrics as the main repo documentation.

21. Ran `uv sync --python 3.11` and created a local `.venv` for the repo.
Why: the user asked for a `uv`-based environment, so the configuration needed to be turned into an actual runnable virtual environment rather than just a set of files.

22. Verified that the new `uv` environment installs and imports the core stack: `torch`, `spikingjelly`, `pandas`, `seaborn`, and `h5py`.
Why: after environment creation, I needed to confirm that the dependencies required for synthetic runs, SHD loading, and plotting were actually available inside `.venv`.

23. Ran a quick `--dataset-diagnostic` smoke test on `sine_frequency` and found a tensor-shape bug in the new chunk-vs-global KL comparison.
Why: end-to-end verification is more informative than just import checks, and this surfaced a real mismatch between chunked and full-sequence FFT lengths.

24. Fixed the dataset diagnostic to compute chunk spectra using the full sequence FFT length for comparisons.
Why: the mechanism-led diagnostic needs comparable spectral support between chunked and global views, otherwise the KL metric is invalid.

25. Re-ran syntax checks, import checks, and the `sine_frequency` dataset diagnostic successfully after the fix.
Why: I needed to verify that both the new environment and the new paper diagnostic path now work in practice, not just in theory.

26. Extracted runtime-estimate inputs from the repo: default model size, dataset defaults, epoch counts, and the paper-facing suite composition.
Why: the user asked how long useful results would take on an RTX 5000 Ada, so I needed estimate inputs grounded in this repo rather than answering with generic GPU guesses.

27. Verified that the common paper variants are small models by current standards, roughly `0.57M` to `0.68M` parameters depending on dataset and gating choice.
Why: parameter count is one of the main determinants of whether the first useful results are hours, days, or weeks on a single workstation GPU.
