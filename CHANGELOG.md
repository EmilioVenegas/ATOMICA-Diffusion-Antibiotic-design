# Changelog

Notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md` — the repository vendors two MIT
  codebases and previously carried no license of its own or attribution for them.
- `MODIFICATIONS.md` — itemises all 853 changed lines across 11 files in the
  vendored DiffSBDD fork, so the boundary between upstream code and this
  project's contribution is inspectable.
- `scripts/diff_upstream.py` — regenerates that modification map against upstream.
- `scripts/compare_conditions.py` — derives the ablation table and figure from the
  per-condition metrics, so published numbers are reproducible rather than
  transcribed.
- `docs/method.md` and `docs/results.md`.
- `tests/` — regression tests asserting the published A/B numbers still follow
  from the committed metrics.
- `.github/workflows/ci.yml` — lint plus verification that the quoted ablation
  numbers stay reproducible and `results/ablation_summary.md` stays in sync.
- `CITATION.cff`.

### Changed
- **`.gitignore` no longer excludes the project's own evidence.** Blanket
  `results/`, `docking/`, `*.json` and `*.png` rules meant the A/B ablation
  metrics — the record backing every number in the README — had never been
  committed. Narrow negations now version the metrics, summaries and figures
  while continuing to exclude the ~46 MB of raw SDF per arm.
- README rewritten to lead with the measured result and its limitations rather
  than the project pitch.
- Loose analysis scripts moved from the repository root into `scripts/`, with
  `sys.path` setup anchored to the repository root instead of the working
  directory so they run from anywhere.
- `pyproject.toml`: corrected author metadata, added license/repository fields
  and dev tooling.
- `CLAUDE.md`: corrected several claims that did not match the code — LoRA
  described as implemented in `egnn_new.py` (it is implemented nowhere), a
  `timestep_adaptive` switch that no code reads, an `adapter_scale` warm-start
  value that does not exist, and a unit-test path that does not exist.

### Known gaps
- No matched binding-affinity comparison between arms A and B; the current
  result rests on target-independent metrics only. See `docs/results.md`.
- Ablation arms C and D were never run; D's LoRA parameters have no
  implementation behind them. See `MODIFICATIONS.md`.
- No unit test for `SE3EquivariantCrossAttention` equivariance or masking.
