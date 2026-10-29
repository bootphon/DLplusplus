# DL++ (Dataloader++)

A feature processing and data loading framework for child-centered long-form audio recordings. Runs a SLURM pipeline that extracts speech activity, speaker types, signal quality, and environmental sound classification (ESC) — then packages everything into WebDataset shards with rich per-clip metadata for model training.

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Configuration](#3-configuration)
4. [Pipeline](#4-pipeline)
5. [Project Structure](#5-project-structure)
6. [Dataloader](#6-dataloader)
7. [Citation](#7-citation)
8. [Component Models](#8-component-models)
9. [Acknowledgements](#9-acknowledgements)

---

## 1. Installation

**Requirements:** Linux or macOS, Python ≥ 3.13, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/).

```bash
git clone https://github.com/bootphon/DLplusplus.git
cd DLplusplus

# Install Python dependencies:
uv sync

# Download model checkpoints (Brouhaha + VTC 2.0):
uv run download-models
```

Model weights are cached to `~/.cache/dlplusplus/` by default. Override with `MODEL_ROOT`:

```bash
MODEL_ROOT=/shared/models uv run download-models
```

---

## 2. Quick Start

### Generate a manifest from a directory

```bash
uv run make-manifest /path/to/audio/ --name my_dataset
```

Recursively scans for all common audio formats (`wav`, `flac`, `mp3`, `ogg`, `opus`, `m4a`, `aac`, `aiff`, `wma`) and writes `manifests/my_dataset.csv` with columns `path` (absolute), `uid` (filename stem), and `ext` (format).

### Single-step inference

```bash
# Voice activity detection (VAD)
uv run python -m audio_pipeline.pipeline.vad my_data \
    --manifest manifests/my_dataset.csv

# Speaker diarization (VTC)
uv run python -m audio_pipeline.pipeline.vtc my_data \
    --manifest manifests/my_dataset.csv
```

### Full pipeline on a SLURM cluster

```bash
# First run with a custom manifest:
bash slurm/pipeline.sh my_data \
    --manifest manifests/my_dataset.parquet \
    --path-col audio_path \
    --audio-root /store/audio/

# Subsequent runs (manifest already normalized):
bash slurm/pipeline.sh my_data
```

This submits five SLURM jobs — four feature extraction steps in parallel, then a packaging step that depends on all four. See the [Pipeline](#4-pipeline) section for full documentation.

---

## 3. Configuration

### Workspace layout

All pipeline outputs are anchored to a single workspace root. Set `DLPP_WORKSPACE` to control where everything lands:

```bash
export DLPP_WORKSPACE=/scratch/my_project
```

| Sub-directory | Default (relative to workspace) | Override env var |
|---------------|----------------------------------|-----------------|
| Manifests | `$DLPP_WORKSPACE/manifests/` | `DLPP_MANIFESTS_DIR` |
| Pipeline output | `$DLPP_WORKSPACE/output/` | `DLPP_OUTPUT_DIR` |
| Figures | `$DLPP_WORKSPACE/figures/` | `DLPP_FIGURES_DIR` |
| SLURM logs | `$DLPP_WORKSPACE/logs/` | `DLPP_LOGS_DIR` |

When `DLPP_WORKSPACE` is unset, all directories resolve relative to the current working directory (original behaviour).

Individual directories can be overridden independently:

```bash
DLPP_OUTPUT_DIR=/fast-scratch/output uv run python -m audio_pipeline.pipeline.vad my_data
```

### Model cache

Downloaded model weights are stored under `MODEL_ROOT`:

```
~/.cache/dlplusplus/       (default MODEL_ROOT)
├── brouhaha/
│   └── best.ckpt
└── vtc/
    ├── model/
    │   ├── best.ckpt
    │   └── config.toml
    └── thresholds/
        ├── f1.toml        # F1-optimal per-label thresholds
        └── hp.toml        # High-precision per-label thresholds
```

VTC 2.0 supports two threshold sets:
- **`f1.toml`** — maximises F1 per label (default)
- **`hp.toml`** — maximises precision (fewer false positives)

Pass `--thresholds_path` to `vtc.py` or `vtc_on_clips.py` to switch between them.

---

## 4. Pipeline

The pipeline orchestrator (`slurm/pipeline.sh`) runs a preflight check, then submits five SLURM jobs:

```
                ┌─── VAD  (CPU)  ───┐
                ├─── VTC  (GPU)  ───┤
Raw Audio ──►   ├─── SNR  (GPU)  ───┼──► Package (CPU)
                └─── ESC  (GPU)  ───┘
```

Steps 1–4 run **in parallel** as independent jobs. Step 5 (Package) depends on all four completing successfully.

| Step | Module | Resource | Description |
|------|--------|----------|-------------|
| **1. VAD** | `audio_pipeline.pipeline.vad` | CPU | TenVAD speech activity detection |
| **2. VTC** | `audio_pipeline.pipeline.vtc` | GPU | BabyHuBERT/segma speaker diarization (KCHI, OCH, MAL, FEM) |
| **3. SNR** | `audio_pipeline.pipeline.snr` | GPU | Brouhaha per-frame SNR & C50 extraction |
| **4. ESC** | `audio_pipeline.pipeline.esc` | GPU | PANNs CNN14 environmental sound classification |
| **5. Package** | `audio_pipeline.pipeline.package` | CPU | Clip tiling + WebDataset shards + dashboards |

**Resume support:** VAD and VTC save checkpoints. Interrupted jobs can be resubmitted and will skip already-completed files.

### Step 1 — VAD (Voice Activity Detection)

Runs [TenVAD](https://github.com/Tencent/TenVAD) with CPU multiprocessing (default: all cores).

**Output:**
- `output/{dataset}/vad_raw/segments.parquet` — per-frame VAD segments
- `output/{dataset}/vad_merged/segments.parquet` — merged overlapping segments
- `output/{dataset}/vad_meta/metadata.parquet` — per-file summary metadata

### Step 2 — VTC (Voice Type Classification)

Runs the BabyHuBERT model via [segma](https://github.com/arxaqapi/segma) on GPU (SLURM array, default 3 shards).

**Output:**
- `output/{dataset}/vtc_raw/` — raw VTC segments (per-shard parquets)
- `output/{dataset}/vtc_merged/` — merged/deduplicated segments across shards
- `output/{dataset}/vtc_meta/` — per-file summary metadata

Segment columns: `uid`, `onset`, `offset`, `duration`, `label` (FEM / MAL / KCHI / OCH).

### Step 3 — SNR (Signal-to-Noise Ratio & Clarity)

Runs [Brouhaha](https://github.com/marianne-m/brouhaha-vad) on GPU (SLURM array, default 2 shards). Produces **per-file time-series arrays** and **speech-masked summary statistics**.

**Output:**
- `output/{dataset}/snr/{uid}.npz` — per-file compressed arrays:
  - `snr` (float16, shape `n_frames`) — per-frame SNR in dB
  - `c50` (float16, shape `n_frames`) — per-frame C50 clarity in dB
  - `vad` (float16, shape `n_frames`) — per-frame Brouhaha VAD probability
  - `step_s` — frame step in seconds (~16 ms)
  - `vad_threshold` — threshold used (0.5)
- `output/{dataset}/snr_meta/shard_{id}.parquet` — per-file metadata:
  - `uid`, `snr_status`, `duration`, `n_raw_frames`, `n_speech_frames`, `speech_fraction`
  - `snr_mean`, `snr_std`, `snr_min`, `snr_max` — computed only on speech frames (VAD > 0.5)
  - `c50_mean`, `c50_std`, `c50_min`, `c50_max` — computed only on speech frames

Downstream steps (e.g. packaging) index into the per-frame arrays by `onset/offset` using `step_s` to compute exact segment-level statistics.

### Step 4 — ESC (Environmental Sound Classification)

Runs [PANNs CNN14](https://github.com/qiuqiangkong/panns_inference) on GPU (SLURM array, default 2 shards). Classifies audio into 13 coarse categories and 527 AudioSet classes.

**Output:**
- `output/{dataset}/esc/{uid}.npz` — per-file compressed arrays:
  - `categories` (float16, shape `n_bins × 13`) — coarse category probabilities
  - `category_names` — the 13 category labels
  - `audioset_probs` (float16, shape `n_bins × 527`) — full AudioSet probabilities
  - `audioset_names` — 527 AudioSet display labels
  - `pool_step_s`, `inference_step_s` — time resolutions
- `output/{dataset}/esc_meta/shard_{id}.parquet` — per-file metadata:
  - `uid`, `esc_status`, `duration`, `n_inference_windows`, `n_pooled_bins`
  - `dominant_category`, `dominant_prob`
  - `prob_{category}` — mean probability for each of 13 categories

**Categories:** alarm_signal, animal, crying, environment, human_activity, impact, laughter, machinery, music, nature, other, silence, singing, tv_radio, vehicle.

### Step 5 — Package (Clip Tiling + WebDataset Shards)

Tiles full audio files into clips of roughly equal length, cutting only at silence gaps (never mid-speech). Cut-point selection uses a **6-tier fallback chain**:

| Tier | Strategy | Severity |
|------|----------|----------|
| 1 | Long silence gap (≥10 s) in VAD∪VTC union | Clean |
| 2 | Any silence gap in VAD∪VTC union | Clean |
| 3 | Gap in VAD-only mask (VTC still active) | Info |
| 4 | Gap in VTC-only mask (VAD still active) | Info |
| 5 | VTC speaker-change boundary (inside active audio) | Warning |
| 6 | Hard cut — no gaps or boundaries | Warning |

Within each tier, the midpoint closest to the ideal evenly-distributed position is chosen. The pipeline output includes a **tier breakdown** showing how many cuts used each strategy.

**Output:**
- `output/{dataset}/shards/` — WebDataset `.tar` shards (WAV/FLAC + JSON metadata)
- `output/{dataset}/shards/manifest.csv` — per-clip metadata
- `output/{dataset}/shards/samples/` — random sample clips for manual validation
- `output/{dataset}/stats/` — Parquet DataFrames at multiple granularities (clip, segment, turn, conversation, file)
- `figures/{dataset}/dashboard/` — 6 PNG diagnostic dashboards

### Clip metadata

Each clip in a shard is stored as two files sharing the key `{uid}_{clip_idx:04d}`:

| File | Format | Contents |
|------|--------|----------|
| `{clip_id}.wav` / `.flac` | WAV / FLAC | Mono audio, 16 kHz |
| `{clip_id}.json` | JSON (UTF-8) | All scalar + structured metadata (see below) |

The `.json` metadata contains:

**Source** — `uid`, `clip_idx`, `clip_id`, `abs_onset`, `abs_offset`, `duration`, `source_path`, `audio_fmt`, `sample_rate`.

**VTC speech** — `vtc_speech_duration`, `vtc_speech_density`, `n_vtc_segments`, `mean_vtc_seg_duration`, `mean_vtc_gap`, `n_turns`, `n_labels`, `labels_present`, `has_adult`, `dominant_label`, `label_durations`, `vad_coverage_by_label`.

**Demographics** — `child_speech_duration`, `adult_speech_duration`, `child_fraction`.

**VAD speech** — `vad_speech_duration`, `vad_speech_density`, `n_vad_segments`.

**VAD–VTC agreement** — `vad_vtc_iou`: frame-level Intersection over Union between the two systems' masks.

**SNR & C50** — Per-VTC-segment SNR and C50 averages are computed by the `segment_snr` post-hoc step and stored in `output/{dataset}/segment_snr/` parquets. During packaging, these are aggregated into per-clip summary statistics in the manifest CSV: `snr_mean`, `snr_std`, `snr_min`, `snr_max`, `c50_mean`, `c50_std`, `c50_min`, `c50_max` (dB). The full per-frame time-series arrays remain available in `snr/{uid}.npz`.

**ESC environment** — `dominant_esc` (category name), `esc_profile` (dict of mean probability per category).

**Segment detail** — `vad_segments` and `vtc_segments`: lists of `{onset, offset, duration}` objects with timestamps relative to the clip start. `vtc_segments` additionally carry a `label` field (FEM / MAL / KCHI / OCH).

### Additional tools

| Module | Purpose |
|--------|---------|
| `audio_pipeline.plotting.compare` | VAD vs VTC comparison (IoU, precision, recall, diagnostics) |
| `audio_pipeline.pipeline.normalize` | Standardize external manifests into `manifests/{dataset}.csv` |
| `audio_pipeline.pipeline.preflight` | Estimate dataset size, GPU needs, and wall-clock time |
| `audio_pipeline.pipeline.segment_snr` | Post-hoc per-VTC-segment SNR/C50 averaging |

---

## 5. Project Structure

```
DLplusplus/
├── src/
│   ├── audio_pipeline/           # Main pipeline package
│   │   ├── paths.py              #   ProjectPaths (DLPP_WORKSPACE and per-dir overrides)
│   │   ├── utils.py              #   Shared utilities (manifest I/O, parquet helpers)
│   │   ├── compat.py             #   Compatibility shims (torchaudio patches)
│   │   ├── make_manifest.py      #   Console script: `uv run make-manifest`
│   │   ├── download_models.py    #   Console script: `uv run download-models`
│   │   ├── pipeline/             #   CLI entry points (one per pipeline step)
│   │   │   ├── vad.py            #     Step 1: TenVAD voice activity detection
│   │   │   ├── vtc.py            #     Step 2: BabyHuBERT speaker diarization
│   │   │   ├── snr.py            #     Step 3: Brouhaha SNR/C50 extraction
│   │   │   ├── esc.py            #     Step 4: PANNs CNN14 ESC
│   │   │   ├── package.py        #     Step 5: Audio clipping + WebDataset shards
│   │   │   ├── vtc_clip_alignment.py  # Post-hoc VTC clip alignment analysis
│   │   │   ├── segment_snr.py    #     Post-hoc per-segment SNR/C50 averaging
│   │   │   ├── normalize.py      #     Manifest normalization
│   │   │   ├── preflight.py      #     Pre-pipeline dataset scan
│   │   │   └── resources.py      #     SLURM resource estimation helpers
│   │   ├── packaging/            #   Clip building, shard writing, listener
│   │   │   ├── clips.py          #     Clip tiling algorithm (6-tier fallback)
│   │   │   ├── stats.py          #     Per-clip/file/conversation statistics
│   │   │   ├── writer.py         #     WebDataset tar shard writer
│   │   │   ├── loaders.py        #     Audio/metadata loaders for packaging
│   │   │   ├── packer.py         #     Shard packing orchestration
│   │   │   └── listener.py       #     Sample extraction for validation
│   │   ├── core/                 #   Reusable, tested modules
│   │   │   ├── intervals.py      #     Interval arithmetic (merge, IoU)
│   │   │   ├── conversations.py  #     Turn/conversation extraction
│   │   │   ├── vad_processing.py #     Per-file VAD (worker code)
│   │   │   ├── parallel.py       #     Process pool driver with progress queue
│   │   │   ├── checkpoint.py     #     Checkpoint save / resume
│   │   │   ├── metadata.py       #     VTC metadata constructors
│   │   │   ├── audio.py          #     Audio I/O helpers
│   │   │   └── brouhaha.py       #     Brouhaha SNR inference helpers
│   │   ├── analysis/             #   Exploratory analysis scripts
│   │   │   ├── vtc_on_clips.py   #     Run VTC on packaged WebDataset clips
│   │   │   └── ...               #     Other analysis tools
│   │   └── plotting/             #   Dashboard figure generation
│   │       ├── figures.py        #     Orchestrator (calls sub-modules)
│   │       ├── master.py         #     Master dashboard layout
│   │       ├── clip_alignment.py #     Clip alignment plots
│   │       ├── compare.py        #     VAD vs VTC comparison
│   │       └── utils.py          #     Plotting utilities
│   └── dataloader/               # Dataloader++ package (see Section 6)
│       ├── types.py              #   Shared type aliases and enums
│       ├── config.py             #   PipelineConfig + FilterConfig
│       ├── build.py              #   build_manifest() — Big Join + filters
│       ├── create.py             #   Dataset creation entry point
│       ├── paths.py              #   Path resolution (mirrors audio_pipeline/paths.py)
│       ├── processor/            #   Feature Processor ABCs (offline extraction)
│       ├── adapters/             #   Pipeline output adapters (VAD, VTC, SNR, ESC)
│       ├── loader/               #   Feature Loader ABCs (waveform + metadata I/O)
│       ├── manifest/             #   Manifest management (schema, joiner, store)
│       ├── transform/            #   Runtime data transforms (audio, label, waveform)
│       ├── batch/                #   Batching and collation (DataBatch, SpeechCollator)
│       ├── dataset/              #   PyTorch Dataset implementations
│       └── compat/               #   Upstream compatibility shims
├── slurm/
│   ├── pipeline.sh               # One-command pipeline orchestrator
│   ├── vad.slurm                 # SLURM: VAD (CPU, 48 workers)
│   ├── vtc.slurm                 # SLURM: VTC (GPU array, 3 shards)
│   ├── snr.slurm                 # SLURM: Brouhaha SNR (GPU array, 2 shards)
│   ├── esc.slurm                 # SLURM: PANNs ESC (GPU array, 2 shards)
│   ├── segment_snr.slurm         # SLURM: Per-segment SNR (GPU array)
│   ├── vtc_clips.slurm           # SLURM: VTC on packaged clips
│   ├── snr_diagnostic.slurm      # SLURM: SNR masking diagnostics
│   ├── package_test.sh           # Quick end-to-end packaging test
│   ├── repackage_test.sh         # Re-package + clip alignment test
│   └── test.slurm                # SLURM: pytest on compute node
├── tests/                        # pytest suite covering all core modules
│   ├── conftest.py               #   Audio fixtures + skip markers
│   ├── fixtures/                 #   Short WAV files (committed)
│   ├── test_intervals.py
│   ├── test_checkpoint.py
│   ├── test_metadata.py
│   ├── test_parallel.py
│   ├── test_clips.py             #   Clip tiling + tier fallback chain
│   ├── test_snr.py               #   Brouhaha SNR extraction
│   ├── test_esc.py               #   PANNs ESC
│   ├── test_vad_processing.py
│   ├── test_reproducibility.py
│   ├── test_stitched_audio.py
│   └── test_create_dataloader.py
├── docs/
│   └── DATALOADER_DESIGN.md      # Dataloader++ specification
├── pyproject.toml
└── README.md
```

### Data flow

```
$DLPP_WORKSPACE/
├── manifests/{dataset}.csv       (input manifest)
├── output/{dataset}/             (pipeline outputs — VAD, VTC, SNR, ESC, shards)
├── figures/{dataset}/            (diagnostic plots)
└── logs/                         (SLURM job logs)
```

### Running tests

```bash
# Login node (TenVAD tests auto-skip on non-compute nodes):
uv run python -m pytest tests/

# Compute node (full suite):
sbatch slurm/test.slurm
```

---

## 6. Dataloader

The `dataloader/` package implements the **Dataloader++** specification for Meta's speech training infrastructure. It bridges the offline feature processing pipeline (above) with online model training.

See [`src/dataloader/README.md`](src/dataloader/README.md) and [`docs/DATALOADER_DESIGN.md`](docs/DATALOADER_DESIGN.md) for the full design document.

| Component | Location | Purpose |
|-----------|----------|---------|
| **Feature Processor** | `dataloader/processor/` | ABC wrapping offline extraction stages (VAD, VTC, SNR, ESC) |
| **Feature Loader** | `dataloader/loader/` | Load waveforms + metadata from WebDataset shards or raw files |
| **Manifest Joiner** | `dataloader/manifest/` | Join heterogeneous metadata manifests by `wav_id` (the "Big Join") |
| **Data Processor** | `dataloader/transform/` | Composable runtime transforms (segment, resample, encode, mask) |
| **Collator / DataBatch** | `dataloader/batch/` | Pad variable-length samples into typed `DataBatch` tensors |
| **Dataset** | `dataloader/dataset/` | PyTorch `Dataset` implementations (WebDataset-backed) |

---

## 7. Citation

```bibtex
@software{dlplusplus,
    title  = {{DL++}: Feature Processing and Data Loading for Child-Centered Long-Form Audio},
    author = {Dager, Daniel and Kunze, Tarek and Charlot, Théo and Cristia, Alejandrina and Dupoux, Emmanuel and Lavechin, Marvin},
    year   = {2026},
    url    = {https://github.com/LAAC-LSCP/DLplusplus},
}
```

---

## 8. Component Models

DL++ integrates the following models as feature processing stages:

### TenVAD — Voice Activity Detection

[Tencent/TenVAD](https://github.com/Tencent/TenVAD) — lightweight speech activity detector used in Step 1 (CPU).

### BabyHuBERT — Voice Type Classification (VTC 2.0)

Speaker diarization into four types (KCHI, OCH, MAL, FEM), trained on child-centered long-form recordings. Used in Step 2 (GPU).

Weights are downloaded automatically to `$MODEL_ROOT/vtc/` by `uv run download-models`.

Training code: [LAAC-LSCP/BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT)

```bibtex
@misc{charlot2025babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings},
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001},
}
```

### Brouhaha — SNR & C50 Estimation

[marianne-m/brouhaha-vad](https://github.com/marianne-m/brouhaha-vad) — per-frame signal-to-noise ratio and clarity (C50) extraction. Used in Step 3 (GPU).

Weights are downloaded automatically to `$MODEL_ROOT/brouhaha/` by `uv run download-models`.

```bibtex
@inproceedings{lavechin2023brouhaha,
    title     = {Brouhaha: Multi-task Training for Voice Activity Detection, Speech-to-Noise Ratio, and Speech Reverberation Estimation},
    author    = {Marvin Lavechin and Marianne Métais and Hadrien Titeux and Alodie Boissonnet and Johan Music and Hervé Bredin and Emmanouil Benetos and Alejandrina Cristia},
    year      = {2023},
    booktitle = {2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
    doi       = {10.1109/ASRU57964.2023.10389642},
}
```

### PANNs CNN14 — Environmental Sound Classification (ESC)

[qiuqiangkong/panns_inference](https://github.com/qiuqiangkong/panns_inference) — AudioSet-based sound event detection (527 classes, grouped into 13 coarse categories). Used in Step 4 (GPU).

```bibtex
@inproceedings{kong2020panns,
    title     = {PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition},
    author    = {Qiuqiang Kong and Yin Cao and Turab Iqbal and Yuxuan Wang and Wenwu Wang and Mark D. Plumbley},
    year      = {2020},
    journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume    = {28},
    pages     = {2880--2894},
    doi       = {10.1109/TASLP.2020.3030497},
}
```

---

## 9. Acknowledgements

This work uses the [segma](https://github.com/arxaqapi/segma) library, inspired by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015450 and 2025-AD011016414)
