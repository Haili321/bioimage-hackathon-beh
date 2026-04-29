# BioImage Hackathon — BEH Dataset

GPU-accelerated pipeline for nested cell segmentation and dynamics quantification on a 30-file epiflourescense timelapse dataset. Built for the Warwick BioImage Analysis Hackathon (April 2026).

**Live briefing page:** [https://www.dcs.warwick.ac.uk/~u1898019/hackathon-beh/](https://www.dcs.warwick.ac.uk/~u1898019/hackathon-beh/)

## Team

| Name | Role |
| --- | --- |
| Haili Yuan (CS) | pipeline, dynamics, visualization |
| Edward Offord (CS) | segmentation, cell centering |
| Badeer Ummat (biology) | dataset, scientific question |

## Dataset at a glance

| Property | Value |
| --- | --- |
| Files | 30 (3 conditions × 10 positions) |
| Conditions | WT (Wild-Type), KO (Knock-Out), KI (Knock-In) |
| Modality | 2D time-lapse epifluorescence, single channel |
| Acquisition | Zeiss + Micro-Manager 2.0, 60 s frame interval |
| Pixel size | ~0.318 µm/px |
| Bit depth | 16-bit unsigned |
| Frames per file | 67 to 360 (varies by position) |
| Total dataset | 1.3 GB, ~5,700 frames |

## Key finding: two intensity batches

While inspecting the data we discovered each condition cleanly splits into two intensity batches, with a 5-10x gap in pixel intensity. Almost certainly two separate imaging sessions, not biological variation. Implications:

- Cannot normalise globally across all 30 files
- Statistical comparison should use a mixed-effects model with batch as a random effect
- Findings should hold across both batches to be considered robust

| Condition | Low batch (mean ~200-240) | High batch (mean ~500-700) |
| --- | --- | --- |
| WT | wt_5, wt_9, wt_23, wt_31, wt_37 | wt_wt4, wt_wt9, wt_wt13, wt_wt18, wt_wt25 |
| KO | ko_3, ko_8, ko_17, ko_21, ko_25 | ko_11, ko_1717, ko_2121, ko_28, ko_333 |
| KI | ki_3, ki_11, ki_14, ki_23, ki_25 | ki_1, ki_6, ki_13, ki_18, ki_28 |

## Pipeline

```
raw .tif (T, H, W)
  → per-frame intensity normalize (1-99 percentile stretch to full uint16)
  → Cellpose cyto3 segmentation (GPU)
  → keep largest mask, fall back to last centroid if empty
  → shift image so cell sits at image center
  → save *_centered.tif (uint16) + *_mask.tif (uint8 0/255)
```

## Run

### On the GPU partition (recommended)

```bash
sbatch pipeline/run_centering.sbatch
```

The job uses 1 GPU on the `falcon` or `gecko` partition, 8 CPUs, 32 GB RAM, with a 2-hour wall-clock limit. On an NVIDIA RTX A5000 the full 30-file dataset processes in ~17 minutes.

Output goes to `~/Desktop/BioImageHackathon_BEH_centered/` with:

```
BioImageHackathon_BEH_centered/
├── wt/  (10 paired _centered.tif + _mask.tif)
├── ko/
├── ki/
└── centering_manifest.json   (per-file metadata + per-frame centroids)
```

### Locally on CPU (slow, for prototyping)

```bash
python pipeline/process_all_centering.py
```

CPU processing is ~12.6 s/frame versus ~0.18 s/frame on GPU.

## Demos

### Day 0 baseline 1: Multi-Otsu 4-layer segmentation
Static mid-frame segmentation per condition using only classical thresholding (no deep learning). Verifies the data IO and tooling chain runs end-to-end.

![Multi-Otsu baseline](demos/demo_v1_multiotsu.png)

### Day 0 baseline 2: Dynamics through time
Same approach extended to full timelapse with global thresholds. Bottom plot shows per-frame ring intensity for the entire time-lapse (745 frames across 3 conditions).

![Dynamics demo](demos/demo_v2_dynamics.png)

### Cellpose + cell centering (Edward's pipeline)
Per-frame Cellpose segmentation followed by centroid-based image shift. Sanity check on `wt_wt9.tif` (3 frames, CPU).

![Cellpose sanity check](demos/cellpose_sanity_check.png)

### Failure case: low-batch + Cellpose `cyto`
Running Cellpose with default settings on `ko_8` (low batch) returns empty masks on 88% of frames. Same algorithm works fine on `ki_18` (high batch). The intensity histogram at the bottom shows why: the low batch occupies less than 1% of the available 16-bit dynamic range.

![Failure case](demos/cellpose_failure_comparison.png)

### Fix: per-frame normalization
Stretching each frame to its 1-99 percentile range before segmentation. Now Cellpose sees consistent contrast on both batches.

![Normalization fix](demos/normalize_demo.png)

## Files

| Path | Description |
| --- | --- |
| `pipeline/process_all_centering.py` | Main pipeline: Cellpose segment + centroid + shift |
| `pipeline/run_centering.sbatch` | SLURM submission script for `falcon`/`gecko` GPU partitions |
| `data/dataset_manifest.json` | Per-file metadata: shape, dtype, intensity range, batch label |
| `demos/*.png` | Day-0 demo screenshots embedded in this README |

## Status

- Data downloaded, verified, manifest built
- Three demo notebooks running end-to-end (Multi-Otsu static, Multi-Otsu dynamics, Cellpose centering)
- Failure mode on low-batch identified, fix in place (normalize + cyto3)
- GPU pipeline deployed, full-dataset run takes ~17 minutes on A5000

Next: align with Badeer on the scientific question (what protein is GFP fused to, what genes are perturbed in KO/KI), then add nested intra-cellular layer extraction and trajectory features for the WT vs KO vs KI comparison.

## Open questions for Badeer

1. What protein is the GFP fused to?
2. What is knocked out in KO and knocked in for KI?
3. Why two intensity batches — two imaging sessions, or different settings?
4. How many layers do you want segmented (2? 3? 4?)?
5. What is the most painful manual step we should automate?

## License

Internal hackathon code. Dataset belongs to the Warwick CAMDU group and is not redistributed via this repository.
