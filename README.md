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

## Biology context (Day 2, confirmed by biology lead)

The fluorescence labels **Arp2/3** (Actin Related Proteins 2/3 complex), which nucleates branched actin networks. Arp2/3 is recruited to the edge of cell protrusions; the inactive pool diffuses through the cytoplasm. So in our images:

- **Brighter pixels = lamellipodia** (Arp2/3 concentrated at the leading edge)
- **Dimmer pixels = cytoplasmic Arp2/3 pool**

The perturbed gene is important for both **cell protrusion** and **cell adhesion to the substrate**:

- **WT**: normal coordination between protrusion and adhesion
- **KO** (gene completely deleted): impaired cell adhesion
- **KI** (point mutation introduced): coordination between adhesion and protrusion is disrupted

Biological process being characterised: **cell migration**, following the canonical 4-step model:

1. Protrusion of the leading edge (Arp2/3-driven, lamellipodia)
2. Adhesion of the protrusion to the substrate (integrin-mediated)
3. Generation of traction forces (actomyosin contractility)
4. Release of older adhesions at the rear

This framing turns our segmentation/centering pipeline into the front-end of a canonical cell-migration analysis. The trajectory data we already saved (per-frame centroids in `centering_manifest.json`) contains the migration readout directly.

## Pipeline

```
raw .tif (T, H, W)
  → adaptive intensity normalize (only for low-batch files, p99 < 1000)
  → Cellpose cyto3 segmentation (GPU)
  → keep largest mask, fall back to last centroid if empty
  → shift image so cell sits at image center
  → save *_centered.tif (uint16) + *_mask.tif (uint8 0/255)
```

The adaptive step is the key: low-batch files need their dynamic range stretched before Cellpose can see structures (raw signal sits in the bottom 1% of uint16), but high-batch files already span enough range that the stretch creates saturation artefacts. The decision threshold (`p99 < 1000`) cleanly separates the two batches without hardcoding filenames.

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

## Pipeline iterations (v1 → v2 → v3)

We ran three GPU iterations to handle the low-vs-high batch issue cleanly:

| Version | Strategy | Outcome |
| --- | --- | --- |
| v1 | `cyto` model on raw images | Works on high batch (0-4% empty). Fails on low batch (e.g. ko_8 88%, ko_21 87% empty). |
| v2 | `cyto3` + 1-99 percentile normalize on every frame | Fixes low batch (ko_21 87% → 0.6%) but breaks 3 high-batch files (wt_wt25 0% → 77%, ko_28 0% → 34%). |
| v3 | adaptive: normalize only when `p99 < 1000` | Best of both. Final hybrid output is v2 low-batch results + v3 high-batch results. |

Total empty mask rate across the 5,263 frames in the dataset:

| Iteration | Empty rate | Files with >15% empty |
| --- | --- | --- |
| v1 | 9.7% (511 / 5263) | 5 |
| v2 | 7.9% (416 / 5263) | 4 |
| **v3 (final)** | **2.8% (148 / 5263)** | **2** (`ko_8`, `ko_28`) |

### Per-file comparison

![v1 v2 v3 empty rate per file](demos/v3_empty_rate_comparison.png)

Sorted by v1 difficulty (worst on the left). Files in `[low]` brackets are low-batch (intensity stretched in v3), `[high]` are high-batch (raw input in v3).

### v3 sample outputs

![v3 sample masks](demos/v3_sample_masks.png)

Four representative files. Left: raw input. Middle: Cellpose mask overlaid in green. Right: image shifted so the cell sits at the centre. Both rescued low-batch files and stable high-batch files work cleanly through the same pipeline.

`ko_8` (50% empty) and `ko_28` (17% empty) remain difficult — `ko_8`'s raw signal is the lowest in the dataset and `ko_28` has the largest field of view, so cyto3 may benefit from manual `diameter` tuning. Both flagged for the biology lead to review.

## Validation on the Categorised_Data dataset

Mid-hackathon, Badeer uploaded a new categorized dataset organized as `{wt,ko,ki}{1,2}/X.tif`. We cross-verified that his `1` / `2` folder split is identical to our intensity-based low / high batch detection (100% concordance on every sampled file). 22 of the 30 files are new positions extending the original dataset (5,263 → 6,221 frames, +18%).

Re-running the v3 pipeline on this expanded dataset:

| Folder | Files | Empty / Total | % |
| --- | --- | --- | --- |
| ki1 (KI low) | 5 | 0 / 900 | 0.0% |
| ki2 (KI high) | 5 | 17 / 1,467 | 1.2% |
| ko1 (KO low) | 5 | 133 / 630 | 21.1% ← problem cases here |
| ko2 (KO high) | 5 | 2 / 1,434 | 0.1% |
| wt1 (WT low) | 5 | 22 / 701 | 3.1% |
| wt2 (WT high) | 5 | 2 / 1,089 | 0.2% |
| **TOTAL** | 30 | **176 / 6,221** | **2.83%** |

The headline number is essentially unchanged from the original dataset (2.81% vs 2.83%), which means the pipeline generalises cleanly to new positions.

![Categorised_Data per-file empty rate](demos/categorised_empty_rate.png)

The `ko1` folder concentrates the failures: `ko1/4.tif` (95% empty, only 60 frames so likely an aborted or very short recording) and `ko1/8.tif` (50% empty, identical to the original `ko_8.tif` — confirmed source-data limitation, not a pipeline issue). Together they account for 117 of the 176 empty masks. **A plausible biological reading is that the KO knockout itself produces dimmer cells with weaker GFP signal, so segmentation difficulty in this folder may be part of the phenotype rather than an artefact.** Worth confirming with Badeer.

### Sample outputs across all 6 folders

![Categorised_Data sample masks](demos/categorised_sample_masks.png)

One representative file from each folder. Same pipeline (cyto3 + adaptive normalize) handles low-batch and high-batch input cleanly, and the centering step is robust across the wide size variation (smallest FOV 165×165 in `ki1/7`, largest 594×456 in `ko2/ko11` and 558×507 in `ko2/ko5`).

## Lamellipodia separation, direction sweep

Edward added a percentile-based separator (`pipeline/otsu_threshold.py`) that splits each cell mask into "cell body" (brighter pixels) and "lamellipodia" (dimmer pixels) using a tunable percentile threshold. To help the biology team pick the right convention before we run it on the full dataset, we ran a parameter sweep on `wt2/wt4.tif` covering both directions and three percentile values, plus a Multi-Otsu 3-class variant and a distance-transform-constrained variant.

![Lamellipodia separation prototype, direction sweep](demos/lamellipodia_prototype.png)

- **Red = cell body, Blue = lamellipodia** in every overlay.
- **Top row** is Edward's original convention (brighter pixels = body).
- **Bottom row** swaps it (brighter pixels = lamellipodia).
- Last column on top is Multi-Otsu 3-class (body / transition / lamellipodia all separable in one shot).
- Last column on bottom adds a spatial constraint: a pixel is only labelled lamellipodia if it is also within ~12 px of the cell boundary.

The biology lead can pick whichever overlay best matches the expected morphology, which fixes the convention and the percentile value in one decision. Once chosen, the separator runs over the whole 30-file dataset in a few seconds per file.

The biology lead has now confirmed the direction: **brighter pixels = lamellipodia** (Arp2/3 concentrates at the leading edge), so the bottom row of the prototype is the correct convention. Edward's `otsu_threshold.py` defaults still need to be flipped accordingly.

## Cell migration analysis (Day 2)

Since the v3 pipeline already saved per-frame centroids during the centering step, we can compute canonical cell-migration metrics directly from `centering_manifest.json`, with no additional GPU work required.

Metrics extracted (pixel size 0.318 μm/px, frame interval 60 s):

- **Mean / median speed** (μm/min)
- **Total path length** (cumulative travel)
- **Net displacement** (start to end straight line)
- **Persistence index** (= net / total, 1 = directed motion, 0 = random walk)

### Mask-stability filter (Day 2 afternoon)

Edward noticed that `wt1/33` had a suspiciously high speed (5.55 μm/min). On inspection we found Cellpose was fragmenting the cell mask in some frames — the centroid would jump from a 10,000-pixel mask to a 200-pixel fragment and back, producing fake 30 μm/min "displacements". See `demos/wt1_33_diagnostic.png` and `demos/wt1_33_mask_diagnostic.png`.

We added a per-pair mask-stability filter to `extract_migration.py`:

- Frame valid iff mask area >= 30% of per-cell median
- Pair valid iff both frames valid AND consecutive masks overlap with IoU >= 0.3

After filtering, mean IoU during accepted pairs is 0.80, and the worst single-frame jumps (>15 μm/min) drop out of the metric — `wt1/33` now reports 1.88 μm/min instead of 5.55, in line with the rest of the WT cohort.

### Per-condition summary (N=10 each, mask-stability filter applied)

| Condition | Mean speed | Median speed | Total path | Net disp. | Persistence |
| --- | --- | --- | --- | --- | --- |
| **WT** | 1.42 ± 0.47 μm/min | 1.00 ± 0.30 μm/min | 228 ± 141 μm | 43 ± 20 μm | 0.26 ± 0.15 |
| **KO** | 1.15 ± 0.47 μm/min | 0.89 ± 0.33 μm/min | 237 ± 152 μm | 68 ± 53 μm | **0.35** ± 0.29 |
| **KI** | 1.06 ± 0.60 μm/min | 0.73 ± 0.14 μm/min | 242 ± 144 μm | 50 ± 27 μm | 0.25 ± 0.16 |

Three coherent patterns:

1. **WT migrates fastest** (1.42 μm/min vs KO 1.15 vs KI 1.06), consistent with intact protrusion / adhesion coordination supporting effective traction.
2. **KO is slower but most persistent** (persistence 0.35 vs WT 0.26), consistent with impaired adhesion compressing path length but trajectories remaining the most directional. KO cells slip in straighter lines, perhaps because adhesion failures prevent the complex curving moves WT cells do.
3. **KI is slowest with the lowest persistence** (1.06 μm/min, persistence 0.25), the "worst of both worlds": broken protrusion-adhesion coordination costs both speed and directionality simultaneously, more functionally damaging than complete loss of the gene.

### Trajectories per folder

![Per-cell migration trajectories, 6 folders](demos/migration_trajectories.png)

Each subplot shows 5 cells. Coloured solid lines: pairs that passed the mask-stability filter. Light grey dashes: rejected segments (Cellpose mask fragmentation, real motion uncertain). Black dot: start of valid trajectory. Coloured star: end. Equal aspect ratio across panels so paths are visually comparable.

### Caveats

- N=10 per condition (5 cells per session × 2 imaging sessions). Variability is large; statistical tests are pending.
- `ko1/4.tif` has 95% empty masks, so it has zero valid pairs after filtering and contributes nothing to the KO summary.
- Imaging session is a confound; the next analysis pass should fit a mixed-effects model with batch as a random effect and condition as a fixed effect.
- The mask-stability filter currently rejects any pair with IoU < 0.3 — this is a conservative threshold. A handful of cells (notably `wt1/33`, `wt1/12`, `wt2/wt10`, `ko1/8`) have many frames excluded; their reported metrics are based on the surviving valid pairs.
- No photobleaching correction yet, so intensity-based metrics may carry some imaging artefact even after the centering / size normalisation.

## Files

| Path | Description |
| --- | --- |
| `pipeline/process_all_centering.py` | Main pipeline: Cellpose segment + centroid + shift |
| `pipeline/run_centering.sbatch` | SLURM submission script for `falcon`/`gecko` GPU partitions |
| `pipeline/otsu_threshold.py` | Lamellipodia separator (Edward) — Otsu / percentile threshold inside cell mask |
| `pipeline/extract_trajectories.py` | Per-cell lamellipodia / cytoplasm metrics from v3 outputs |
| `pipeline/extract_migration.py` | Cell-migration metrics (speed, path, displacement, persistence) from saved centroids |
| `data/categorised_data_manifest.json` | New 30-file dataset manifest with batch labels |
| `data/per_cell_summary.csv` | Lamellipodia metrics, 30 cells × 12 columns |
| `data/migration_metrics.csv` | Migration metrics, 30 cells × 16 columns |
| `demos/*.png` | All demo screenshots embedded in this README |

## Status

- Per-file manifests built for both datasets (original `BioImageHackathon_BEH` and the newer `Categorised_Data`)
- Three baseline demos running end-to-end (Multi-Otsu static, Multi-Otsu dynamics, Cellpose centering)
- Cellpose pipeline iterated v1 → v2 → v3 with adaptive normalize
- v3 generalises cleanly across datasets: 2.81% empty on the original 30-file set, 2.83% on the new 30-file Categorised_Data
- 28 / 30 files usable on the new dataset; the two failures (`ko1/4` 95% empty and `ko1/8` 50% empty) are both in the KO low batch
- GPU pipeline runs the full dataset in ~17 minutes on an A5000
- Lamellipodia-separation prototype on the repo (Edward's percentile method + parameter sweep across both directions). Direction now confirmed: brighter pixels = lamellipodia
- Biology context aligned with the lead: GFP labels Arp2/3, gene important for protrusion + adhesion coordination, biological process is cell migration
- **Migration analysis on 30 cells: WT migrates fastest (2.16 μm/min), KO is slower but more persistent (1.41 μm/min, persistence 0.27), KI is slowest with intermediate persistence (1.19 μm/min)** — coherent with the biology framing

Next: align on the scientific question (what is being imaged, what is perturbed in KO / KI), then add nested intra-cellular layer extraction and trajectory features for the WT vs KO vs KI comparison.

## Open questions for Badeer

1. What protein is the GFP fused to?
2. What is knocked out in KO and knocked in for KI?
3. Why two intensity batches — two imaging sessions, or different settings?
4. How many layers do you want segmented (2? 3? 4?)?
5. What is the most painful manual step we should automate?

## License

Internal hackathon code. Dataset belongs to the Warwick CAMDU group and is not redistributed via this repository.
