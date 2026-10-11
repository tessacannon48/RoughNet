
# Mapping Arctic Ice Terrain Using Diffusion-Based Super-Resolution of Satellite Imagery
![alt text](https://github.com/tessacannon48/RoughNet/blob/main/figures/final_figures/model_diagram.png)

## About The Project

### Goal

This research develops a custom conditional diffusion model to super-resolve satellite imagery of Arctic landfast ice into high-resolution topographic maps. Specifically, the model learns a mapping from 10-m Sentinel-2 images to 1-m locally normalized DEMs, corresponding to a 100× increase in spatial resolution.

### Motivation

The motivation of this study is to enable safer navigation across landfast ice for Indigenous communities in the Arctic. Through the generation of high-resolution topography derived from LiDAR, this work seeks to provide accurate representations of sea-ice surface roughness (SIR), which is a key indicator of ice safety.
  
### Repository Structure

```
Dissertation
├── config.yaml                   # Project configuration file
├── copernicus_login.py           # Copernicus data service authentication
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── figures/                      # Figures and visualizations
│   ├── model_diagram.png
│   ├── noise_schedule.png
│   └── ...                       # Other figure files
├── input_data/                   # Input data for experiments (samples only)
│   ├── sample_lidar_patch/
│   └── sample_s2_patch/
├── models/                       # Trained model checkpoints (.pth)
├── notebooks/                    # Jupyter notebooks for dataset construction and analysis
├── scripts/                      # Python scripts for modeling and evaluation
├── src/                          # Source code for the project
```

---

## Data Source

### LiDAR 

**Provider:** Private research team. 

Data for this study were collected from three Arctic regions home to Inuit communities in Northern Canada: Pond Inlet
(Baffin Island), Tuktoyaktuk (Inuvik Region), and Cambridge Bay (Victoria Island) This diverse range of Arctic
regions was chosen for data collection due to the relevance they hold for Inuit populations, allowing us to train models
best suited for the terrain used by indigenous communities for travel. LiDAR data were collected through airborne
surface topography mapping during the annual IceBird sea ice surveying campaigns using a RIEGL LMS VQ-580 laser
scanner mounted on a Basler BT67 airplane in April 2024 (Pond Inlet: 26-04-24, Cambridge: 18-04-24, Tuktoyaktuk: 16-04-24; Haas et al., 2024). 

### Satellite Images
**Provider:** [ESA Copernicus Data Space](https://dataspace.copernicus.eu/)

The multispectral satellite imagery used in this study is obtained from the European Space Agency’s (ESA) Sentinel-2 mission. Four of the 13 bands, RGB+NIR (10 m), were selected for this study due to their high spatial resolution and their ability to capture fine-scale surface texture and reflectance characteristics.

---

## Dataset Construction
![alt text](https://github.com/tessacannon48/RoughNet/blob/main/figures/final_figures/sample_patch.png)

1. Preprocessing
- Script: `/Dissertation/scripts/lidar_preprocessing.py`  
- The LiDAR data were originally recorded as three-dimensional point clouds at 1m resolution (WGS84). The coordinates were reprojected into a locally optimized, meter-based coordinate system using a custom Transverse Mercator projection. To remove large-scale elevation trends and emphasize local surface roughness, the raw elevations were converted to RANSAC residuals by fitting a quadratic surface to each dataset. 

2. Geolocation
- Notebook:`/Dissertation/notebooks/data_collocation.ipynb`  
- To identify valid Sentinel-2 imagery for training, a querying pipeline was developed using the Copernicus Data Space Ecosystem (CDSE) API to match optical satellite images with the spatial extent of the airborne LiDAR data for each of the three data collection regions.

3. Patching
   
- Notebook: `/Dissertation/notebooks/patching.ipynb`
- The dataset construction followed three main steps. First, the LiDAR GeoTIFF tiles were mosaicked into a unified, geographically aligned grid. Second, a sliding window of 256⇥256 pixels with a stride of 128 pixels (50% overlap) was applied to extract LiDAR patches. For each patch, the geographic bounds were reprojected from the LiDAR coordinate reference system (CRS) to the Sentinel-2 CRS in order to extract the corresponding 26⇥26 pixel windows from each of the six Sentinel-2 products.

5. Transformations
- Script: `/Dissertation/scripts/main.py` 
- The dataset class used to create the input dataset applies several selections and transformations to adequately prepare the data for modeling. First, each Sentinel-2 patch is resized to the dimensions of the LiDAR patch (256×256 pixels) using bilinear interpolation. Sentinel-2 data is normalized on a per-patch basis: for each band (R, G, B, NIR), all Sentinel-2 products are pooled, clipped to the 2nd--98th percentiles, and linearly rescaled to [0,1] with a minimum range safeguard of 1x10^[-3]. The LiDAR input consists of RANSAC residual elevations, which are further processed using patch-wise mean removal computed only over valid pixels. Specifically, the mean residual elevation of each patch is subtracted from all valid pixels so that the model learns local roughness structure rather than absolute elevation offsets, while invalid pixels are masked to zero. The training set is then randomly augmented to increase sample diversity and improve model robustness. Finally, the attributes are parsed for each Sentinel-2 patch and encoded as follows: cloud coverage percentage is scaled to [0,1], image age is represented as a signed scalar corresponding to the temporal difference between Sentinel-2 and LiDAR acquisition dates (scaled approximately in months), Zenith angles are scaled to [0,1], and Azimuth angles are transformed into sine and cosine components using sinusoidal encoding. The difference in angle encoding arises because Zenith angles lie in a range of 0 degrees to 90 degrees, representing the vertical angle between an object and a line pointing upward from the ground, whereas Azimuth angles lie in a range of 0 degrees to 360 degrees, representing the horizontal angle measured clockwise from True North.

---

## Model

The model is a conditional U-Net diffusion architecture designed for cross-modal generation. It takes the noisy single-channel LiDAR patch as input and conditions on a collocated, fused representation of Sentinel-2 patches, where the Sentinel-2 inputs are spatially aggregated using learned, attribute-dependent weights. The network is trained to iteratively recover high-resolution synthetic elevation maps from noisy inputs, guided by the conditioning. The forward sampling process follows the Pseudo Linear Multi-Step (PLMS) framework in order to accelerate convergence and improve sample quality.

Note that the modeling setup enables dynamic adjustment of the model architecture to allow for ablation studies of architectural variants and sampling methods. 

**Inputs**  
- LiDAR residual map: `[1, H, W]`  
- Sentinel-2 context: `[4*6, H, W]` (4 bands × 6 patches)  
- Attributes: `[8k]` (per-patch attributes)  
- Diffusion timestep: `[1]`  

**Architecture**
- **Base**: U-Net with dynamic depth (default = 4) and base channels (default = 128).  
- **Conditioning**: Sentinel-2 patches (4 bands × 6), metadata vectors, and diffusion timestep. Conditioning injected at every block via FiLM-modulated GroupNorm.  
- **Encoder/Decoder**: Standard downsampling (MaxPool + DoubleConv) and upsampling (TransposeConv + DoubleConv) with skip connections.  
- **Attention**: Optional self-attention modules, default = bottleneck only.  
- **Blocks**: DoubleConv = two 3×3 convs with GroupNorm, FiLM conditioning, and GELU activation.  
- **Output**: Final 1×1 conv producing a single-channel LiDAR residual map.  
- **Dynamic behavior**: Depth, channels, number of context patches (*k*), and attention placement can all be varied for experiments.

## Training Configuration

### Default Training parameters
- **Batch Size**: 8
- **Epochs**: 200
- **Learning Rate**: 0.0001
- **Timesteps**: 1000
- **Noise Schedule**: Linear
- **Loss Function**: MSE (masked on valid LiDAR regions)
- **Context *k***: 1
- **Randomize Context**: False

### Experiments

The code in this repository is designed to allow dynamic configuration of model architecture and hyperparameters. Below are the parameters, architectural variants, and methods tested during a series of controlled ablation studies to determine the optimal model approach for this task. Alternative values of hyperparameters can be tested using command-line arguments to adjust the config.  

- **Baseline tuning**:
  - Input: 1 Sentinel-2 patch + attributes, output: 256×256 LiDAR residuals  
  - Diffusion: DDPM (1000 steps), loss: Masked MSE  
  - Sweeps:  
    - Learning rate (1e-3, 1e-4, 1e-5)  
    - Noise schedule (linear vs. cosine)  
    - Embedding dimension (128, 256)  
    - Loss variations: Masked MAE, Hybrid (MAE/MSE + gradient loss, λ = {0.1, 0.5, 1.0})  

- **Architectural variations**  
  - Tested on baseline:  
    - Attention placement (bottleneck, medium attention, heavy attention)  
    - UNet depth (shallow (3) vs. deep (5))  
    - Channel width (narrow vs. wide)  

- **Sampling strategies**  
  - Compared deterministic samplers: DDPM, DDIM, PLMS  
  - Measured trade-off between reconstruction quality and runtime  

---

## Limitations and Constraints

- During model development, training was limited to 50 epochs in order to balance computational resource use with experimental breadth.
- A sequential search strategy was adopted with only a limited number of overlapping parameter combinations tested, rather than a full grid search, due to computational and time constraints.
- Evaluation metrics used to judge model variants not perfect indicators of reconstruction quality.
---

## Setup & Execution

Follow the steps below to setup and execute the project with your LiDAR data.

### Installation
1. Set up the environment:

```bash
git clone https://github.com/tessacannon48/Dissertation.git
cd Dissertation
pip install -r requirements.txt
```

2. Set up LiDAR data:
- Download LiDAR data to /raw_data folder.
- LiDAR data should be preprocessed as RANSAC residuals such that they are roughly normally distributed around zero.

### Data Collocation

1. Launch Jupyter and open the notebook:
```bash
jupyter notebook data_collocation.ipynb
```
   - This notebook identifies all Sentinel-2 Level-2A products that cover the LiDAR collection area, filters for usable (cloud-free) imagery, and prepares them for training. 

2. In cell 4, set:
   - Your Copernicus Data Space (CDSE) username and password (or import it from separate file)
   - The LiDAR `.tif` directory path (e.g., `raw_data/pondinlet_lidar_2024`)
   - The date range for Sentinel-2 products to query (+/- 14 days of the LiDAR collection date)
     
3. Execute cells 4-6 to:
   - Cell 4: Query CDSE to find Sentinel-2 products that overlap with the LiDAR area.
   - Cell 5: Filter the CDSE search results to products that cover 100% of the LiDAR area.
   - Cell 6: Visualize tiles of all of the remaining Sentinel-2 products (single band visualization to reduce processing). 

4. In cell 7, set: 
   - The directory where you would lke the products to be downloaded to (e.g., `./raw_data/pondinlet_sentinel_downloads`)
   - After visually inspecting the figures from Cell 5, enter a list of the best products (identified by the integer label in the subtitle of each image) from best to worst visibility in the `selected_product_indices` list (e.g., [8,16,22,21,23,6])

5. Execute cells 7-8 to:
   - Cell 7: Download the selected Sentinel-2 products to the specified directory. 
   - Cell 8: Unzip the zip files (numbered by the custom order you identified in cell 6). 

### Patching

1. Launch Jupyter and open the notebook:
```bash
jupyter notebook patching.ipynb
```
   - This notebook extracts spatially aligned Sentinel-2 and LiDAR patch sets to prepare datasets for model training.

2. Set the input and output paths in cell 3:
  - Replace these with your own Sentinel-2 and LiDAR paths.
  - Output directories are where the patch sets will be stored.
     
3. Execute cell 5 to:
   - Divide LiDAR into 10 regions with roughly equal number of patches
   - Execute patching:
      - Stacks Sentinel-2 bands (RBG + NIR)
      - Slides window across LiDAR region
      - Transforms LiDAR bounds to extract corresponding Sentinel-2 patch
      - Saves each Sentinel-2 patch set + metadata and LiDAR patch set to out directories (patches are paired using tile IDs ex. 00001)

### Training

Run the following commands from `Dissertation/`, using a Python environment
with the project dependencies installed and the paired patches in `input_data/`.
The corrected splits retain the validation zones and exclude training neighbours
whose combined LiDAR and six Sentinel-2 footprints intersect validation
footprints. Zone IDs alone do not prevent overlap. No patch re-extraction is needed.

Prepare the local footprint manifests:

```bash
python scripts/prepare_spatial_splits.py
```

This creates the two full split manifests, compact ID assignments and cosine
training configurations under `splits/`. The full JSON manifests record local
file sizes and timestamps and are excluded from Git; retain the manifest used
for each checkpoint. Preparation refuses to replace a different existing full
manifest. The committed compact IDs provide portable sample membership.

| Configuration | Training regions | Train / validation | Held-out test |
|---|---|---:|---|
| `original_clean` | Pond Inlet + Tuktoyaktuk | 11,076 / 1,290 | Cambridge Bay: 2,111 |
| `original_clean_linear` | Pond Inlet + Tuktoyaktuk | 11,076 / 1,290 | Cambridge Bay: 2,111 |
| `tuk_heldout_clean` | Pond Inlet + Cambridge Bay | 11,589 / 1,333 | Tuktoyaktuk: 1,676 |

The configurations freeze six randomized context views, depth 4, 64 base
channels, mid attention, hybrid masked MSE with alpha 1, learning rate 1e-4,
batch size 8 and 200 epochs. Only `original_clean_linear` uses the linear
schedule; the other two use cosine. Each model is trained from scratch and
selected by source-region validation loss.

```bash
bash scripts/train_clean_fold.sh original_clean
bash scripts/train_clean_fold.sh original_clean_linear
bash scripts/train_clean_fold.sh tuk_heldout_clean
```

The training wrapper uses `../.venv-py313/bin/python`, disables W&B uploads,
locks each fold against duplicate writers and appends a local training log.
With an interpreter installed elsewhere, invoke the training CLI directly;
replace the configuration filename for the other folds:

```bash
WANDB_MODE=disabled python -u scripts/main.py --config splits/original_clean.yaml --resume
```

Rerunning the same command resumes from the latest completed epoch with the
saved optimizer and RNG state. Keep the configuration and split unchanged.
Use `models/<fold>/<fold>_best.pth` for inference; `_latest.pth` holds recovery
state. Full evaluation is a separate step.

For baseline collaborators, share the tracked
[training and evaluation split guide](docs/baseline_data_split.md) and
[splits/train_val_split_ids.json](splits/train_val_split_ids.json). Fit weights
and learned preprocessing on `train` IDs only, select on `validation`, and
evaluate Cambridge Bay after choices are fixed. Excluded IDs are not training
samples. The alternate fold's assignments are in
[splits/tuk_heldout_clean_ids.json](splits/tuk_heldout_clean_ids.json).

### Evaluation

`scripts/evaluation.py` performs inference, mosaicking, figure generation and
metric computation. Its default cosine checkpoint is now
`models/original_clean/original_clean_best.pth`; `--model linear` selects the
corrected linear checkpoint. Use `--checkpoint PATH` to override it and
`--root PATH` when running outside the directory containing `models/` and
`input_data/`. Fresh inference outputs go under `figures/original_clean/`
(or `figures/original_clean_linear/`) rather than the historical figure folders.

To regenerate the three paper mosaics from the completed cosine PLMS evaluation,
run from `Dissertation/`:

```bash
python scripts/evaluation.py --region tuk --mosaics-only
python scripts/evaluation.py --region cambridge --mosaics-only
python scripts/evaluation.py --region pondinlet --mosaics-only
```

This uses `reconstructions/noise_sampler_clean/` and checks the checkpoint,
cached archive hashes, cohort identities and LiDAR grids. It runs on CPU,
without sampling again or exporting duplicate patch TIFFs. The cohorts are
168 Tuktoyaktuk validation patches (zone 13), 1,122 Pond Inlet validation
patches (zone 4) and all 2,111 Cambridge Bay test patches. It writes
`figures/final_figures/{region}_mosaic_2d_demeaned.png` and a JSON provenance
record beside each image. Override these locations with `--cached-study PATH`
and `--figure-dir PATH`.

The cached fields are already patch-wise residuals. Both fields use the valid
LiDAR mask and the existing distance-weighted overlap blending; the prediction
mean is retained. This corrects the historical mosaic's inclusion of predicted
pixels outside the valid LiDAR footprint. GT and predictions use the shared
±0.5 m symmetric-log scale, and errors use a linear ±0.5 m scale. Rotations
remain 90° for Pond Inlet/Tuktoyaktuk and 28° for Cambridge Bay; Tuktoyaktuk
automatically uses the two-strip layout. `--rotation` and `--split-stack` are
available for manual presentation changes.

For fresh inference, omit `--mosaics-only`; for example,
`python scripts/evaluation.py --region pondinlet --sampler plms`.
`--skip-predict` reuses mosaics in that corrected model's output folder.
The paper figure mode preserves the completed studies' manifests. Because
the study runners hash source code, editing the evaluator does not permit
resuming a historical study with the changed source revision.

### Noise schedule and sampler evaluation

The evaluation wrappers require CUDA and default to
`../.venv-py313/bin/python`. Set `PYTHON_BIN` to the absolute path of another
compatible interpreter when needed; this setting applies to evaluation
wrappers, not the training wrapper. For example:

```bash
PYTHON_BIN=/absolute/path/to/python bash scripts/run_noise_sampler_ablation.sh validation
```

Compare corrected original-fold linear/cosine checkpoints with DDPM, DDIM and
PLMS. All use 1,000 steps; DDIM eta is 0 and PLMS order is 4. Initial noise is
paired across configurations using seed 42; DDPM innovations are paired across
schedules for each fixed batch. Review validation results and freeze the
selected configuration before evaluating Cambridge Bay:

```bash
bash scripts/run_noise_sampler_ablation.sh validation
bash scripts/run_noise_sampler_ablation.sh test
```

Results appear in `reconstructions/noise_sampler_clean/{validation,test}/`:
`comparison.csv`, `comparison.json`, `comparison.md` and `per_patch.csv`.
Validation includes 1,122 Pond Inlet and 168 Tuktoyaktuk patches; test includes
all 2,111 Cambridge Bay patches. Pooled summaries weight patches equally;
equal-region averages must be calculated separately. The eight metrics remain
RMSE, signed bias, sigma error, NAE, ZNCC, JSD, log-PSD RMSE and nRMSE/sigma.

Evaluate the alternate fold with its frozen cosine PLMS configuration:

```bash
bash scripts/run_alternate_fold_evaluation.sh test
```

This uses `models/tuk_heldout_clean/tuk_heldout_clean_best.pth` and writes the
same comparison filenames under
`reconstructions/tuk_heldout_clean_evaluation/test/` for all 1,676 Tuktoyaktuk
patches. The wrapper also accepts `validation` for the 1,122 Pond Inlet and
211 Cambridge Bay source-validation patches.

### Conditioning ablation

Run normal, shuffled and zero conditioning with the corrected original-fold
cosine checkpoint:

```bash
bash scripts/run_corrected_conditioning_ablation.sh
```

The wrapper verifies the training-split digest and evaluates all 1,122 Pond
Inlet validation, 168 Tuktoyaktuk validation and 2,111 Cambridge Bay test
patches with 1,000-step PLMS and paired initial noise, seed 42. Shuffling moves
all six images and their metadata together to a spatially separated donor
within the same region and split. Null conditioning zeros both images and
attributes; it is not a separately trained unconditional model.

Outputs under `reconstructions/conditioning_ablation_clean/` include
`summary.csv`, `summary.json`, `per_patch.csv`, `paired_deltas.csv`, `table.tex`
and float32 prediction archives. Use this wrapper for the corrected study;
the older generic preparation defaults refer to the historical checkpoint.

All three evaluation wrappers prepare a study only when its manifest is
absent, verify completed batches and skip them on restart. Interrupted batches
are recomputed. Keep each study's inputs, checkpoint, source revision, batch
settings and recorded GPU/software environment unchanged when resuming.
Retain the complete study directory, including prediction archives and shards.
Use a fresh study directory on a different runtime; do not combine partial
results from incompatible manifests. The Cambridge matrix, alternate-fold
test and conditioning study can run concurrently on separate GPUs, selected
with `CUDA_VISIBLE_DEVICES`, or on separate machines with their own study
directories.

## Contact

Authors: Tessa Cannon, Michel Tsamados, Petru Manescu, Thomas Newman, Christian Haas, Veit Helm, and Weibin Chen.

Please email me tessacannon48@gmail.com if you would like to discuss this work.
