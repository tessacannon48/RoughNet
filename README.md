
# Mapping Arctic Ice Terrain Using Diffusion-Based Super-Resolution of Satellite Imagery
![alt text](https://github.com/tessacannon48/RoughNet/blob/main/figures/final_figures/model_diagram_temp.png)

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
![alt text]()

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
- The dataset class used to create the input dataset applies several selections and transformations to adequately prepare the data for modeling. First, each Sentinel-2 patch is resized to the dimensions of the LiDAR patch (256×256 pixels) using bilinear interpolation. Sentinel-2 data is normalized on a per-patch basis: for each band (R, G, B, NIR), all Sentinel-2 products are pooled, clipped to the 2nd--98th percentiles, and linearly rescaled to [0,1] with a minimum range safeguard of \(1\times10^{-3}\). The LiDAR input consists of RANSAC residual elevations, which are further processed using patch-wise mean removal computed only over valid pixels. Specifically, the mean residual elevation of each patch is subtracted from all valid pixels so that the model learns local roughness structure rather than absolute elevation offsets, while invalid pixels are masked to zero. The training set is then randomly augmented to increase sample diversity and improve model robustness. Finally, the attributes are parsed for each Sentinel-2 patch and encoded as follows: cloud coverage percentage is scaled to [0,1], image age is represented as a signed scalar corresponding to the temporal difference between Sentinel-2 and LiDAR acquisition dates (scaled approximately in months), Zenith angles are scaled to [0,1], and Azimuth angles are transformed into sine and cosine components using sinusoidal encoding. The difference in angle encoding arises because Zenith angles lie in a range of \(0^\circ\) to \(90^\circ\), representing the vertical angle between an object and a line pointing upward from the ground, whereas Azimuth angles lie in a range of \(0^\circ\) to \(360^\circ\), representing the horizontal angle measured clockwise from True North.

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

1. Edit config.yaml: 
   - Specify directories to LiDAR and Sentinel-2
   - Set default training parameters
   - Set WANDB login
2. Run main.py, specifying optional config changes in terminal:
```bash
python Dissertation/scripts/main.py --context_k 1 --attention_variant mid --sampling_methods plms --lr 1e-4 --epochs 200 --unet_depth 4 --noise_schedule cosine --base_channels 64 --loss_name masked_hybrid_mse_loss --loss_alpha 1.0 --evaluate --run_name final_improved_baseline
```
- Outputs trained model, reconstruction figure, and patch-wise reconstruction statistics

### Evaluation

The `evaluation.py` script performs **region-wide inference**, **mosaicking**, **figure generation** (demeaned 2D mosaics and PDFs), and **metric computation** (per-patch and region-level).

Each evaluation run is configured by selecting a **region preset** using the `--region` argument.  
All file paths (Sentinel-2 patches, LiDAR patches, model checkpoint, output directory) and any region filtering (e.g., `zone_ids=[4]` for Pond Inlet) are handled automatically through the preset system.

Before running: At the top of `evaluation.py`, inside the `get_region_preset()` function:

- **Update the `root` directory path** if needed  
- **Update the model checkpoint path** (`ckpt_path`)  
- **Optionally adjust region-specific dataset paths** if your directory structure changes  

Example runs: 

#### Validation region 1 – Pond Inlet, zone_ids=[4]
```bash
python evaluation.py --region pondinlet
```
#### Validation region 2 – Tuktoyaktuk, zone_ids=[13]
```bash
python evaluation.py --region tuk
```
#### Test region – Cambridge Bay, all tiles
```bash
python evaluation.py --region cambridge
```
#### Reuse existing mosaics for Pond Inlet
```bash
python evaluation.py --region pondinlet --skip-predict
```

## Contact

Authors: Tessa Cannon, Michel Tsamados, Petru Manescu, Thomas Newman, Christian Haas, and Veit Helm
Please email me tessacannon48@gmail.com if you would like to discuss this work.
