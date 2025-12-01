import glob # type: ignore
import random # type: ignore
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import rasterio
from tqdm import tqdm
import os
import json
import datetime as _dt
import hashlib
from src.data.processing import per_patch_percentile_scale_bandwise_shared


# =============================================================================
# DATASET
# =============================================================================

class LidarS2Dataset(Dataset):
    """
    Returns:
      dict {
        lidar: [1, H, W],      # the data band (RANSAC residuals)
        s2:    [k×4, Hc, Wc],  # k × (R,G,B,NIR)
        attrs: [k×8],          # k × (cloud[1] + sun[3] + view[3] + age_days[1])
        mask:  [H, W],         # validity mask from LiDAR channel 1
        chosen_ids: [k],       # Indices of the S2 patches used for conditioning
      }
    """

    def __init__(self, lidar_dirs, s2_dirs,
             context_k=1, randomize_context=True, augment=True,
             debug=False, target_s2_hw=(256, 256), ref_date="2024-04-26",
             split_pids=None, split="train", s2_p_low: float = 2.0,
                 s2_p_high: float = 98.0,
                 s2_min_range: float = 1e-3):
        super().__init__()
        self.lidar_dirs = lidar_dirs if isinstance(lidar_dirs, list) else [lidar_dirs]
        self.s2_dirs = s2_dirs if isinstance(s2_dirs, list) else [s2_dirs]
        self.augment = augment
        self.target_s2_hw = target_s2_hw
        self.context_k = context_k
        self.randomize_context = randomize_context
        self.max_s2 = 6
        self.ref_date = _dt.date.fromisoformat(str(ref_date)[:10])
        self.split = split
        self.s2_p_low = s2_p_low
        self.s2_p_high = s2_p_high
        self.s2_min_range = s2_min_range

        # Load all patch paths based on the provided IDs
        # Collect lidar patch paths from all lidar dirs
        all_lidar_paths = []
        self.root_refdate = {}
        for lidar_dir in self.lidar_dirs:
            root_abs = os.path.abspath(lidar_dir)
            self.root_refdate[root_abs] = self._resolve_region_ref_date(lidar_dir)
            for p in sorted(glob.glob(os.path.join(lidar_dir, "lidar_patch_*.tif"))):
                all_lidar_paths.append((os.path.abspath(p), root_abs))

        if split_pids is None:
            lidar_paths_to_load = all_lidar_paths
        else:
            lidar_paths_to_load = []
            for pid in split_pids:
                for lidar_dir in self.lidar_dirs:
                    path = os.path.join(lidar_dir, f"lidar_patch_{pid}.tif")
                    if os.path.exists(path):
                        lidar_paths_to_load.append((os.path.abspath(path), os.path.abspath(lidar_dir)))
                        break

        if debug:
            print("DEBUG MODE: Using a subset of 100 samples.")
            lidar_paths_to_load = lidar_paths_to_load[:100]

        # Store only the file paths and pids, not the data itself
        self.samples = []
        for lidar_path, lidar_root in lidar_paths_to_load:
            pid = self._extract_id(lidar_path)
            # Find the corresponding S2 group dir across all roots 
            s2_group_dir = None
            for s2_dir in self.s2_dirs:
                candidate = os.path.join(s2_dir, f"s2_patch_{pid}")
                if os.path.isdir(candidate):
                    s2_group_dir = candidate
                    break
            if s2_group_dir is None:
                continue  # Skip if no match found

            available_ids = [i for i in range(self.max_s2)
                if os.path.exists(os.path.join(s2_group_dir, f"t{i}.tif"))]

            if len(available_ids) >= self.context_k:
                self.samples.append({
                    "lidar_path": lidar_path,
                    "s2_group_dir": s2_group_dir,
                    "tile_id": pid,
                    "available_ids": available_ids,
                    "ref_date": self.root_refdate.get(lidar_root, self.ref_date),
                })

        self.num_samples = len(self.samples)
        print(f"Prepared {self.num_samples} matched LiDAR↔S2 groups (k={self.context_k}).")

    def _extract_id(self, path: str) -> str:
        return os.path.basename(path).split("_")[-1].split(".")[0]

    @staticmethod
    def _encode_angles_deg(az_deg, ze_deg):
        rad = math.pi / 180.0
        az = float(az_deg); ze = float(ze_deg)
        return torch.tensor([math.sin(az * rad), math.cos(az * rad), ze / 90.0], dtype=torch.float32)

    def _days_from_ref(self, date_val, ref_date: _dt.date = None) -> float:
        if date_val is None:
            return 0.0
        try:
            d = _dt.date.fromisoformat(str(date_val)[:10])
            base = ref_date if ref_date is not None else self.ref_date
            return float((d - base).days)
        except Exception:
            return 0.0
    
    def _resolve_region_ref_date(self, lidar_root_path: str) -> _dt.date:
        name = os.path.basename(os.path.normpath(lidar_root_path)).lower()
        if "cambridge" in name:
            return _dt.date(2024, 4, 18)
        if "tuq" in name:
            return _dt.date(2024, 4, 16)
        if "pondinlet" in name:
            return _dt.date(2024, 4, 26)
        return self.ref_date

    def _parse_attrs_json(self, json_path, ref_date: _dt.date):
        if not os.path.exists(json_path):
            return [torch.zeros(8) for _ in range(self.max_s2)]
        with open(json_path, "r") as f:
            recs = json.load(f)
        feats = []
        for r in recs[:self.max_s2]:
            if r is None:
                feats.append(torch.zeros(8))
                continue
            cloud = torch.tensor(float(r.get("cloud_cover", 0.0)) / 100.0).clamp(0, 1)
            saz = self._encode_angles_deg(r.get("sun_azimuth_mean", 0.0), r.get("sun_zenith_mean", 0.0))
            vaz = self._encode_angles_deg(r.get("view_azimuth_mean", 0.0), r.get("view_zenith_mean", 0.0))
            # SCALE AGE (months) rather than raw days; ideally reference LiDAR date per tile
            age = torch.tensor(self._days_from_ref(r.get("acquisition_date"), ref_date=ref_date) / 30.0, dtype=torch.float32).view(1)
            feats.append(torch.cat([cloud.view(1), saz, vaz, age], dim=0))  # [8]
        while len(feats) < self.max_s2:
            feats.append(torch.zeros(8))
        return feats  # list of length max_s2, each [8]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        sample_paths = self.samples[idx]
        lidar_path   = sample_paths["lidar_path"]
        s2_group_dir = sample_paths["s2_group_dir"]
        tile_id      = sample_paths["tile_id"]
        available_ids = sample_paths.get("available_ids", list(range(self.max_s2)))

        # Choose which S2 times to use (define chosen_ids FIRST)
        pool = sorted(available_ids)
        if len(pool) < self.context_k:
            raise RuntimeError(
                f"Patch {tile_id}: only {len(pool)} S2 times available, "
                f"but context_k={self.context_k}."
            )

        if self.randomize_context:
            if self.split == "train":
                chosen_ids = random.sample(pool, self.context_k)
            else:
                # deterministic selection for val/test
                seed = int(hashlib.sha1(tile_id.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
                rng = random.Random(seed)
                chosen_ids = rng.sample(pool, self.context_k)
        else:
            # take the first k available deterministically
            chosen_ids = pool[:self.context_k]

        # Load LiDAR (data + mask)
        with rasterio.open(lidar_path) as src:
            lidar_full = torch.from_numpy(src.read().astype(np.float32))
        if lidar_full.shape[0] == 1:
            data = lidar_full[0:1]
            mask = torch.ones_like(data[0])
        else:
            data = lidar_full[0:1]
            mask = lidar_full[1]

        # Read only the chosen S2 times
        s2_list = []
        for i in chosen_ids:
            s2_path = os.path.join(s2_group_dir, f"t{i}.tif")
            with rasterio.open(s2_path) as src:
                arr = torch.from_numpy(src.read()[:4].astype(np.float32))  # R,G,B,NIR
            if arr.shape[-2:] != self.target_s2_hw:
                arr = F.interpolate(arr.unsqueeze(0), size=self.target_s2_hw,
                                    mode="bilinear", align_corners=False).squeeze(0)
            s2_list.append(arr)
        s2 = torch.cat(s2_list, dim=0)   # [k*4, Hc, Wc]

        # Attributes (only for chosen times)
        sample_ref_date = sample_paths["ref_date"]
        all_attrs = self._parse_attrs_json(os.path.join(s2_group_dir, "attrs.json"), ref_date=sample_ref_date)
        attrs = torch.cat([all_attrs[i] for i in chosen_ids], dim=0)  # [k*8]

        # Normalize S2 per-patch, bandwise, shared across times -> [0,1]
        s2 = per_patch_percentile_scale_bandwise_shared(s2, p_low=self.s2_p_low, p_high=self.s2_p_high, min_range=self.s2_min_range, clamp01=True)

        # Data augmentation (train only)
        if self.augment and self.split == "train":
            if random.random() > 0.5:
                data = TF.hflip(data); s2 = TF.hflip(s2); mask = TF.hflip(mask)
            if random.random() > 0.5:
                data = TF.vflip(data); s2 = TF.vflip(s2); mask = TF.vflip(mask)
            if random.random() > 0.5:
                ang = random.choice([90, 180, 270])
                data = TF.rotate(data, ang)
                s2   = TF.rotate(s2, ang)
                mask = TF.rotate(mask.unsqueeze(0), ang).squeeze(0)

        return {
            "s2": s2.float(),
            "lidar": data.float(),
            "mask": mask.float(),
            "attrs": attrs.float(),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "tile_id": tile_id
        }



