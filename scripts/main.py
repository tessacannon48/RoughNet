# main.py

# Import packages
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import wandb
import os, sys
import json
from torch.utils.data import Subset
import time
import yaml
import random
import warnings
import glob
import rasterio
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.argparse import parse_arguments
from src.data.dataset import LidarS2Dataset
from src.data.processing import compute_s2_mean_std_multi
from src.model.unet import ConditionalUNet
from src.diffusion.scheduler import LinearDiffusionScheduler, CosineDiffusionScheduler
from src.diffusion.sampling import p_sample_loop_ddpm, p_sample_loop_ddim, p_sample_loop_plms
from src.utils.metrics import compute_topographic_rmse, normalize_batch, masked_mse_loss, masked_mae_loss, masked_hybrid_mse_loss, masked_hybrid_mae_loss
from src.utils.recon_metrics import (
    rmse as rmse_recon,
    bias as bias_recon,
    sigma_error,
    corr_length_error,
    normal_angle_error,
    average_jsd_multiscale,
    log_psd_rmse,
    agg
)

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(config):
    """Unified training function with early stopping."""
    
    # Set device
    device = torch.device(config['system']['device'])

    # Set save directory
    save_dir = config["logging"]["save_dir"]

    # Set noise scheduler
    if config["training"]["noise_schedule"] == "linear":
        scheduler = LinearDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)
    else:
        scheduler = CosineDiffusionScheduler(timesteps=config["training"]["timesteps"], device=device)

    # Initialize wandb with run name
    if not config["logging"]["wandb_name"]:
        attention_flag = "att" if config["model"]["attention_variant"] != "none" else "noatt"
        debug_suffix = "debug" if config["system"]["debug"] else ""
        wandb_name = f"{config['logging']['run_name']}_k{config['training']['context_k']}_{attention_flag}{f'_{debug_suffix}' if debug_suffix else ''}"
        config["logging"]["wandb_name"] = wandb_name
    
    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_name"],
        config=config,
    )

    # Map loss function names to their corresponding functions
    loss_functions = {
        'masked_mse_loss': masked_mse_loss,
        'masked_mae_loss': masked_mae_loss,
        'masked_hybrid_mse_loss': masked_hybrid_mse_loss,
        'masked_hybrid_mae_loss': masked_hybrid_mae_loss
    }
    
    # Get loss function and alpha from config
    loss_name = config['training']['loss']['name']
    loss_alpha = config['training']['loss']['alpha']
    
    # Select the loss function
    criterion = loss_functions.get(loss_name)

    # Specify training location names from config
    train_locations = config["data"].get("train_locations", [])
    location_tag = "-".join(train_locations) if train_locations else "all"

    # Load all patch IDs and their regions from multiple S2 dirs
    all_patch_ids = []
    for s2_dir in config["data"]["s2_dirs"]:
        patch_ids = [os.path.basename(p).split('_')[-1].split('.')[0] 
                    for p in glob.glob(os.path.join(s2_dir, "s2_patch_*")) if os.path.isdir(p)]
        all_patch_ids.extend(patch_ids)
    train_pids = []
    val_pids = []
    
    # Get regions to use as validation from config
    validation_regions = []
    for region in config["data"]["validation_regions"]:
        validation_regions.append(region)
    
    print("Separating patches by region...")
    for pid in tqdm(all_patch_ids):
        for s2_dir in config["data"]["s2_dirs"]:
            region_path = os.path.join(s2_dir, f"s2_patch_{pid}", "region.json")
            if os.path.exists(region_path):
                with open(region_path, 'r') as f:
                    region_data = json.load(f)
                region_id = region_data.get("region_id", -1)

                if region_id in validation_regions: # Specify validation regions
                    val_pids.append(pid)
                else:
                    train_pids.append(pid)
                break
        # break early when in debug mode
        if config["system"]["debug"] and len(train_pids) >= 100 and len(val_pids) >= 8:
            break
            
    print(f"Number of training patches: {len(train_pids)}")
    print(f"Number of validation patches (Regions {', '.join(map(str, validation_regions))}): {len(val_pids)}")
    
    # Create a list of S2 patch directories for the training set only
    train_s2_dirs = []
    for s2_dir in config["data"]["s2_dirs"]:
        for pid in train_pids:
            patch_path = os.path.join(s2_dir, f"s2_patch_{pid}")
            if os.path.isdir(patch_path):
                train_s2_dirs.append(patch_path)

    # Create training and validation datasets using the pre-defined patch IDs
    train_dataset = LidarS2Dataset(
        lidar_dirs=config["data"]["lidar_dirs"],
        s2_dirs=config["data"]["s2_dirs"],
        #s2_means=s2_means,
        #s2_stds=s2_stds,
        context_k=config["training"]["context_k"],
        randomize_context=config["training"]["randomize_context"],
        augment=True,
        debug=config["system"]["debug"],
        split_pids=train_pids,
        split="train"
    )

    val_dataset = LidarS2Dataset(
        lidar_dirs=config["data"]["lidar_dirs"],
        s2_dirs=config["data"]["s2_dirs"],
        #s2_means=s2_means,
        #s2_stds=s2_stds,
        context_k=config["training"]["context_k"],
        randomize_context=config["training"]["randomize_context"],
        augment=False, # No augmentation for validation set
        debug=config["system"]["debug"],
        split_pids=val_pids,
        split="val"
    )
    
    # Set dataset split tags 
    train_dataset.split = "train"
    val_dataset.split = "val"

    # Empty cache
    torch.cuda.empty_cache()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"] // 2 if config["training"]["num_workers"] > 1 else 1
    )

    # Initialize model
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=4 * config["training"]["context_k"],
        attr_dim=8 * config["training"]["context_k"],
        base_channels=config["model"]["base_channels"],
        embed_dim=config["model"]["embed_dim"],
        unet_depth=config["model"]["unet_depth"],
        attention_variant=config["model"]["attention_variant"],
        cond_k=config["training"]["context_k"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    train_losses = []
    
    # Early Stopping Variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 200 
    max_epochs = config["training"]["epochs"]

    # Create models directory
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)

    # Initialize a list to store epoch durations
    epoch_durations = []

    # Training loop
    for epoch in range(max_epochs): 
        
        model.train()
        epoch_start_time = time.perf_counter()
        epoch_loss = 0

        # Training metrics
        total_train_loss = 0
        total_train_pixel_loss = 0
        total_train_gradient_loss = 0

        # Training step
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            lidar = batch["lidar"].to(device)
            s2    = batch["s2"].to(device)
            attrs = batch["attrs"].to(device)
            mask  = batch["mask"].to(device)
            t = torch.randint(0, config["training"]["timesteps"], (lidar.size(0),), device=device).long()
            
            # Training step
            noisy = scheduler.q_sample(lidar, t)
            pred = model(noisy, s2, attrs, t)
            
            # Select loss function with or without alpha
            if 'hybrid' in loss_name:
                loss, pixel_loss_component, gradient_loss_component = criterion(pred, lidar, mask, alpha=loss_alpha)
                total_train_pixel_loss += pixel_loss_component.item()
                total_train_gradient_loss += gradient_loss_component.item()
            else:
                loss = criterion(pred, lidar, mask)
                # If not a hybrid loss, the other components are zero
                pixel_loss_component = loss
                gradient_loss_component = torch.tensor(0.0)

            total_train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation step
        model.eval()
        total_val_loss = 0
        total_val_pixel_loss = 0
        total_val_gradient_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                lidar = batch["lidar"].to(device)
                s2    = batch["s2"].to(device)
                attrs = batch["attrs"].to(device)
                mask  = batch["mask"].to(device)
                t = torch.randint(0, config["training"]["timesteps"], (lidar.size(0),), device=device).long()

                # Evaluation
                noisy = scheduler.q_sample(lidar, t)
                pred  = model(noisy, s2, attrs, t)
                
                # Select loss function with or without alpha
                if 'hybrid' in loss_name:
                    batch_val_loss, pixel_val_component, gradient_val_component = criterion(pred, lidar, mask, alpha=loss_alpha)
                    total_val_pixel_loss += pixel_val_component.item()
                    total_val_gradient_loss += gradient_val_component.item()
                else:
                    batch_val_loss = criterion(pred, lidar, mask)
                    pixel_val_component = batch_val_loss
                    gradient_val_component = torch.tensor(0.0)
                
                total_val_loss += batch_val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # End of epoch timing
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)

        # Log metrics to wandb
        log_metrics = {
            f"train_{loss_name}": avg_loss,
            f"val_{loss_name}": avg_val_loss,
            "epoch": epoch
        }
        if 'hybrid' in loss_name:
            log_metrics.update({
                f"train_{'mse' if 'mse' in loss_name else 'mae'}": total_train_pixel_loss / len(train_loader),
                f"train_gradient": total_train_gradient_loss / len(train_loader),
                f"val_{'mse' if 'mse' in loss_name else 'mae'}": total_val_pixel_loss / len(val_loader),
                f"val_gradient": total_val_gradient_loss / len(val_loader)
            })
        wandb.log(log_metrics)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            best_path = os.path.join(save_dir, f"{wandb_name}_best.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    average_time = sum(epoch_durations) / len(epoch_durations)
    if wandb.run:
        wandb.log({"average_training_time_sec_per_epoch": average_time})
    print(f"Average training time per epoch: {average_time:.2f} seconds")
    
    # Final model path for evaluation
    final_model_path = os.path.join(save_dir, f"{wandb_name}_best.pth")

    # Run reconstruction evaluation 
    if config["evaluation"]["evaluate"]:
        print(f"Loading best model from {final_model_path} for evaluation...")
        best_checkpoint = torch.load(final_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Best model loaded (epoch {best_checkpoint['epoch']}, val_loss: {best_checkpoint['val_loss']:.4f})")
        
        run_reconstruction_evaluation(model, val_dataset, config, scheduler)

    return {model.__class__.__name__: {"losses": train_losses, "model_path": final_model_path}}

# =============================================================================
# RECONSTRUCTION EVALUATION
# =============================================================================

def run_reconstruction_evaluation(model, val_dataset, config, scheduler=None):
    """
    Run reconstruction evaluation with S2 selection and error map visualization, including comprehensive metric logging and statistical data saving.
    """
    print("\n" + "="*60)
    print("RUNNING RECONSTRUCTION EVALUATION")
    print("="*60)

    # Set split for augmentation control
    val_dataset.split = "val"

    # Define output directories
    output_dir = config["logging"]["output_dir"]
    stats_dir = os.path.join(output_dir, "reconstruction_statistics")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Set model to eval
    model.eval()
    
    # Get hand-picked validation set from config
    eval_pids = config["data"]["evaluation_patch_ids"]

    # Find the indices of these patches in the validation dataset
    eval_indices = [i for i, sample in enumerate(val_dataset.samples) if sample['tile_id'] in eval_pids]

    # Handle the case where not all patches are found
    if len(eval_indices) != len(eval_pids):
        print(f"Warning: Found {len(eval_indices)} out of 8 requested evaluation patches. Continuing with found patches.")
    
    # Create a Subset of the validation dataset using the found indices
    eval_subset = Subset(val_dataset, eval_indices)

    # Create eval loader
    eval_loader = DataLoader(eval_subset, batch_size=len(eval_subset), shuffle=False)
    
    # Access the single batch from the evaluation loader
    batch = next(iter(eval_loader))
    s2 = batch["s2"].to(config["system"]["device"])
    lidar = batch["lidar"].to(config["system"]["device"])
    attrs = batch["attrs"].to(config["system"]["device"])
    mask = batch["mask"].to(config["system"]["device"])
    chosen_ids_batch = batch["chosen_ids"]
    tile_ids_batch = batch["tile_id"] 

    # Get batch dimensions and context information
    B = lidar.size(0)
    context_k = config["training"]["context_k"]
    run_name = config["logging"]["run_name"]

    # Get sampling methods
    all_samplers = {
        "ddpm": lambda m, s, c, a, d: p_sample_loop_ddpm(m, scheduler, s, c, a, d) if scheduler else None,
        "ddim": lambda m, s, c, a, d: p_sample_loop_ddim(m, scheduler, s, c, a, d) if scheduler else None,
        "plms": lambda m, s, c, a, d: p_sample_loop_plms(m, scheduler, s, c, a, d) if scheduler else None,
    }
    requested_methods = config["evaluation"]["sampling_methods"]
    p_samplers = {m: all_samplers[m] for m in requested_methods if m in all_samplers}
    if not p_samplers:
        print("No valid samplers available")
        return

    # Determine which sentinel-2 patches were used to condition the model
    used_patch_ids = chosen_ids_batch

    # Iterate over each sampling method
    for sampler_name, sampler_func in p_samplers.items():
        print(f"\nSampling method: {sampler_name}")

        # Sample from the model
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            pred = sampler_func(model, lidar.shape, s2, attrs, config["system"]["device"])
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            sampling_time = end_time - start_time

            # Log sampling time
            if wandb.run:
                wandb.log({f"{sampler_name}_sampling_time_sec": sampling_time})

            # Move tensors to CPU
            gt = lidar.cpu()
            pred = pred.cpu()
            m = mask.cpu() if mask is not None else None

            # Get metric parameters from config
            px = float(config["data"].get("pixel_size_m", 1.0))
            jsd_scales = tuple(config["evaluation"].get("jsd_scales_m", [1.0, 2.0, 5.0, 10.0]))
            jsd_bins   = int(config["evaluation"].get("jsd_bins", 256))
            use_sobel  = bool(config["evaluation"].get("nae_use_sobel", True))
            deg        = bool(config["evaluation"].get("nae_degrees", True))
            use_window = bool(config["evaluation"].get("psd_window", True))

            # Calculate and log core reconstruction metrics for the entire batch
            rmse_tile, bias_tile, sigma_tile_pct = [], [], []
            clen_tile_pct, nae_tile, jsd_tile, psd_tile = [], [], [], []
            valid_counts = []
            for i in range(B):
                # Get data for each tile
                gi, pi = gt[i], pred[i]

                # Get mask
                mm = m[i] if m is not None else None
                valid_counts.append(int(mm.sum()))
                mm = mm.bool() if mm is not None else None
                
                # Calculate metrics
                rmse_tile.append(rmse_recon(gi, pi, mask=mm).item())
                bias_tile.append(bias_recon(gi, pi, mask=mm).item())
                sigma_tile_pct.append(sigma_error(gi, pi, mask=mm).item())
                clen_tile_pct.append(corr_length_error(gi, pi, mask=mm, pixel_size=px).item())
                nae_tile.append(normal_angle_error(gi, pi, mask=mm, pixel_size=px, use_sobel=use_sobel, degrees=deg).item())
                jsd_tile.append(average_jsd_multiscale(gi, pi, scales_m=jsd_scales, pixel_size=px, bins=jsd_bins, mask=mm).item())
                psd_tile.append(log_psd_rmse(gi, pi, pixel_size=px, mask=mm, window=use_window).item())
                
                # Save reconstruction statistics for each sample
                tile_id = tile_ids_batch[i]
                reco_stats = {
                    "tile_id": tile_id,
                    "rmse_phys_m": rmse_tile[i],
                    "bias_phys_m": bias_tile[i],
                    "sigma_error_pct": sigma_tile_pct[i],
                    "corr_length_error_pct": clen_tile_pct[i],
                    "normal_angle_error_deg": nae_tile[i],
                    "jsd": jsd_tile[i],
                    "psd_rmse": psd_tile[i],
                    "valid_pixel_count": valid_counts[i],
                    "gt_min_val": float(gt[i].min()),
                    "gt_max_val": float(gt[i].max()),
                    "gt_mean_val": float(gt[i].mean()),
                    "gt_std_val": float(gt[i].std()),
                    "pred_min_val": float(pred[i].min()),
                    "pred_max_val": float(pred[i].max()),
                    "pred_mean_val": float(pred[i].mean()),
                    "pred_std_val": float(pred[i].std()),
                    "gt_mode_val": float(torch.mode(gt[i].flatten().to(torch.float32))[0]),
                    "pred_mode_val": float(torch.mode(pred[i].flatten().to(torch.float32))[0]),
                }
                
                debug_flag = "_debug_" if config["system"]["debug"] else ""
                stats_path = os.path.join(stats_dir, f"{tile_id}{debug_flag}{run_name}_{sampler_name}_stats.json")
                with open(stats_path, "w") as f:
                    json.dump(reco_stats, f, indent=4)
                print(f"Saved stats for tile {tile_id} to {stats_path}")

            # Calculate averages for logging
            rmse_mean   = agg(rmse_tile, weights=valid_counts, reduce="weighted")  # micro-average
            bias_mean   = agg(bias_tile, weights=valid_counts, reduce="weighted")
            sigma_mean  = agg(sigma_tile_pct, reduce="mean")     # macro-average
            sigma_med   = agg(sigma_tile_pct, reduce="median")   # robustness
            ell_mean    = agg(clen_tile_pct, reduce="mean")
            ell_med     = agg(clen_tile_pct, reduce="median")
            nae_mean    = agg(nae_tile, weights=valid_counts, reduce="weighted")
            jsd_mean    = agg(jsd_tile, reduce="mean")
            psd_mean    = agg(psd_tile, reduce="mean")
            
            # Log metrics
            if wandb.run:
                wandb.log({
                    f"{sampler_name}_rmse": rmse_mean,
                    f"{sampler_name}_bias": bias_mean,
                    f"{sampler_name}_sigma_error_pct": sigma_mean,
                    f"{sampler_name}_sigma_error_med_pct": sigma_med,
                    f"{sampler_name}_corr_length_error_pct": ell_mean,
                    f"{sampler_name}_corr_length_error_med_pct": ell_med,
                    f"{sampler_name}_normal_angle_error_deg": nae_mean,
                    f"{sampler_name}_jsd": jsd_mean,
                    f"{sampler_name}_psd_rmse": psd_mean,
                })

            # Determine the total number of rows (S2 patches + GT + Pred + Error)
            num_s2_patches = context_k # Assuming context_k is the number of S2 patches
            num_data_rows = 3 # GT, Pred, Error
            total_rows = num_s2_patches + num_data_rows 
            num_cols = B # Number of examples

            # Calculate figure size (adjust these constants based on desired output size)
            tile_size_inches = 4
            # fig_w: columns * size. We need extra width for the left-side text labels.
            fig_w = num_cols * tile_size_inches + 2 
            fig_h = total_rows * tile_size_inches + 1 # Add space for footer/labels
            
            # Create a figure and grid of subplots
            fig, axes = plt.subplots(total_rows, num_cols, 
                                     figsize=(fig_w, fig_h), 
                                     squeeze=False, # Ensure axes is 2D array even for 1xN
                                     gridspec_kw={'hspace': 0.05, 'wspace': 0.05}) 
            
            # Determine global elevation and error ranges for consistent colormapping
            global_elev_min = torch.min(gt.min(), pred.min()).item()
            global_elev_max = torch.max(gt.max(), pred.max()).item()
            signed_error = gt.squeeze(1) - pred.squeeze(1) 
            max_abs_error = signed_error.abs().max().item()

            # Process S2 patches for visualization 
            tileid_to_s2dir = {s['tile_id']: s['s2_group_dir'] for s in val_dataset.samples}
            s2_viz_data = [] 
            for i in range(B):
                tile_id = tile_ids_batch[i]
                s2_group_dir = tileid_to_s2dir.get(tile_id)
                if s2_group_dir is None:
                    print(f"Warning: no s2_group_dir found for tile {tile_id}; skipping S2 viz.")
                    s2_viz_data.append([np.zeros((*gt.shape[-2:], 3), dtype=np.float32) for _ in range(context_k)])
                    continue
                chosen_ids = chosen_ids_batch[i].tolist()
                
                processed_sample = []
                for t_id in chosen_ids:
                    s2_path = os.path.join(s2_group_dir, f"t{t_id}.tif")
                    with rasterio.open(s2_path) as src:
                        arr = torch.from_numpy(src.read()[:4].astype(np.float32))
                    
                    rgb = arr[[0, 1, 2], :, :]
                    rgb = normalize_batch(rgb.unsqueeze(0)).squeeze(0)
                    rgb = F.interpolate(rgb.unsqueeze(0), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
                    processed_sample.append(rgb.permute(1, 2, 0).numpy()) # Convert to HxWx3 NumPy
                s2_viz_data.append(processed_sample)

            # Set row titles
            row_titles = [f"S2 (t{t})" for t in chosen_ids_batch[0].tolist()] + ["GT LiDAR", "Pred LiDAR", "Error"]
        
            # Iterate over examples (columns)
            for col in range(num_cols): 
                # Data for the current example 
                gt_i = gt[col].squeeze().numpy()
                pred_i = pred[col].squeeze().numpy()
                err_i = (gt[col].squeeze() - pred[col].squeeze()).numpy()
                
                # S2 Patches 
                for row in range(num_s2_patches):
                    ax = axes[row, col]
                    s2_np = s2_viz_data[col][row]
                    
                    ax.imshow(s2_np) 
                    ax.axis("off")

                # Lidar GT
                row_gt = num_s2_patches
                ax = axes[row_gt, col]
                im_gt = ax.imshow(gt_i, cmap='terrain', vmin=global_elev_min, vmax=global_elev_max)
                ax.axis("off")
               

                # Lidar Pred
                row_pred = num_s2_patches + 1
                ax = axes[row_pred, col]
                im_pred = ax.imshow(pred_i, cmap='terrain', vmin=global_elev_min, vmax=global_elev_max)
                ax.axis("off")

                # Error Map
                row_err = num_s2_patches + 2
                ax = axes[row_err, col]
                # Define the linear scale around zero 
                linthresh_val = 0.5
                # Create a symmetric logarithmic normalizer
                norm_error = mcolors.SymLogNorm(
                    linthresh=linthresh_val, 
                    linscale=1.0, # Linear scale for linthresh region
                    vmin=-max_abs_error, 
                    vmax=max_abs_error
                )
                im_err = ax.imshow(err_i, cmap='seismic',norm=norm_error)
                ax.axis("off")
            
            # Set row titles
            for row in range(total_rows):
                ax = axes[row, 0] 
                ax.text(-0.1, 0.5, row_titles[row], 
                        ha="right", va="center", 
                        transform=ax.transAxes, 
                        fontsize=20, 
                        fontweight='bold', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.3'))

            # Set footer
            footer = (f"RMSE: {rmse_mean:.2f} m  |  Bias: {bias_mean:.2f} m  |  Δσ: {sigma_mean:.2f} %  |  Δℓ: {ell_mean:.2f} %  |  NAE: {nae_mean:.2f}°  |  JSD: {jsd_mean:.2f}  |  PSD: {psd_mean:.2f}")
            fig.text(0.5, 0.01, footer, ha='center', fontsize=23, weight='bold', color='darkblue')

            # Identify the axes in the last column to anchor color bars
            ax_gt_anchor = axes[row_gt, num_cols - 1]
            ax_err_anchor = axes[row_err, num_cols - 1]

            # GT / Pred Colorbar 
            axins_gt = inset_axes(ax_gt_anchor, 
                                  width="5%", 
                                  height="205%", 
                                  loc='right', 
                                  bbox_to_anchor=(0.1, -0.52, 1, 1),
                                  bbox_transform=ax_gt_anchor.transAxes)
            fig.colorbar(im_gt, cax=axins_gt)

            # Error Colorbar
            axins_err = inset_axes(ax_err_anchor, 
                                   width="5%", 
                                   height="100%", 
                                   loc='right', 
                                   bbox_to_anchor=(0.1, 0, 1, 1), 
                                   bbox_transform=ax_err_anchor.transAxes)
            fig.colorbar(im_err, cax=axins_err)

            # Adjust layout
            plt.subplots_adjust(left=0.08, bottom=0.08, top=0.98, right=0.95)
            
            # Save the figure
            out_name = config["logging"]["wandb_name"] or config["logging"]["run_name"]
            out_path = os.path.join(config["logging"]["output_dir"], f"{out_name}_{sampler_name}_vis.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight') 
            plt.close()

            print(f"Saved visualization to {out_path}")
            
# =============================================================================
# SET GLOBAL SEED
# =============================================================================

def set_seed(seed):
    """Sets the seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Set global seed
    seed = 42
    set_seed(seed)

    # Parse command line arguments
    args = parse_arguments()
    
    # Load the base configuration from the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override YAML values with command-line arguments if provided
    if args.s2_dir: config['data']['s2_dirs'] = [args.s2_dir]
    if args.lidar_dir: config['data']['lidar_dirs'] = [args.lidar_dir]
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.lr: config['training']['lr'] = args.lr
    if args.timesteps: config['training']['timesteps'] = args.timesteps
    if args.noise_schedule: config['training']['noise_schedule'] = args.noise_schedule
    if args.num_workers: config['training']['num_workers'] = args.num_workers
    if args.context_k: config['training']['context_k'] = args.context_k
    if args.randomize_context: config['training']['randomize_context'] = True
    if args.wandb_project: config['logging']['wandb_project'] = args.wandb_project
    if args.wandb_name: config['logging']['wandb_name'] = args.wandb_name
    if args.run_name: config['logging']['run_name'] = args.run_name
    if args.save_dir: config['logging']['save_dir'] = args.save_dir
    if args.output_dir: config['logging']['output_dir'] = args.output_dir
    if args.sampling_methods: config['evaluation']['sampling_methods'] = args.sampling_methods
    if args.evaluate: config['evaluation']['evaluate'] = True
    if args.eval_index_json: config['evaluation']['eval_index_json'] = args.eval_index_json
    if args.device: config['system']['device'] = args.device
    if args.debug: config['system']['debug'] = True
    if args.unet_depth: config['model']['unet_depth'] = args.unet_depth
    if args.base_channels: config['model']['base_channels'] = args.base_channels
    if args.embed_dim: config['model']['embed_dim'] = args.embed_dim
    if args.attention_variant: config['model']['attention_variant'] = args.attention_variant
    if args.loss_name: config['training']['loss']['name'] = args.loss_name
    if args.loss_alpha is not None: config['training']['loss']['alpha'] = args.loss_alpha

    # Auto-detect device if not specified
    device = config['system']['device']
    if device == 'auto':
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA device(s)")
            device = "cuda" if device_count > 0 else "cpu"
            print(f"Using device: {device}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    config['system']['device'] = device
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print(f"Debug Mode: {config['system']['debug']}")
    print("="*50)
    print(f"Model Type: Standard U-Net")
    print(f"Data Paths:")
    # Handle multiple directories and region names
    train_regions = config['data'].get('train_regions', [])
    s2_dirs = config['data'].get('s2_dirs', [])
    lidar_dirs = config['data'].get('lidar_dirs', [])
    print(f"  Training Regions: {', '.join(train_regions)}")
    print("  Sentinel-2 Directories:")
    for d in s2_dirs:
        print(f"    - {d}")
    print("  LiDAR Directories:")
    for d in lidar_dirs:
        print(f"    - {d}")
    print(f"Training Parameters:")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning Rate: {config['training']['lr']}")
    print(f"  Timesteps: {config['training']['timesteps']}")
    print(f"  Noise Schedule: {config['training']['noise_schedule']}")
    print(f" Loss Function: {config['training']['loss']['name']} (alpha={config['training']['loss'].get('alpha', 'N/A')})")
    print(f"Model Parameters:")
    print(f"  Base Channels: {config['model']['base_channels']}")
    print(f"  Embed Dim: {config['model']['embed_dim']}")
    print(f"Context k: {config['training']['context_k']}")
    print(f"Randomize Context: {config['training']['randomize_context']}")
    print(f"Attention Variant: {config['model']['attention_variant']}")
    print(f"Eval Index File: {config['evaluation']['eval_index_json']}")
    print(f"System:")
    print(f"  Device: {config['system']['device']}")
    print(f"  Num Workers: {config['training']['num_workers']}")
    print(f"Logging:")
    print(f"  W&B Project: {config['logging']['wandb_project']}")
    print(f"  W&B Run Name: {config['logging']['wandb_name'] or config['logging']['run_name']}")
    print(f"  Run Label: {config['logging']['run_name']}")
    print(f"  Save Directory: {config['logging']['save_dir']}")
    print(f"  Output Directory: {config['logging']['output_dir']}")
    if config['evaluation']['evaluate']:
        print(f"Evaluation:")
        print(f"  Sampling Methods: {', '.join(config['evaluation']['sampling_methods'])}")
    print("="*50)
    
    # Empty cache
    torch.cuda.empty_cache()
    print("\nStarting training...")
    
    # Train model
    results = train_model(config)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {results[list(results.keys())[0]]['model_path']}")