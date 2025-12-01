import argparse # type: ignore

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    """
    Parses command line arguments, including the path to a config file and
    optional overrides for key hyperparameters.
    """
    parser = argparse.ArgumentParser(description='Train diffusion model for LiDAR generation')

    # Core argument: path to the config file
    parser.add_argument('--config', type=str, default='/cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation/config.yaml',
                        help='Path to the YAML configuration file')

    # Data paths
    parser.add_argument('--s2_dir', type=str, help='Override for S2 patches directory')
    parser.add_argument('--lidar_dir', type=str, help='Override for LiDAR patches directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Override for batch size')
    parser.add_argument('--epochs', type=int, help='Override for number of epochs to train')
    parser.add_argument('--lr', type=float, help='Override for learning rate')
    parser.add_argument('--timesteps', type=int, help='Override for diffusion timesteps')
    parser.add_argument("--noise_schedule", type=str, choices=["linear","cosine"], help='Override for noise schedule')
    
    # Model parameters
    parser.add_argument("--unet_depth", type=int, help='Override for UNet depth')
    parser.add_argument('--base_channels', type=int, help='Override for base channels')
    parser.add_argument('--embed_dim', type=int, help='Override for embedding dimension')
    parser.add_argument('--attention_variant', type=str, choices=['default', 'none', 'mid', 'heavy', 'all'], 
                        help='Override for UNet attention config')

    # Training options
    parser.add_argument('--num_workers', type=int, help='Override for number of data loader workers')
    parser.add_argument('--context_k', type=int, help='Override for number of S2 patches to condition on')
    parser.add_argument('--randomize_context', action='store_true', help='Override to randomly sample S2 context during training')

    # Loss function options
    parser.add_argument('--loss_name', type=str, choices=['masked_mse_loss', 'masked_mae_loss', 'masked_hybrid_mse_loss', 'masked_hybrid_mae_loss'],
                        help='Override for the loss function name')
    parser.add_argument('--loss_alpha', type=float, help='Override for the alpha weighting of hybrid losses')
    
    # Logging and saving
    parser.add_argument('--wandb_project', type=str, help='Override for Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, help='Override for Weights & Biases run name')
    parser.add_argument('--run_name', type=str, help='Override for run name')
    parser.add_argument('--save_dir', type=str, help='Override for directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, help='Override for directory to save reconstruction images')
    
    # Sampling and evaluation
    parser.add_argument('--sampling_methods', type=str, nargs='+', choices=['ddpm', 'ddim', 'plms'], 
                        help='Override for sampling methods to use for reconstruction')
    parser.add_argument('--evaluate', action='store_true', help='Override to run reconstruction evaluation after training')
    parser.add_argument("--eval_index_json", type=str, help='Override for evaluation index JSON file')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to a pre-trained model .pth file for evaluation.')
    
    # Device
    parser.add_argument('--device', type=str, help='Override for device (cuda/cpu/auto)')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Override to use 100-sample subset for quick debugging')
    
    return parser.parse_args()