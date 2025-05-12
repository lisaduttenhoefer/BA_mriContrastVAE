import os
import argparse
import time
import logging
import datetime
import pandas as pd
import torch
from torch.cuda.amp import GradScaler

# Import your modules
from utils.config_utils_model import Config_2D
from utils.dataset_utils import prepare_datasets
from utils.logging_utils import setup_logging

# Import the Ray Tune implementation
from raytune_integration import (
    ContrastVAE_2D,
    run_ray_tune_optimization,
    train_ContrastVAE_2D
)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Ray Tune optimization for ContrastVAE_2D")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials for Ray Tune")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum epochs per trial")
    parser.add_argument("--cpus_per_trial", type=float, default=1, help="CPUs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=0.5, help="GPUs per trial")
    parser.add_argument("--train_best", action="store_true", help="Train the best model after tuning")
    args = parser.parse_args()
    
    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    figures_dir = os.path.join(output_dir, "figures")
    
    # Create directories if they don't exist
    for directory in [output_dir, model_dir, log_dir, figures_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_dir, f"{timestamp}_raytune.log")
    setup_logging(log_file)
    logging.info("Starting Ray Tune optimization for ContrastVAE_2D")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create base configuration
    config = Config_2D()
    config.DATA_DIR = args.data_dir
    config.OUTPUT_DIR = output_dir
    config.MODEL_DIR = model_dir
    config.LOG_DIR = log_dir
    config.FIGURES_DIR = figures_dir
    config.TIMESTAMP = timestamp
    config.DEVICE = device
    config.RUN_NAME = "raytune_experiment"
    config.BATCH_SIZE = 32
    config.CHECKPOINT_INTERVAL = 5
    config.START_EPOCH = 0
    config.EARLY_STOPPING = True
    config.STOP_LEARNING_RATE = 1e-6
    config.UMAP_NEIGHBORS = 15
    config.UMAP_DOT_SIZE = 10
    config.DONT_PLOT_N_EPOCHS = 2
    config.SEED = 42
    config.SCHEDULE_ON_VALIDATION = True
    
    # Load and prepare datasets
    logging.info("Loading and preparing datasets...")
    train_loader, valid_loader, test_loader, annotations = prepare_datasets(
        data_dir=args.data_dir,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED
    )
    
    # Define dataset properties for model initialization
    dataset_properties = {
        "num_classes": len(annotations["Diagnosis"].unique()),
        "input_dim": next(iter(train_loader))[0][0].shape[0]  # Get the dimension of the first sample
    }
    
    logging.info(f"Dataset properties: {dataset_properties}")
    
    # Run Ray Tune optimization
    logging.info("Starting Ray Tune optimization...")
    results = run_ray_tune_optimization(
        train_loader=train_loader,
        valid_loader=valid_loader,
        annotations=annotations,
        base_config=config,
        dataset_properties=dataset_properties,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
        cpus_per_trial=args.cpus_per_trial,
        gpus_per_trial=args.gpus_per_trial
    )
    
    # Get the best trial
    best_trial = results.get_best_trial("accuracy", "max", "last")
    best_config = best_trial.config
    
    logging.info(f"Best trial config: {best_config}")
    logging.info(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    
    # Save the best configuration
    best_config_path = os.path.join(output_dir, "best_config.txt")
    with open(best_config_path, "w") as f:
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
    
    # Train the best model if requested
    if args.train_best:
        logging.info("Training the best model configuration...")
        
        # Update config with best parameters
        for key, value in best_config.items():
            setattr(config, key, value)
        
        # Set number of epochs for full training
        config.FINAL_EPOCH = 50  # You can adjust this
        
        # Initialize metrics dataframe
        model_metrics = pd.DataFrame(
            columns=[
                "train_loss",
                "valid_loss",
                "accuracy",
                "t_contr_loss",
                "t_recon_loss",
                "t_kldiv_loss",
                "v_contr_loss",
                "v_recon_loss",
                "v_kldiv_loss",
                "learning_rate",
            ]
        )
        
        # Create gradient scaler for mixed precision training
        scaler = GradScaler()
        
        # Initialize model with best hyperparameters
        best_model = ContrastVAE_2D(
            num_classes=dataset_properties["num_classes"],
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            recon_loss_weight=config.RECON_LOSS_WEIGHT,
            kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
            contr_loss_weight=config.CONTR_LOSS_WEIGHT,
            scaler=scaler,
            contr_temperature=config.CONTR_TEMPERATURE,
            input_dim=dataset_properties["input_dim"],
            hidden_dim_1=config.HIDDEN_DIM_1,
            hidden_dim_2=config.HIDDEN_DIM_2,
            latent_dim=config.LATENT_DIM,
            device=config.DEVICE,
            dropout_prob=config.DROPOUT_PROB,
            schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
            scheduler_patience=config.SCHEDULER_PATIENCE,
            scheduler_factor=config.SCHEDULER_FACTOR
        )
        
        # Train the best model
        config.RUN_NAME = "best_model"
        train_ContrastVAE_2D(
            model=best_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            annotations=annotations,
            model_metrics=model_metrics,
            config=config,
            accuracy_tune=False,
            no_plotting=False,
            no_val_plotting=False,
            no_saving=False
        )
        
        logging.info("Best model training completed!")
        
        # Evaluate on test set if available
        if test_loader is not None:
            logging.info("Evaluating best model on test set...")
            test_metrics = best_model.validate(test_loader, epoch=config.FINAL_EPOCH - 1)
            logging.info(f"Test metrics: {test_metrics}")
            
            # Save test metrics
            test_metrics_path = os.path.join(output_dir, "test_metrics.txt")
            with open(test_metrics_path, "w") as f:
                for key, value in test_metrics.items():
                    f.write(f"{key}: {value}\n")
    
    logging.info("Ray Tune optimization completed!")

if __name__ == "__main__":
    main()