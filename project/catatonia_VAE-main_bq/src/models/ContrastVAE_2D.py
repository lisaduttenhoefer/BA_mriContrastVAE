import logging
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from pytorch_msssim import ssim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_model import (
    loss_proportions,
    supervised_contrastive_loss,
    update_performance_metrics,
)
from utils.config_utils import Config

from utils.data_processing import (
    combine_latent_spaces,
    process_latent_space,
    save_model,
    save_model_metrics,
)
from utils.logging_utils import (
    end_logging,
    log_and_print,
    log_checkpoint,
    log_early_stopping,
    log_extracting_latent_space,
    log_model_metrics,
    log_training_start,
)
from utils.plotting_utils import (
    latent_space_batch_plot,
    latent_space_details_plot,
    latent_space_plot,
    metrics_plot,
    recon_data_plot,
)
#hab ich nicht????

"""
NOW: implemented for processing 2D tabular data instead of 3D MRI images

ContrastVAE builds on CVAE7. 
- It adds a supervised contrastive loss to the model. This loss is used to make the samples of the same type group in the latent space, and separate from others.
- It removes classifier loss, as the contrastive loss is used instead.
- It no longer involves the injection of label information. It is just normal supervised learning.

ContrastVAE generally suffers from the fact that with our data / sample sizes or setup, the contrastive loss is somehwat unstable.
"""


class ContrastVAE_2D(nn.Module):

    def __init__(
        self, 
        #channels: int,
        num_classes: int, 
        #number of classes the model should predict
        learning_rate: float, #learning rate optimizer
        weight_decay: float, #weight reconstruction loss
        recon_loss_weight: float, #weight KL divergence loss
        kldiv_loss_weight: float, #weight classifier loss
        contr_loss_weight: float, #!! contrast loss !!
        scaler: GradScaler, #The gradient scaler for mixed precision training
        contr_temperature: float = 0.1, #??
        
        input_dim: int = None, #input feature vector dim
        hidden_dim_1: int = 100, # The dimension of the hidden layers in encoder and decoder
        hidden_dim_2: int = 100,
        latent_dim: int = 20, #latent dimension of the model
        device: torch.device = None,  # The gpu to train the model on
        #dropout_prob: float = 0.1,
        #use_ssim: bool = False, #MSE  statt SSIM -> default?
        schedule_on_validation: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
    ):
        super(ContrastVAE_2D, self).__init__()
        
        self.num_classes = num_classes

        # Encoder with 1 layer-block (one hidden layer before latent space)
        #self.encoder = nn.Sequential(
        #    # Layer 1
        #    nn.Linear(input_dim, hidden_dim),
        #    nn.LeakyReLU(1e-2),
            # Layer 2
        #    nn.Linear(hidden_dim, latent_dim),)
        
        #ENCODER wie in "Pinaya-Using normative modelling to detect disease progression...""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),   # 1st hidden layer
            nn.LeakyReLU(1e-2),
            #nn.BatchNorm1d(hidden_dim),
            #nn.Dropout(dropout_prob), 

            nn.Linear(hidden_dim_1, hidden_dim_2),         # 2nd hidden layer
            nn.LeakyReLU(1e-2),
            #nn.BatchNorm1d(hidden_dim_2),
            #nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim_2, latent_dim)           # Output ->  Latent Code
)
        
        #no lokal spacial structure -> no convolution, no flattening -> Feedforward-Multilayer-Perceptron (MLP)langt

        self.encoder_feature_dim = int(latent_dim)

        # Adjust the input dimensions for fc_mu and fc_var to include label dimensions
        self.fc_mu = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_feature_dim, latent_dim)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),   # 1. Layer nach dem Latent Code
            nn.LeakyReLU(1e-2),
            # nn.BatchNorm1d(hidden_dim_2),
            # nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim_2, hidden_dim_1),   # 2. Hidden Layer
            nn.LeakyReLU(1e-2),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim_1, input_dim)       # Rekonstruktion des Inputs
        )


        # Set training params
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        #scheduler for learning rate like in original
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        
        
        self.scaler = scaler

        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.contr_loss_weight = contr_loss_weight

        # additional loss metrics
        self.contr_temperature = contr_temperature #what is this?
        #self.use_ssim = use_ssim -> nicht weil für 3D -> braucht man stattdessen was?

        # Record some values for logging
        self.schedule_on_validation = schedule_on_validation 
        self.scheduler_patience = scheduler_patience 
        self.scheduler_factor = scheduler_factor 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

       
        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)

        # Initialize weights
        self.apply(self.weights_init)

    #sampling within the neuronal network (Werte aus Verteilung), without interrupting the backpropagation
    # This function performs reparameterization, which is used to sample from the latent space before decoding.
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # self: The latent space produced by the encode

        # standard deviation
        std = torch.exp(0.5 * logvar)
        # random element between 0 and 1 (Standardnormalverteilung)
        eps = torch.randn_like(std)

        # returns a random sample from the latent space distribution
        return mu + eps * std

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        x = self.encoder(x)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        x = self.decoder(z)

        return x, mu, logvar

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.encoder(x)

        # Get latent space
        mu = self.fc_mu(x)

        return mu

    #TRAINING (1 epoche)
    def train_one_epoch(self, train_loader: DataLoader, epoch: int,) -> Dict[str, float]:
        # set to train mode
        self.train()
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0   # Initialize

        # Iteriate through batches (TRAIN_LOADER)
        # Go through each batch in the training dataset
        for batch_idx, (measurements, labels, names) in enumerate(train_loader):
            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)

            # Autocast for mixed precision training
            with autocast():
                # Forward pass
                recon_data, mu, logvar = self(batch_measurements)

                # Calculate loss
                (
                    b_total_loss,
                    b_contr_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_measurements=recon_data,
                    meas=batch_measurements,
                    mu=mu,
                    log_var=logvar,
                    labels=labels,
                )

            # Backward pass with gradient scaling
            self.scaler.scale(b_total_loss).backward()

            # Remove NaNs from gradients
            for param in self.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    param.grad.nan_to_num_(0.0)

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

            # Update weights w.r.t. loss
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # save total data loss stats
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "t_contr_loss": contr_loss / len(train_loader.dataset),
            "t_recon_loss": recon_loss / len(train_loader.dataset),
            "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
        }

        epoch_props = loss_proportions("train_loss", epoch_metrics)

        # log batch metrics
        log_model_metrics(
            epoch,
            epoch_props,
            type="Training Metrics:",
        )

        if not self.schedule_on_validation:
            # adjust learning rate based on training loss (may lead to overfitting)
            self.scheduler.step(total_loss / len(train_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr

        return epoch_metrics

    @torch.no_grad() #PyTorch won’t compute gradients for anything inside sample_latent
    def validate(
        self,
        valid_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        # set to evaluation mode
        self.eval()

        # Initialize loss values
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Go through each batch in the validation dataset
        for batch,idx, (measurements, labels, names) in enumerate(valid_loader):
            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)

            # Calculate loss
            with autocast():
                # Forward pass
                recon_data, mu, logvar = self(batch_measurements)

                (
                    b_total_loss,
                    b_contr_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_measurements=recon_data,
                    meas=batch_measurements,
                    mu=mu,
                    log_var=logvar,
                    labels=labels,
                )
                 
            # save total data loss stats
            total_loss += b_total_loss.item()
            contr_loss += b_contr_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

        epoch_metrics = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_contr_loss": contr_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
        }

        epoch_props = loss_proportions("valid_loss", epoch_metrics)

        log_model_metrics(
            epoch,
            epoch_props,
            type="Validation Metrics:",
        )

        if self.schedule_on_validation:
            # adjust learning rate based on validation loss
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr

        return epoch_metrics

    # Function to extract latent space -> AnnData analysis
    @torch.no_grad()
    def extract_latent_space(
        self, data_loader: DataLoader, data_type: str
    ) -> ad.AnnData:
        log_extracting_latent_space(data_type)

        self.eval()
        latent_spaces = []
        sample_names = []

        for batch_idx, (measurements, labels, names) in enumerate(data_loader):
            # collect batch info
            batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

            # forward to latent space
            mu = self.to_latent(batch_measurements)

            # collect running output info
            latent_spaces.append(mu.cpu().numpy())
            sample_names.extend(names)

        # create anndata object of latent space
        adata = ad.AnnData(np.concatenate(latent_spaces))
        adata.obs_names = sample_names

        return adata

    def weights_init(self, param):
        #kann man drin lassen, aber auch rausnehmen? Ist doch sicher linear?
        if isinstance(param, (nn.Conv3d, nn.ConvTranspose3d)):       # if the parameter is a convolutional layer, use Kaiming Uniform initialization
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            if param.bias is not None:
                nn.init.constant_(param.bias, 0)
        elif isinstance(param, nn.Linear):                 # if the parameter is a linear layer, use Xavier Uniform initialization
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)

#MAIN LOSS FUNCTION 
    def combined_loss_function(self, recon_measurements: torch.Tensor,meas: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, labels: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #recon_measurements: recon_data after decoder
        #meas: original data
        #labels: diagnostic group labels

        # calculate supervised contrastive loss and weight it
        contr_loss = supervised_contrastive_loss(mu, labels, self.contr_temperature)
        contr_loss = contr_loss * self.contr_loss_weight

       # calculate reconstruction loss and weight it -> original -> ssim
        recon_loss = F.mse_loss(recon_mes, mes, reduction="mean")
        recon_loss = recon_loss * self.recon_loss_weight

        # calculate KLD loss per latent dimension and weight it
        kldiv_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight

        # sum all losses
        total_loss = contr_loss + recon_loss + kldiv_loss

        # return all losses
        return total_loss, contr_loss, recon_loss, kldiv_loss

#TRAINING FUNCTION 
def train_ContrastVAE_2D(
    model: ContrastVAE_2D,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    annotations: pd.DataFrame,
    model_metrics: pd.DataFrame,
    config: Config,
    # whether to report the accuracy to Ray Tune for hyperparameter tuning
    accuracy_tune: bool = False,
    
    no_plotting: bool = True, # UMAP plotting
    no_val_plotting: bool = False, #training plotting

    no_saving: bool = False #model saving @checkpoints
):
    # (logging) Print / log training start message 
    log_training_start()

    # Train and Validate once per epoch
    for epoch in tqdm(
        range(config.START_EPOCH, config.FINAL_EPOCH),
        initial=config.START_EPOCH,
        ncols=100,
        desc="\nEpochs",
    ):
        # Separate log messages from different epochs
        print("\n")

        # determine if it is a checkpoint epoch
        is_checkpoint = (
            epoch % config.CHECKPOINT_INTERVAL == 0 or epoch == config.FINAL_EPOCH - 1
        )

       
       
       
        # Validate the model
        valid_metrics = model.validate(valid_loader=valid_loader, epoch=epoch)

        # train the model for one epoch
        train_metrics = model.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch, )
       #Niklas:
       best_accuracy = 0
       if epoch != config.START_EPOCH:
            if valid_metrics["accuracy"] > best_accuracy:
                best_accuracy = valid_metrics["accuracy"]
                save_model(
                        model=model,
                        save_path=config.MODEL_DIR,
                        timestamp=config.TIMESTAMP,
                        descriptor=config.RUN_NAME,
                        epoch='best',
                    )

        
        model_metrics = update_performance_metrics(
            model_metrics, [train_metrics, valid_metrics]
        )

        # if not no_plotting:
        #     # plot epoch losses
        #     metrics_plot(
        #         metrics=model_metrics,
        #         save_path=config.FIGURES_DIR,
        #         timestamp=config.TIMESTAMP,
        #         skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        #     )
        #     # plot rolling window metrics as well
        #     metrics_plot(
        #         metrics=model_metrics,
        #         save_path=config.FIGURES_DIR,
        #         timestamp=config.TIMESTAMP,
        #         skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        #         rolling_window=config.METRICS_ROLLING_WINDOW,
        #     )

        # # save model metrics
        # save_model_metrics(
        #     model_metrics=model_metrics,
        #     save_path=config.MODEL_DIR,
        #     timestamp=config.TIMESTAMP,
        #     descriptor=config.RUN_NAME,
        # )

        # checkpoint in case of early stop
        if config.EARLY_STOPPING:
            if model_metrics["learning_rate"].iloc[-1] < config.STOP_LEARNING_RATE:
                is_checkpoint = True

        
        if is_checkpoint:

            if not no_plotting:
            # extract latent space and process it
                valid_latent = model.extract_latent_space(valid_loader, "Validation Data")
                train_latent = model.extract_latent_space(train_loader, "Training Data")

               
                train_latent = process_latent_space(
                    adata=train_latent,
                    annotations=annotations,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="train",
                )
                valid_latent = process_latent_space(
                    adata=valid_latent,
                    annotations=annotations,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="valid",
                )
                # combination of validation & training -> similar?
               
                combi_latent = combine_latent_spaces(
                    tdata=train_latent,
                    vdata=valid_latent,
                    umap_neighbors=config.UMAP_NEIGHBORS,
                    seed=config.SEED,
                    save_data=True,
                    save_path=config.DATA_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    data_type="combined",
                )
              
                # plot latent space umap
                latent_space_plot(
                    data=combi_latent,
                    epoch=epoch,
                    plot_types=["umap"],
                    plot_by=[
                        "Diagnosis", 
                        "Dataset",
                        "Sex",
                        "Age",
                    ],
                    data_type="combined",
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    save=True,
                    show=False,
                )

                # plot latent space umap with detailed annotations -> combined & alone
                # If there's only validation data, we use bigger dots as there's less data
                for data, data_type, descriptor, size in zip(
                    [combi_latent, combi_latent, valid_latent],
                    ["train", "valid", "valid"],
                    ["", "", "_solo"],
                    [
                        config.UMAP_DOT_SIZE,
                        config.UMAP_DOT_SIZE,
                        config.UMAP_DOT_SIZE * 3,
                    ],
                ):
                    latent_space_details_plot(
                        data=data,
                        epoch=epoch,
                        plot_by=[
                            "Diagnosis",
                            "Sex",
                            "Age",
                        ],  # Data Type is set, Dataset makes no sense
                        data_type=data_type,
                        save_path=config.FIGURES_DIR,
                        timestamp=config.TIMESTAMP,
                        save=True,
                        show=False,
                        descriptor=descriptor,
                        size=size,
                    )

                # plot latent space batch plot. This shows off batch effect in individual diagnoses well
                # since it has less data, we use bigger dots
                for data, data_type, size in zip(
                    [train_latent, valid_latent],
                    ["train", "valid"],
                    [config.UMAP_DOT_SIZE * 3, config.UMAP_DOT_SIZE * 9],
                ):
                    latent_space_batch_plot(
                        data=data,
                        epoch=epoch,
                        data_type=data_type,
                        save_path=config.FIGURES_DIR,
                        timestamp=config.TIMESTAMP,
                        save=True,
                        show=False,
                        size=size,
                    )

                # plot reconstructed images that the decoder produces
                recon_images_plot(
                    data_loader=valid_loader,
                    model=model,
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    epoch=epoch,
                    slice_index=64,
                    device=config.DEVICE,
                    n_model_outputs=4,
                )

            # log that we've done the checkpoint plots
            log_checkpoint(figure_path=config.FIGURES_DIR)

            # save model and performance metrics, but not if it's the first epoch (either useless or already saved)
            if epoch != config.START_EPOCH and not no_saving:
                save_model(
                    model=model,
                    save_path=config.MODEL_DIR,
                    timestamp=config.TIMESTAMP,
                    descriptor=config.RUN_NAME,
                    epoch=epoch,
                )

            # stop training early if learning rate is too low
            if config.EARLY_STOPPING:
                if model_metrics["learning_rate"].iloc[-1] <= config.STOP_LEARNING_RATE:

                    # log that we're stopping early
                    log_early_stopping(
                        model_metrics["learning_rate"].iloc[-1],
                        config.STOP_LEARNING_RATE,
                        epoch,
                    )

                    # break the training loop
                    break

    #if not no_plotting:
            # plot epoch losses
    if not no_val_plotting:
        metrics_plot(
            metrics=model_metrics,
            save_path=config.FIGURES_DIR,
            timestamp=config.TIMESTAMP,
            skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        )
        # # plot rolling window metrics as well
        # metrics_plot(
        #     metrics=model_metrics,
        #     save_path=config.FIGURES_DIR,
        #     timestamp=config.TIMESTAMP,
        #     skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        #     rolling_window=config.METRICS_ROLLING_WINDOW,
        # )

        # save model performance metrics
        save_model_metrics(
            model_metrics=model_metrics,       
           
            save_path=config.MODEL_DIR,
            timestamp=config.TIMESTAMP,
            descriptor=config.RUN_NAME,
            epoch=epoch,
        )

           
    # log the best accuracy and at which epoch it was achieved -> Niklas 
    log_and_print(f"Best Accuracy: {model_metrics['accuracy'].max()} at epoch {model_metrics['accuracy'].idxmax()}")

    end_logging(config)