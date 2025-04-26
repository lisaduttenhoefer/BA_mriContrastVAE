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
    recon_images_plot,
)

"""
ContrastVAE builds on CVAE7. 
- It adds a supervised contrastive loss to the model. This loss is used to make the samples of the same type group in the latent space, and separate from others.
- It removes classifier loss, as the contrastive loss is used instead.
- It no longer involves the injection of label information. It is just normal supervised learning.

ContrastVAE generally suffers from the fact that with our data / sample sizes or setup, the contrastive loss is somehwat unstable.
"""


class ContrastVAE(nn.Module):

    def __init__(
        self,
        channels: int,
        num_classes: int,
        learning_rate: float,
        weight_decay: float,
        recon_loss_weight: float,
        kldiv_loss_weight: float,
        contr_loss_weight: float,
        scaler: GradScaler,
        contr_temperature: float = 0.1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        image_size: int = 128,
        latent_dim: int = 128,
        device: torch.device = None,
        dropout_prob: float = 0.1,
        use_ssim: bool = False,
        schedule_on_validation: bool = True,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
    ):
        super(ContrastVAE, self).__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            # Stack 1
            nn.Conv3d(
                channels, 16, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack2 with Dropout
            nn.Conv3d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack3
            nn.Conv3d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack 4
            nn.Conv3d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack 5
            nn.Conv3d(
                128, 256, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack 6
            nn.Conv3d(
                256, 512, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # Stack 7
            nn.Conv3d(
                512, 1024, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(p=dropout_prob),
            # reshape for latent space
            nn.Flatten(),
        )

        self.conv_layers = 7
        self.out_channels = int(16 * pow(2, self.conv_layers - 1))
        self.encoded_image_dim = int(image_size / pow(2, self.conv_layers))
        self.encoder_output_dim = int(
            self.out_channels * pow(self.encoded_image_dim, 3)
        )

        # Adjust the input dimensions for fc_mu and fc_var to include label dimensions
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            # Reshape for convolutional layers
            nn.Linear(latent_dim, self.encoder_output_dim),
            nn.Unflatten(
                1,
                (
                    self.out_channels,
                    self.encoded_image_dim,
                    self.encoded_image_dim,
                    self.encoded_image_dim,
                ),
            ),
            # Stack 7
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                1024,
                512,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 6
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                512,
                256,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 5
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                256,
                128,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 4
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                128,
                64,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 3
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                64,
                32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 2
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                32,
                16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_prob),
            # Stack 1
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.ConvTranspose3d(
                16,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm3d(channels),
            nn.Sigmoid(),
        )

        # Set training params
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scaler = scaler

        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.contr_loss_weight = contr_loss_weight

        # additional loss metrics
        self.contr_temperature = contr_temperature
        self.use_ssim = use_ssim

        # Record some values for logging
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob

        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)

        # Initialize weights
        self.apply(self.weights_init)

    def reparameterize(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        # set to train mode
        self.train()

        # Initialize loss values
        total_loss, contr_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Go through each batch in the training dataset
        for batch in train_loader:

            # Move samples to device
            images = batch["mri"][tio.DATA].to(self.device)
            labels = batch["Diagnosis"].to(self.device).argmax(dim=1)

            # Autocast for mixed precision training
            with autocast():
                # Forward pass
                recon_images, mu, logvar = self(images)

                # Calculate loss
                (
                    b_total_loss,
                    b_contr_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_img=recon_images,
                    img=images,
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
            contr_loss += 0 # b_contr_loss.item()
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

    @torch.no_grad()
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
        for batch in valid_loader:
            # Move samples to device
            images = batch["mri"][tio.DATA].to(self.device)
            labels = batch["Diagnosis"].to(self.device).argmax(dim=1)

            # Calculate loss
            with autocast():
                # Forward pass
                recon_images, mu, logvar = self(images)

                (
                    b_total_loss,
                    b_contr_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_img=recon_images,
                    img=images,
                    mu=mu,
                    log_var=logvar,
                    labels=labels,
                )

            # save total data loss stats
            total_loss += b_total_loss.item()
            contr_loss += 0 # b_contr_loss.item()
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
            # adjust learning rate based on validation loss (this is a form of regularization called early stopping, we stop training before overfitting occurs)
            self.scheduler.step(total_loss / len(valid_loader))
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)
            epoch_metrics["learning_rate"] = current_lr

        return epoch_metrics

    # Function to extract latent space
    def extract_latent_space(
        self, data_loader: DataLoader, data_type: str
    ) -> ad.AnnData:
        log_extracting_latent_space(data_type)

        # any run in self.train will have dropout etc, will not be a real representation of the latent space
        self.eval()
        latent_spaces = []
        sample_names = []

        with torch.no_grad():
            for batch in data_loader:
                # collect batch info
                images = batch["mri"][tio.DATA].to(self.device)
                names = batch["name"]

                # forward to latent space
                mu = self.to_latent(images)

                # collect running output info
                latent_spaces.append(mu.cpu().numpy())
                sample_names.extend(names)

        # create anndata object of latent space
        adata = ad.AnnData(np.concatenate(latent_spaces))
        adata.obs_names = sample_names

        return adata

    def weights_init(self, param):
        if isinstance(param, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            if param.bias is not None:
                nn.init.constant_(param.bias, 0)
        elif isinstance(param, nn.Linear):
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)

    def combined_loss_function(
        self,
        recon_img: torch.Tensor,
        img: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate supervised contrastive loss and weight it
        # contr_loss = supervised_contrastive_loss(mu, labels, self.contr_temperature)
        contr_loss = 0 # contr_loss * self.contr_loss_weight

        # We can use ssim loss or MSE loss, depending.
        if self.use_ssim:
            # Because of autograd we need to cast both images to be the same dtype
            img = img.type(recon_img.dtype)
            recon_loss = 1 - ssim(img, recon_img, data_range=1.0, size_average=True)
            recon_loss = recon_loss * self.recon_loss_weight
        else:
            # calculate reconstruction loss and weight it
            recon_loss = F.mse_loss(recon_img, img, reduction="mean")
            recon_loss = recon_loss * self.recon_loss_weight

        # calculate KLD loss per latent dimension and weight it
        kldiv_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight

        # sum all losses
        total_loss = contr_loss + recon_loss + kldiv_loss

        # return all losses
        return total_loss, contr_loss, recon_loss, kldiv_loss


def train_ContrastVAE(
    model: ContrastVAE,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    annotations: pd.DataFrame,
    model_metrics: pd.DataFrame,
    config: Config,
):
    # Print / log training start message
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

        # Train and Validate. We validate first because validation doesn't alter the model, training does.
        # So by doing it in this order, the loss generated in valid and train is based on the same model
        # Otherwise the comparison is off by one epoch, valid would be after the first correction.
        # Validate the model
        valid_metrics = model.validate(valid_loader=valid_loader, epoch=epoch)

        # train the model for one epoch
        train_metrics = model.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
        )

        # export information about the model
        # update model performance metrics
        model_metrics = update_performance_metrics(
            model_metrics, [train_metrics, valid_metrics]
        )

        # plot epoch losses
        metrics_plot(
            metrics=model_metrics,
            save_path=config.FIGURES_DIR,
            timestamp=config.TIMESTAMP,
            skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        )

        # plot rolling window metrics for the separate plot
        metrics_plot(
            metrics=model_metrics,
            save_path=config.FIGURES_DIR,
            timestamp=config.TIMESTAMP,
            skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
            rolling_window=config.METRICS_ROLLING_WINDOW,
        )

        # save model metrics
        save_model_metrics(
            model_metrics=model_metrics,
            save_path=config.MODEL_DIR,
            timestamp=config.TIMESTAMP,
            descriptor=config.RUN_NAME,
        )

        # If we're going to stop training early, we need to do a checkpoint
        if config.EARLY_STOPPING:
            if model_metrics["learning_rate"].iloc[-1] < config.STOP_LEARNING_RATE:
                is_checkpoint = True

        # save more detailed information at checkpoints
        if is_checkpoint:

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
                    "Data_Type",
                    "Sex",
                    "Age",
                    "Augmented",
                ],
                data_type="combined",
                save_path=config.FIGURES_DIR,
                timestamp=config.TIMESTAMP,
                save=True,
                show=False,
            )

            # plot latent space umap with detailed annotations
            for data, data_type, descriptor, size in zip(
                [combi_latent, combi_latent, valid_latent],
                ["train", "valid", "valid"],
                ["", "", "_solo"],
                [config.UMAP_DOT_SIZE, config.UMAP_DOT_SIZE, config.UMAP_DOT_SIZE * 3],
            ):
                latent_space_details_plot(
                    data=data,
                    epoch=epoch,
                    plot_by=[
                        "Diagnosis",
                        "Sex",
                        "Age",
                        "Augmented",
                    ],  # Data Type is set, Dataset makes no sense
                    data_type=data_type,
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    save=True,
                    show=False,
                    descriptor=descriptor,
                    size=size,
                )

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

            recon_images_plot(
                data_loader=valid_loader,
                model=model,
                save_path=config.FIGURES_DIR,
                timestamp=config.TIMESTAMP,
                epoch=epoch,
                slice_index=64,
                device=config.DEVICE,
                n_model_outputs=3,
            )

            log_checkpoint(figure_path=config.FIGURES_DIR)

            # save model and performance metrics, but not for first epoch (either useless or already saved)
            if epoch != config.START_EPOCH:
                save_model(
                    model=model,
                    save_path=config.MODEL_DIR,
                    timestamp=config.TIMESTAMP,
                    descriptor=config.RUN_NAME,
                    epoch=epoch,
                )

            # stop training early if learning rate is too low
            if config.EARLY_STOPPING:
                if model_metrics["learning_rate"].iloc[-1] < config.STOP_LEARNING_RATE:
                    log_early_stopping(
                        model_metrics["learning_rate"].iloc[-1],
                        config.STOP_LEARNING_RATE,
                        epoch,
                    )

                    break

    end_logging(config)
