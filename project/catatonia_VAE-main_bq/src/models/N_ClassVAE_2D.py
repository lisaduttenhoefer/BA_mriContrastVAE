import logging
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from pytorch_msssim import ssim
from ray import train
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_model import loss_proportions, update_performance_metrics
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
    recon_images_plot,
)

"""
ClassVAE builds on CVAE7 and ContrastVAE. 
- It replacers supervised contrastive learning with a classifier.
- It no longer involves the injection of label information. It is just normal supervised learning.

ClassVAE was the most promising model, with classifier loss generally generalizing well and being robust.
"""


class ClassVAE_2D(nn.Module):

    def __init__(
        self,
        # The number of classes (diagnostic groups) the model should predict
        num_classes: int,
        # The learning rate for the optimizer
        learning_rate: float,
        # The weight decay for the optimizer
        weight_decay: float,
        # The weight of the reconstruction loss
        recon_loss_weight: float,
        # The weight of the KL Divergence loss
        kldiv_loss_weight: float,
        # The weight of the classifier loss
        class_loss_weight: float,
        # The gradient scaler for mixed precision training
        scaler: GradScaler,
        # The dimension of the input feature vector
        input_dim: int = None,
        # The dimension of the hidden layers in encoder and decoder
        hidden_dim: int = 100, 
        # The latent dimension of the model
        latent_dim: int = 20,
        # The gpu to train the model on
        device: torch.device = None,
        # Should the model use SSIM for the reconstruction loss
        # use_ssim: bool = False,
        # Should the learning rate be scheduled based on the validation loss (True) or training loss (False)
        schedule_on_validation: bool = True,
        # The patience for the learning rate scheduler
        scheduler_patience: int = 10,
        # The factor learning rate is reduced by when the scheduler is triggered
        scheduler_factor: float = 0.5,
    ):
        super(ClassVAE_2D, self).__init__()  # should be possible to change this to super().__init__()

        # This model has 3 main elements:
        # 1. Encoder
        # 2. Classifier on Latent Space
        # 3. Decoder

        # 1. Encoder
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(1e-2),
            # Layer 2
            nn.Linear(hidden_dim, latent_dim),
        )

        self.encoder_feature_dim = int(latent_dim)

        # Latent Space, generates as mu and logvar for reparameterization
        # Latent space has the dimensionality of the latent_dim parameter.
        # Whenever something uses the latent space, it will usually be referring to mu, since z is noisy.
        self.fc_mu = nn.Linear(self.encoder_feature_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_feature_dim, latent_dim)

        # 2. Classifier
        # The classifier is a simple neural network.
        # It takes the mu as input and outputs logits for each class.
        # The number of classes predicted is determined by the num_classes parameter.
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            # Layer 2
            nn.Linear(latent_dim, int(latent_dim / 2)),
            nn.ReLU(),
            # Output
            nn.Linear(int(latent_dim / 2), num_classes),
        )

        # 3. Decoder
        self.decoder = nn.Sequential(
            # Layer 1
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(1e-2),
            # Layer 2
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For normalization to [0,1]
        )

        # Set training parameters
        # Set optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Set learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        # Set gradient scaler
        self.scaler = scaler

        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.class_loss_weight = class_loss_weight

        # set number of classes
        self.num_classes = num_classes

        # additional loss metrics
        # self.use_ssim = use_ssim

        # Record some values for logging
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim

        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)

        # Initialize weights
        self.apply(self.weights_init)

    # The base forward function for the model
    def forward(
        self,
        # The input image
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Encoder
        x = self.encoder(x)

        # Get latent space
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # Classifier
        class_logits = self.classifier(mu)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decoder
        x = self.decoder(z)

        # Return the reconstructed image, the latent space, the log variance, and the classifier logits
        return x, mu, logvar, class_logits

    # This function is used to convert an image to the latent space.
    # This is useful for investigating the latent space the mode produces.
    def to_latent(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        x = self.encoder(x)

        # Get latent space
        mu = self.fc_mu(x)

        # Classifier
        class_logits = self.classifier(mu)

        # Return the the latent space and the classifier logits
        return mu, class_logits

    # This function performs reparameterization, which is used to sample from the latent space before decoding.
    def reparameterize(
        self,
        # The latent space produced by the encoder
        mu: torch.Tensor,
        # The log variance produced by the encoder
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        # standard deviation
        std = torch.exp(0.5 * logvar)

        # random element between 0 and 1
        eps = torch.randn_like(std)

        # return a random sample from the latent space distribution
        return mu + eps * std

    # This function trains the model for one epoch.
    def train_one_epoch(
        self,
        # the training data loader
        train_loader: DataLoader,
        # the epoch number (used for logging)
        epoch: int,
    ) -> Dict[str, float]:

        # set model to train mode (important for batch normalization and dropout)
        self.train()

        # Initialize loss values, total loss is the sum of the class, reconstruction, and KL Divergence losses
        total_loss, class_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Go through each batch in the training dataset
        for batch_idx, (measurements, labels, names) in enumerate(train_loader):

            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)
            #images = batch["mri"][tio.DATA].to(self.device)
            #labels = batch["Diagnosis"].to(self.device)

            # Autocast for mixed precision training (faster training)
            with autocast():
                # Forward pass through model
                recon_measurements, mu, logvar, class_logits = self(batch_measurements)

                # Calculate loss
                (
                    b_total_loss,
                    b_class_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_mes=recon_measurements,
                    mes=batch_measurements,
                    mu=mu,
                    log_var=logvar,
                    class_logits=class_logits,
                    labels=batch_labels,
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
            class_loss += b_class_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

        # calculate loss averaged by the number of samples
        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "t_class_loss": class_loss / len(train_loader.dataset),
            "t_recon_loss": recon_loss / len(train_loader.dataset),
            "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
        }

        # loss proportions are a bit more interpretable for printing and logging
        epoch_props = loss_proportions("train_loss", epoch_metrics)

        # log and print loss proportions
        log_model_metrics(
            epoch,
            epoch_props,
            type="Training Metrics:",
        )

        # if we're not scheduling on validation, adjust learning rate based on training loss
        if not self.schedule_on_validation:
            # adjust learning rate based on training loss (may lead to overfitting)
            self.scheduler.step(total_loss / len(train_loader))

            # get current learning rate, log it
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)

            # save learning rate as a metric
            epoch_metrics["learning_rate"] = current_lr

        # return epoch losses (not proportions!)
        return epoch_metrics

    # This function validates the model on the validation dataset.
    # It produces important metrics for evaluating the model.
    # No gradients are calculated during validation.
    @torch.no_grad()
    def validate(
        self,
        # the validation data loader
        valid_loader: DataLoader,
        # the epoch number (used for logging)
        epoch: int,
    ) -> Dict[str, float]:

        # set model to evaluation mode, important for batch normalization and dropout
        self.eval()

        # Initialize loss values, total loss is the sum of the class, reconstruction, and KL Divergence losses
        total_loss, class_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Initialize lists to store classifier predictions and diagnostic labels
        all_preds = []
        all_labels = []

        # Go through each batch in the validation dataset
        for batch_idx, (measurements, labels, names) in enumerate(valid_loader):

            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)
            batch_labels = torch.stack([label for label in labels]).to(self.device)

            # Calculate loss, autocast for mixed precision training (faster)
            with autocast():
                # Forward pass
                recon_measurments, mu, logvar, class_logits = self(batch_measurements)

                # Calculate loss
                (
                    b_total_loss,
                    b_class_loss,
                    b_recon_loss,
                    b_kldiv_loss,
                ) = self.combined_loss_function(
                    recon_mes=recon_measurments,
                    mes=batch_measurements,
                    mu=mu,
                    log_var=logvar,
                    class_logits=class_logits,
                    labels=batch_labels,
                )

            # sum loss stats for this batch
            total_loss += b_total_loss.item()
            class_loss += b_class_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

            # Calculate Model Predictions and format diagnostic labels
            preds = torch.argmax(class_logits, dim=1)
            labels = torch.argmax(labels, dim=1)

            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Caclulate confusion matrix metrics based on all predictions and labels
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=np.nan
        )

        # Store confusion matrix metrics
        epoch_confusion = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }

        # Calculate loss averaged by the number of samples
        epoch_losses = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_class_loss": class_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
        }

        # loss proportions are a bit more interpretable for printing and logging
        epoch_props = loss_proportions("valid_loss", epoch_losses)

        # log and print loss proportions as well as confusion matrix metrics
        log_model_metrics(
            epoch,
            {**epoch_confusion, **epoch_props},
            type="Validation Metrics:",
        )

        # collect all metrics (but not loss proportions)
        epoch_metrics = {**epoch_losses, **epoch_confusion}

        # if we're scheduling on validation, adjust learning rate based on validation loss
        if self.schedule_on_validation:
            # adjust learning rate based on validation loss (this is a form of regularization called early stopping, we stop training before overfitting occurs)
            self.scheduler.step(total_loss / len(valid_loader))

            # get current learning rate, log it
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)

            # save learning rate as a metric
            epoch_metrics["learning_rate"] = current_lr

        # return all metrics
        return epoch_metrics

    # Function to extract latent embedding of data fed to the model.
    # This is useful for investigating the latent space the model produces.
    # Returns an AnnData object with the latent space and classifier predictions.
    # No gradients are calculated during this function.
    @torch.no_grad()
    def extract_latent_space(
        self,
        # The data loader to get the laten representation of
        data_loader: DataLoader,
        # A string to describe the data type (for logging)
        data_type: str = "Data",
    ) -> ad.AnnData:

        # log that we are extracting latent space
        log_extracting_latent_space(data_type)

        # any run in self.train will have dropout etc, will not be a real representation of the latent space
        self.eval()

        # Initialize lists to store latent spaces, sample names, and classifier predictions
        # Any other metadata can be added by the process_latent_space function instead.
        latent_spaces = []
        sample_names = []
        predictions = []

        # Go through each batch in the data loader
        for batch_idx, (measurements, labels, names) in enumerate(data_loader):
            # collect batch info
            batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

            # forward to latent space
            mu, class_logits = self.to_latent(batch_measurements)

            # collect running output info
            latent_spaces.append(mu.cpu().numpy())
            sample_names.extend(names)
            predictions.append(class_logits.argmax(dim=1).cpu().numpy())

        # create anndata object of latent space
        adata = ad.AnnData(np.concatenate(latent_spaces))

        # add metadata to anndata object, namely sample names and classifier predictions
        adata.obs_names = sample_names
        adata.obs["Diagnosis_pred"] = pd.Categorical(np.concatenate(predictions))

        # return anndata object
        return adata

    # Function to initialize weights of the model, using the Kaiming Uniform and Xavier Uniform initializations.
    # This is important for starting the model with good weights, which can help with training.
    def weights_init(self, param: nn.Module):

        # if the parameter is a convolutional layer, use Kaiming Uniform initialization
        if isinstance(param, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            if param.bias is not None:
                nn.init.constant_(param.bias, 0)

        # if the parameter is a linear layer, use Xavier Uniform initialization
        elif isinstance(param, nn.Linear):
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)

    # Function to calculate the combined loss of the model.
    # This is the main loss function of the model, which is used to train the model.
    # The total loss is the sum of the classifier, reconstruction, and KL Divergence losses.
    # The total loss is used to update the model weights.
    def combined_loss_function(
        self,
        # The reconstructed image, after passing through the decoder
        recon_mes: torch.Tensor,
        # The original 3D MRI image
        mes: torch.Tensor,
        # The latent space mean, produced by the encoder
        mu: torch.Tensor,
        # The latent space log variance, produced by the encoder
        log_var: torch.Tensor,
        # The classifier logits (1 per diagnostic group), produced by the classifier
        class_logits: torch.Tensor,
        # The diagnostic group labels
        labels: torch.Tensor,
    ):
        # calculate diagnosis classification loss and weight it
        target_for_loss = labels.argmax(dim=1).flatten() # flatten to bring from [N,1] to [N,] for cross entropy calculation. 

        class_loss = F.cross_entropy(
            class_logits.float(), target_for_loss, reduction="mean"
        )
        class_loss = class_loss * self.class_loss_weight

        # We can use ssim loss or MSE loss, depending.
        # if self.use_ssim:
        #     # Because of autograd we need to cast both images to be the same dtype
        #     img = img.type(recon_img.dtype)
        #     recon_loss = 1 - ssim(img, recon_img, data_range=1.0, size_average=True)
        #     recon_loss = recon_loss * self.recon_loss_weight
        # else:
            # calculate reconstruction loss and weight it
        recon_loss = F.mse_loss(recon_mes, mes, reduction="mean")
        recon_loss = recon_loss * self.recon_loss_weight

        # calculate KLD loss and weight it
        kldiv_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kldiv_loss = kldiv_loss * self.kldiv_loss_weight

        # sum all losses
        total_loss = recon_loss + kldiv_loss + class_loss

        # return all losses
        return total_loss, class_loss, recon_loss, kldiv_loss


# This function trains the entire ClassVAE model.
# It uses many presets and parameters from the Config class.
# It trains the model for a set number of epochs, and can be stopped early if the learning rate is too low.
# It also saves the model and performance metrics at checkpoints.
# The model is trained using the train_loader and validated using the valid_loader.
# The annotations are used to process the latent space and plot the latent space.
# The model_metrics are used to store the performance metrics of the model.
# The config is used to set the training parameters.
# The accuracy_tune parameter is used to report the accuracy to Ray Tune for hyperparameter tuning.
# The no_plotting parameter is used to skip plotting, which can be unstable with Ray Tune.
def train_ClassVAE_2D(
    # the ClassVAE model you want to train
    model: ClassVAE_2D,
    # the training data loader
    train_loader: DataLoader,
    # the validation data loader
    valid_loader: DataLoader,
    # the annotations for the training and validation data
    # should contain columns for "Data_Type", "Diagnosis", "Dataset", "Age", "Augmented" and "Sex"
    annotations: pd.DataFrame,
    # the model metrics to store the performance metrics of the model
    # If you're loading a model, you can load the model metrics from the previous training run
    model_metrics: pd.DataFrame,
    # the configuration for the training run
    config: Config,
    # whether to report the accuracy to Ray Tune for hyperparameter tuning
    accuracy_tune: bool = False,
    # whether to skip plotting of UMAPs. Plotting can be unstable with Ray Tune
    no_plotting: bool = True,
    # wether to skip plotting of training metrics. Plotting can be unstable with Ray Tune.
    no_val_plotting: bool = False,
    # wether to save at every checkpoint the respective model
    no_saving: bool = False
):
    # Print / log training start message
    log_training_start()

    best_accuracy = 0
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

        # export information about the model
        # update model performance metrics
        model_metrics = update_performance_metrics(
            model_metrics, [train_metrics, valid_metrics]
        )

        # report accuracy to Ray Tune for hyperparameter tuning
        if accuracy_tune:
            train.report({"accuracy": model_metrics["accuracy"].max()})

        # If we're going to stop training early, we need to do a checkpoint
        if config.EARLY_STOPPING:
            if model_metrics["learning_rate"].iloc[-1] <= config.STOP_LEARNING_RATE:
                is_checkpoint = True

        # save more detailed information at checkpoints
        if is_checkpoint:

            if not no_plotting:
            # extract latent space and process it
                valid_latent = model.extract_latent_space(valid_loader, "Validation Data")
                train_latent = model.extract_latent_space(train_loader, "Training Data")

                # processing latent space means adding aligned annotations
                # as well as calculating PCA, Neighbors, and UMAP
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
                # combine the validation and training latent spaces
                # this is useful for plotting the overall latent space and to see if
                # training is generalizing well (i.e. validation is similar to training)
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
                        "Diagnosis_pred",
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

                # plot latent space umap with detailed annotations
                # we plot the combined latent space, as it contains both training and validation data
                # but we also plot validation data solo, to see how well it generalizes
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
                            "Diagnosis_pred",
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
        )

    # log the best accuracy and at which epoch it was achieved
    log_and_print(
        f"Best Accuracy: {model_metrics['accuracy'].max()} at epoch {model_metrics['accuracy'].idxmax()}"
    )

    # once the training loop is done, end logging
    end_logging(config)