import math
import os
import random as rd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Literal
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import torchio as tio
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from module.data_processing_hc import load_mri_data_2D
from utils.logging_utils import log_and_print

"""
Plotting functions for all the train_ModelXYZ() functions used in the  run_ModelXYZ.py files.
"""


# This function plots the metrics of a model over epochs. It can plot each metric in a separate subplot, but it can also
# combine all the training and validation losses in two subplots, accompanied by an overall train/validation comparison.
# The function can also plot a rolling average of the metrics, and can skip the first N epochs of plotting (the firs
# epochs of training can mess with the plot scales). The function can save the plots to a specified path, and can also
# show the plots in the console.
def metrics_plot(
    # The metrics dataframe, containing the metrics of the model over epochs. Index is epochs, columns are names of metrics.
    # The expected column names are detailed in the metric_annotations dictionary.
    metrics: pd.DataFrame,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The number of epochs to skip in the beginning of the plot. Useful when the first epochs have very differen train/val losses.
    skip_first_n_epochs: int = 0,
    # Do you want to plot the metrics separately or combined?
    plot_types: List[Literal["separate", "combined"]] = ["separate", "combined"],
    # Should the rolling average of metrics be plotted instead? If so, what is the window size of the rolling average?
    rolling_window: int = None,
    # What path should the plots be saved to?
    save_path: str = None,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = True,
    # Should the plots be shown in the console?
    show: bool = False,
):

    # make sure not to modify the original dataframe
    metrics_copy = metrics.copy()

    # remove columns where every value is 0, setting weights for certain metrics to 0 is a common way to ignore that metric
    metrics_copy = metrics_copy.loc[:, (metrics_copy != 0).any(axis=0)]

    # These addendums are used to add information to the title and file name of the plots based on if certain epochs are
    # skipped or rolling averages are used.
    title_addendum = ""
    file_addendum = ""

    # Sometimes the first N epochs of a plot frame the rest of the data poorly
    if skip_first_n_epochs > 0:
        # If there are enough epochs to skip
        if len(metrics_copy) > skip_first_n_epochs:
            # Remove the first N epochs
            metrics_copy = metrics_copy.iloc[skip_first_n_epochs:, :]
            # Add information to the plot title
            title_addendum = f" (first {skip_first_n_epochs} epochs not shown)"

        # If there are not enough epochs to skip
        else:
            # Log that you couldn't skip epochs. Don't change the metrics dataframe.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Cannot skip first {skip_first_n_epochs} epochs in plot, not enough epochs to skip."

            log_and_print(prefix + message)

    # If a rolling average is used
    if rolling_window is not None:
        # If the rolling window is too large, set it to the maximum possible value
        if rolling_window > (len(metrics_copy) - 1):
            # Log that the rolling window is being set to the maximum possible value.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Rolling window of {rolling_window} too large, setting window to {max(len(metrics_copy) - 1, 1)} instead."

            log_and_print(prefix + message)

            # Set the rolling window to the maximum possible value
            rolling_window = max(len(metrics_copy) - 1, 1)

        # If the rolling window is valid, calculate the rolling average
        for metric in metrics_copy.columns:
            # Calculate the rolling average
            metrics_copy[metric] = (
                metrics_copy[metric].rolling(window=rolling_window).mean()
            )

        # Add information to the plot title and file name if a rolling average was used.
        title_addendum += f" (rolling average over {rolling_window} epochs)"
        file_addendum = f"_rolling"

    # Titles and colors for all the metrics we expect to see in the metrics dataframe
    metric_annotations = {
        # annotations for loss components
        "class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for training loss components
        "t_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "t_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "t_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "t_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for validation loss components
        "v_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "v_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "v_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "v_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        # annotations for general losses
        "conf_loss": {"title": "Confounder Adversarial Loss", "color": "blue"},
        "train_loss": {"title": "Total Training Loss", "color": "green"},
        "valid_loss": {"title": "Total Validation Loss", "color": "purple"},
        # annotations for confusion metrics
        "accuracy": {"title": "Accuracy", "color": "blue"},
        "precision": {"title": "Precision", "color": "orange"},
        "recall": {"title": "Recall", "color": "green"},
        "f1-score": {"title": "f1-Score", "color": "red"},
        # annotation for learning_rate
        "learning_rate": {"title": "Learning Rate", "color": "black"},
        "VAE_learning_rate": {"title": "VAE Learning Rate", "color": "black"},
        "adv_class_lr": {
            "title": "Adversarial Encoder Learning Rate",
            "color": "black",
        },
        "adv_encod_lr": {
            "title": "Adversarial Classifier Learning Rate",
            "color": "black",
        },
    }

    # If the "separate" plot was desired, plot all metrics separately
    if "separate" in plot_types:
        # collect relevant colors and title names
        titles = [
            metric_annotations[metric]["title"] for metric in metrics_copy.columns
        ]
        colors = [
            metric_annotations[metric]["color"] for metric in metrics_copy.columns
        ]

        # Each row of the plot will contain 4 subplots
        nrows = math.ceil(len(metrics_copy.columns) / 4)

        # Plot each metric in a separate subplot using pandas plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, ax = plt.subplots(figsize=(24, 6 * nrows))
            metrics_copy.plot(
                kind="line",
                subplots=True,
                xlabel="Epoch",
                ylabel="Loss",
                layout=(nrows, 4),
                grid=True,
                title=titles,
                color=colors,
                ax=ax,
                use_index=True,
            )

        # Set plot title and layout
        plt.suptitle("Loss Components Over Epochs" + title_addendum, fontsize=30)
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_metrics_separate{file_addendum}.png"
                )
            )
        if show:
            plt.show()

        # Close the plot (for stability)
        plt.close(fig)

    # If the "combined" plot was desired, plot all metrics in three subplots:
    # - one for the total training and validation loss together
    # - one for training loss components (KLD loss, recon loss, class loss, etc)
    # - one for validation loss components (KLD loss, recon loss, class loss, etc)
    if "combined" in plot_types:
        # Plot losses in one figure
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Get the toal valid and total train loss
        main_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("valid_")
            ]
        ]

        # Get the components of the train and valid losses
        train_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("t_")
            ]
        ]
        valid_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("valid_")
                | metrics_copy.columns.str.startswith("v_")
            ]
        ]

        # Plot the losses in the three subplots
        for i, model_losses, title in zip(
            range(3),
            [main_losses, train_losses, valid_losses],
            ["Combined", "Train", "Validation"],
        ):
            # Get the relevant colors and titles for the metrics
            colors = [
                metric_annotations[metric]["color"] for metric in model_losses.columns
            ]
            titles = [
                metric_annotations[metric]["title"] for metric in model_losses.columns
            ]

            # Plot the losses in the chosen subplot
            model_losses.plot(
                kind="line",
                xlabel="Epochs",
                ylabel="Losses",
                ax=ax[i],
                color=colors,
                label=titles,
                title=f"{title} Loss Over Epochs" + title_addendum,
                use_index=True,
            )

        # Set the plot title and layout
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_losses_combined{file_addendum}.png"
                )
            )
        if show:
            plt.show()

        plt.close(fig)


# Latent space plots are generated on model checkpoints to visualize the latent space of the model. The function can
# plot the latent space in PCA and UMAP, and can plot the latent space for different variables. The function can save
# the plots to a specified path, and can also show the plots in the console.
def latent_space_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    # These variables should be present in the data.obs dataframe, and should be categorical or continuous.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The type of data being plotted. Used for the plot title and file name.
    data_type: str,
    # The types of plots to generate. The function can generate PCA and UMAP plots. This functione expects the anndata to already
    plot_types: List[str] = ["umap", "pca"],
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
):
    # If PCA was specified, plot the latent space in PCA
    if "pca" in plot_types:
        # Plot latent space pca
        sc.pl.pca(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            palette=sns.color_palette(),
            show=show,
            return_fig=show,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca.png",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

    # If UMAP was specified, plot the latent space in UMAP
    if "umap" in plot_types:
        # Plot latent space umap
        sc.pl.umap(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            palette=sns.color_palette(),
            show=show,
            return_fig=show,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap.png",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

def latent_space_plot_2D(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    # These variables should be present in the data.obs dataframe, and should be categorical or continuous.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The type of data being plotted. Used for the plot title and file name.
    data_type: str,
    # Name of the atlas / model whose latent space is being plotted.
    atlas_name: str,
    # The types of plots to generate. The function can generate PCA and UMAP plots. This functione expects the anndata to already
    plot_types: List[str] = ["umap", "pca"],
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
):
    # If PCA was specified, plot the latent space in PCA
    if "pca" in plot_types:
        # Plot latent space pca
        sc.pl.pca(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            palette=sns.color_palette(),
            show=show,
            return_fig=show,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca_atlas-{atlas_name}.png",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

    # If UMAP was specified, plot the latent space in UMAP
    if "umap" in plot_types:
        # Plot latent space umap
        sc.pl.umap(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            palette=sns.color_palette(),
            show=show,
            return_fig=show,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap_atlas-{atlas_name}.png",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()


# This function plots the latent space of the model, broken down by diagnosis. Each diagnosis will be plotted in a separate
# column, and each row will be one of the datasets in the data. The first row is all the datasets combined, as an overview.
# The function can save the plots to a specified path, and can also show the plots in the console.
def latent_space_batch_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The type of data being plotted, such as "training" or "validation". Used for the plot title and file name.
    data_type: str,
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
    # A descriptor to add to the end of file name of the plot.
    descriptor: str = "",
    # The size of the points in the plot.
    size: int = 30,
):
    # Get the unique diagnoses and datasets in the data
    diagnoses = data.obs["Diagnosis"].unique()
    datasets = data.obs["Dataset"].unique()

    # The number of rows and columns in the plot, 1 column per diagnosis, 1 row per dataset +1 for all datasets combined
    nrows = len(datasets) + 1
    ncols = len(diagnoses)

    # Initialize the plot
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    # Assign a color to each dataset so that the palette is consistent across all subplots
    palette = {dataset: sns.color_palette()[i] for i, dataset in enumerate(datasets)}

    # For each diagnosis; dia is used for the column index, and diagnosis is the diagnosis name
    for dia, diagnosis in enumerate(diagnoses):

        # Get the data for only this diagnosis
        diagnosis_data = data[data.obs["Diagnosis"] == diagnosis].copy()

        # Plot all the data in this diagnosis
        sc.pl.umap(
            diagnosis_data,
            ax=axs[0, dia],
            title=f"{diagnosis} - All",
            color="Dataset",
            palette=palette,
            size=size,
            show=False,
            return_fig=False,
        )

        # For each dataset; dat is used for the row index, and dataset is the dataset name
        for dat, dataset in enumerate(data.obs["Dataset"].unique(), start=1):

            # Get the data for this dataset and diagnosis
            subset_data = data[
                (data.obs["Diagnosis"] == diagnosis) & (data.obs["Dataset"] == dataset)
            ].copy()

            # If there is no data for this dataset and diagnosis, skip this subplot
            if len(subset_data) == 0:
                # remove distracting axis
                axs[dat, dia].axis("off")
                continue

            # plot grey background of all the data in this diagnosis
            sc.pl.umap(
                diagnosis_data,
                ax=axs[dat, dia],
                show=False,
                return_fig=False,
                size=size,
            )

            # plot the data for this dataset and diagnosis in color
            sc.pl.umap(
                subset_data,
                ax=axs[dat, dia],
                title=f"{diagnosis} - {dataset}",
                color="Dataset",
                palette=palette,
                size=size,
                show=False,
                return_fig=False,
            )

    # Set the plot title and layout
    plt.suptitle(
        f"Batch Effect Breakdown of {data_type.capitalize()} Data (UMAP)",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.90)

    # Save if specified, show if specified
    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_batch_{data_type}{descriptor}.png",
            )
        )

    plt.show() if show else plt.close()


# This plot breaks down a set of variables by dataset in the latent space. Each variable passed through plot_by will be
# plotted in the latent space as a column. And then each row will be one of the datasets in the data. The first row is
# all the datasets combined, as an overview. The function can save the plots to a specified path, and can also show the
# plots in the console.
def latent_space_details_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The type of data being plotted, such as "training" or "validation". Used for the plot title and file name.
    data_type: str,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
    # A descriptor to add to the end of file name of the plot.
    descriptor: str = "",
    # The size of the points in the plot.
    size: int = 30,
):
    # One row for each dataset, +1 row for all the datasets combined
    nrows = len(data.obs["Dataset"].unique()) + 1
    # One column for each variable to plot by
    ncols = len(plot_by)
    # Initalize the plot
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    # Make sure every subplot has the same color palette
    palettes = {}
    # For each variable to plot by
    for variable in plot_by:
        # If the variable is categorical, assign a color palette to it (non-categorical variables will use the default palette)
        if data.obs[variable].dtype.name == "category":
            # Assign a color to each subgroup of the variable, and then by passing this palette later, each subgroup will have
            # the same color in every subplot
            palettes[variable] = {
                subgroup: sns.color_palette()[i]
                for i, subgroup in enumerate(data.obs[variable].unique())
            }

    # For each variable to plot by: c is used for the column index, and category is the variable name
    for c, category in enumerate(plot_by):
        # Is it a categorical variable? We want to know if we need to pass the palette from above.
        is_category = data.obs[category].dtype.name == "category"
        # Is the variable "Diagnosis"?
        is_diagnosis = category == "Diagnosis"
        # If it is DIagnosis, the legend should be on the data, otherwise on the right margin
        legend_loc = "on data" if is_diagnosis else "right margin"

        # Plot a grey background of the data
        sc.pl.umap(
            data,
            ax=axs[0, c],
            show=False,
            return_fig=False,
            size=size,
        )
        # Plot all the data, colored by this variable
        sc.pl.umap(
            data[data.obs["Data_Type"] == data_type].copy(),
            ax=axs[0, c],
            title=f"{category} - All",
            color=[category],
            show=False,
            return_fig=False,
            # pass the palette if this is one of the categorical variables
            palette=(palettes[category] if is_category else None),
            size=size,
            legend_loc=legend_loc,
            # For the "Age" variable we set a colorbar manually
            vmax=80 if category == "Age" else None,
            vmin=20 if category == "Age" else None,
        )

        # Then for each dataset, repeat the process. d is used for the row index, and dataset is the dataset name
        for d, dataset in enumerate(data.obs["Dataset"].unique(), start=1):
            # Grey background of all the data
            sc.pl.umap(
                data,
                ax=axs[d, c],
                show=False,
                return_fig=False,
                size=size,
            )
            # Foreground of this datatset's data, colored by this variable
            sc.pl.umap(
                data[
                    (data.obs["Data_Type"] == data_type)
                    & (data.obs["Dataset"] == dataset)
                ].copy(),
                ax=axs[d, c],
                title=f"{category} - {dataset}",
                color=[category],
                show=False,
                return_fig=False,
                # pass the palette if this is one of the categorical variables
                palette=palettes[category] if is_category else None,
                size=size,
                legend_loc=legend_loc,
                # For the "Age" variable we set a colorbar manually
                vmax=(80 if category == "Age" else None),
                vmin=(20 if category == "Age" else None),
            )

    # Set the plot title and layout
    plt.suptitle(
        f"Latent Space of {data_type.capitalize()} Data (UMAP)",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.90)

    # Save if specified, show if specified
    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_details_{data_type}{descriptor}.png",
            )
        )

    plt.show() if show else plt.close()


# This plot is rarely used in the code because it was superceded by the metrics plot. It plots accuracy, precision, recall,
# and f1-score over epochs into one plot. The function can only save the plots to a specified path.
def confusion_matrix_plot(
    # The metrics dictionary, containing the metrics of the model over epochs. The keys are the metric names, and the values
    # are lists of the metric values over epochs. The keys should contain "accuracy", "precision", "recall", and "f1-score".
    metrics: Dict[str, float],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plot.
    timestamp: str,
):
    # The keys of the metrics dictionary, and the colors and titles for each metric
    # Using zip we will iterate over the keys, colors, and titles at the same time
    metric_keys = ["accuracy", "precision", "recall", "f1-score"]
    colors = [
        "blue",
        "orange",
        "green",
        "red",
    ]
    titles = [
        "Accuracy",
        "Precision",
        "Recall",
        "f1-Score",
    ]

    # Initialize the plot
    fig, ax = plt.subplots(1, figsize=(12, 6))

    # Plot each metric with the appropriate color and title
    for i, (key, color, title) in enumerate(zip(metric_keys, colors, titles)):
        ax.plot(metrics[key], label=title, color=color)

    # Set the plot title and labels
    ax.set_title("Confusion Metrics Over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Metric Value")
    ax.legend()

    # Set the layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_path, f"{timestamp}_confusion_stats_.png"))

    # Close the plot
    plt.close(fig)


# This function takes a model and a data loader, and runs a few images from the data though the model, collecting the
# reconstructed images. The original and reconstructed images are then plotted side by side. The function only saves the
# plots to a specified path.
def recon_images_plot(
    # The data loader containing the Subjects that have mri images to plot.
    data_loader: DataLoader,
    # The VAE model to run the images through.
    model,
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plot.
    timestamp: str,
    # The current epoch of the model, used for the file name.
    epoch: int,
    # The gpu to run the model on.
    device: str,
    # The index of the slice to plot (i.e. at what depth to plot the brain). The slice is the same for all 3 angles.
    slice_index: int = 64,
    # The number of images to plot. The function will plot n different images from random batches of the data loader.
    n_images: int = 3,
    # The number of outputs the model should have. The function will check that the model has at least this many outputs.	
    # This first output should be the reconstructed image.
    n_model_outputs: int = 3,
):

    # Ensure the model is on the correct device
    model.to(device)

    # Initialize a dictionary to hold the original and reconstructed images
    image_dict = {}

    # For each image to plot
    for i in range(n_images):
        # Start going though the data loader
        for batch in data_loader:
            # Get the original image
            origi_image = batch["mri"][tio.DATA]

            # Move the image to the correct device
            origi_image = (
                origi_image.data[i].unsqueeze(0).to(device)
            )  

            # Keep redefining the image until we hit a random break
            # this gets us the i-th image of a random batch
            if rd.uniform(0, 1) < 0.2:
                break

        # Forward pass through the model
        output = model(origi_image)

        # Ensure the output is a tuple with the expected number of outputs
        if len(output) >= n_model_outputs:
            recon_image = output[0]
        else:
            # Raise an error if the model doesn't behave as expected
            raise ValueError(
                f"Expected at least {n_model_outputs} outputs, but got {len(output)}"
            )

        # For each image, number the original and the appropriate reconstruction, and add them to the dictionary as tio.ScalsrImages
        for (
            image_type,
            image,
        ) in zip([f"original_{i}", f"reconstructed_{i}"], [origi_image, recon_image]):
            
            # Move the image back to CPU for plotting
            image = image.data[0].cpu()  
            # Save the image as a tio.ScalarImage
            image = tio.ScalarImage(tensor=image)
            # Add the image to the dictionary
            image_dict[image_type] = image
    
    # Create one subject with all the original and reconstructed images as named maps
    subject = tio.Subject(image_dict)

    # Set the output path
    output_path = os.path.join(save_path, f"{timestamp}_e{epoch}_image_reconstruction")

    # Use the Subject.plot method to produce the plot
    subject.plot(
        output_path=output_path,
        indices=(slice_index, slice_index, slice_index),
        show=False,
        figsize=(6 * 3, 6 * n_images),
    )

# This function takes a model and a data loader, and runs a few images from the data though the model, collecting the
# reconstructed images. The original and reconstructed images are then plotted side by side. The function only saves the
# plots to a specified path. It is the same function as above, but adapted to the ClassCVAE model because it requires
# dataset information to be reconstruct images.
def recon_images_plot_ClassCVAE(
    # The data loader containing the Subjects that have mri images to plot. 
    data_loader: DataLoader,
    # The ClassCVAE model to run the images through.
    model,
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The current epoch of the model, used for the file name
    epoch: int,
    # The gpu to run the model on.
    device: str,
    slice_index: int = 64,
    n_images: int = 3,
    n_model_outputs: int = 3,
):

    # Ensure the model is on the correct device
    model.to(device)

    image_dict = {}

    for i in range(n_images):
        for batch in data_loader:
            origi_image = batch["mri"][tio.DATA]
            origi_cond = batch["Dataset"]

            # get the information from the first subject in the batch
            origi_image = origi_image.data[i].unsqueeze(0)
            origi_cond = origi_cond.data[i].unsqueeze(0)

            # Move the data to the correct device
            origi_image = origi_image.to(device)
            origi_cond = origi_cond.to(device)

            # Keep redefining the image / condition until we hit a random break
            # this gets us the i-th image of a random batch
            if rd.uniform(0, 1) < 0.2:
                break

        # Forward pass through the model
        output = model(origi_image, origi_cond)

        # Ensure the output is a tuple with the expected number of outputs
        if len(output) >= n_model_outputs:
            recon_image = output[0]
        else:
            raise ValueError(
                f"Expected at least {n_model_outputs} outputs, but got {len(output)}"
            )

        for (
            image_type,
            image,
        ) in zip([f"original_{i}", f"reconstructed_{i}"], [origi_image, recon_image]):
            image = image.data[0].cpu()  # Move the image back to CPU for plotting
            image = tio.ScalarImage(tensor=image)
            image_dict[image_type] = image

    subject = tio.Subject(image_dict)

    output_path = os.path.join(save_path, f"{timestamp}_e{epoch}_image_reconstruction")

    subject.plot(
        output_path=output_path,
        indices=(slice_index, slice_index, slice_index),
        show=False,
        figsize=(6 * 3, 6 * n_images),
    )


# This function plots the MRI data from a list of subjects
def plot_subjects(subjects: List[tio.Subject], index: int, slice: int):

    subject = subjects[index]
    print(subject["name"])
    subject.plot(indices=(slice, slice, slice))

    subjects, annotations = load_mri_data(
        "../data/processedcat12_histm", "../data/cat12_hm_valid.csv", covars=["Dataset"]
    )


# Loads mri data, applies up to N transforms, and plots the intensity histograms.
# The firs row of each different dataset in the annotations file is used as a subject.
# Transforms should be provided as "name of transform" : tio.Compose([transforms]).
def intensity_hist_plot(
    annotations: pd.DataFrame, data_path: str, transforms: Dict[str, tio.Compose]
):

    # load data
    subjects, annotations = load_mri_data(data_path, annotations, covars=["Dataset"])

    # get the indices of one subject from each dataset in annotations
    indices = []

    for dataset in annotations["Dataset"].unique():
        subset = annotations[annotations["Dataset"] == dataset]
        indices.append(subset.index[0])
        print(dataset, subset.index[0])

    # get our subplots ready
    nrows = len(indices)
    ncols = len(transforms)

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * nrows, 5 * ncols))

    # For each dataset, for each transform, plot the histogram
    for ax_row, subject_index in enumerate(indices):

        # get a subject from the list
        subject = subjects[subject_index]

        for (
            ax_col,
            title,
            transform,
        ) in enumerate(transforms.items()):

            # Transform the subject
            transformed_subject = transform(subject)

            # Get intensity values
            intensity_values = transformed_subject.mri.data.numpy().flatten()

            # Plot the histogram
            axs[ax_row, ax_col].hist(
                intensity_values, bins=100, color="blue", alpha=0.7
            )

            axs[ax_row, ax_col].set_title(title)
            axs[ax_row, ax_col].set_xlabel("Intensity")
            axs[ax_row, ax_col].set_ylabel("Frequency")
            axs[ax_row, ax_col].set_yscale("log")
            axs[ax_row, ax_col].set_ylim(1)

    plt.tight_layout()
    plt.show()

def plot_latent_space(latent_vectors, labels, save_path, method='tsne', title='Latent Space Visualization'):
    """
    Visualize the latent space using t-SNE or UMAP.
    
    Args:
        latent_vectors: Latent vectors from the VAE model
        labels: Labels for coloring points (e.g., diagnosis, datasets)
        save_path: Path to save the figure
        method: 'tsne' or 'umap'
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(latent_vectors.detach().cpu().numpy())
        plt.title(f"t-SNE {title}")
    else:  # umap
        reducer = umap.UMAP(random_state=42)
        reduced_data = reducer.fit_transform(latent_vectors.detach().cpu().numpy())
        plt.title(f"UMAP {title}")
    
    # Create scatter plot
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                    c=[cmap(i)], label=str(label), alpha=0.7)
    
    plt.colorbar(ticks=range(len(unique_labels)), 
                 label='Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return reduced_data

def plot_learning_curves(train_losses, val_losses, kl_losses, recon_losses, save_path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        kl_losses: List of KL divergence losses
        recon_losses: List of reconstruction losses
        save_path: Path to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # KL divergence loss
    plt.subplot(2, 2, 3)
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Log scale for total loss
    plt.subplot(2, 2, 4)
    plt.semilogy(train_losses, label='Training Loss (log scale)')
    plt.semilogy(val_losses, label='Validation Loss (log scale)')
    plt.title('Total Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bootstrap_metrics(bootstrap_metrics, save_path):
    """
    Plot metrics from bootstrap models.
    
    Args:
        bootstrap_metrics: List of dictionaries containing metrics for each bootstrap model
        save_path: Path to save the figure
    """
    df = pd.DataFrame(bootstrap_metrics)
    
    plt.figure(figsize=(15, 10))
    
    # Final loss distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['final_val_loss'], kde=True)
    plt.title('Distribution of Final Validation Loss')
    plt.xlabel('Validation Loss')
    plt.grid(True)
    
    # Final KL loss distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['final_kl_loss'], kde=True)
    plt.title('Distribution of Final KL Divergence Loss')
    plt.xlabel('KL Loss')
    plt.grid(True)
    
    # Final reconstruction loss distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['final_recon_loss'], kde=True)
    plt.title('Distribution of Final Reconstruction Loss')
    plt.xlabel('Reconstruction Loss')
    plt.grid(True)
    
    # Best epoch distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['best_epoch'], kde=False, discrete=True)
    plt.title('Distribution of Best Epochs')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

