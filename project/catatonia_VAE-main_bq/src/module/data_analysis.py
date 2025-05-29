import os
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap

# === INPUT DATEIEN FESTLEGEN ===
metadata_path = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/metadata_20250110/full_data_train_valid_test.csv"
data_h5_paths = [
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_cobra_t.h5",
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_lpba40_t.h5",
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_neuromorphometrics_t.h5",
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_suit_t.h5",
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_thalamic_nuclei_t.h5",
    "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/data/train_xml_data_t/Aggregated_thalamus_t.h5"
]

# === AUSGABEORDNER ===
output_dir = "/raid/bq_lduttenhofer/project/catatonia_VAE-main_bq/results_analysis"
os.makedirs(output_dir, exist_ok=True)

# === METADATEN LADEN ===
metadata_df = pd.read_csv(metadata_path, index_col=0)
data_overview = metadata_df.to_dict(orient="index")  # für Lookups später

# === 1. ALTER-HISTOGRAMME NACH DIAGNOSE ===
conditions = ["HC", "MDD", "SCHZ", "CTT"]
total_fig = make_subplots(rows=4, cols=1, subplot_titles=conditions)
total_fig.update_layout(height=800)

for idx, cond in enumerate(conditions):
    fig = px.histogram(
        data_frame=metadata_df[(metadata_df["Diagnosis"] == cond) & ~(metadata_df["Dataset"] == "NU")],
        x="Age"
    )
    total_fig.add_trace(fig.data[0], row=idx + 1, col=1)
    total_fig.update_yaxes(range=[0, 300], row=idx + 1, col=1)
    total_fig.update_xaxes(range=[0, 90], row=idx + 1, col=1)

total_fig.write_html(os.path.join(output_dir, "age_histograms_by_diagnosis.html"))

# === 2. BALKENDIAGRAMME GESCHLECHT NACH DIAGNOSE ===
total_fig_sexes = make_subplots(rows=1, cols=4, subplot_titles=conditions)
total_fig_sexes.update_layout(width=1000)

for idx, cond in enumerate(conditions):
    df_cond = metadata_df[metadata_df["Diagnosis"] == cond]
    sex_counts = df_cond.groupby("Sex").size().reset_index(name='Count')
    fig = px.bar(sex_counts, x="Sex", y="Count")
    total_fig_sexes.add_trace(fig.data[0], row=1, col=idx + 1)

total_fig_sexes.write_html(os.path.join(output_dir, "sex_counts_by_diagnosis.html"))

# === 3. BOX-PLOT ALTER NACH DIAGNOSE ===
filter_df = metadata_df[metadata_df["Dataset"] != "NU"]
categories = sorted(filter_df["Diagnosis"].unique())
fig = go.Figure()

for category in categories:
    fig.add_trace(go.Box(
        y=filter_df[filter_df["Diagnosis"] == category]["Age"],
        name=category,
        marker_color=px.colors.qualitative.Plotly[categories.index(category) % len(px.colors.qualitative.Plotly)]
    ))

fig.update_layout(
    yaxis=dict(range=[0, 90], title="Age"),
    xaxis=dict(title="Diagnosis"),
    boxmode='group'
)
fig.write_html(os.path.join(output_dir, "age_boxplot_by_diagnosis.html"))

# === 4. UMAP-PROJEKTION ===
def read_hdf5_to_df(path):
    df = pd.read_hdf(path)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.loc[:, (slice(None), "Vgm")]  # nur Vgm-Spalten
        df.columns = df.columns.droplevel(1)  # nur ROI-Namen behalten
    return df

dfs = [read_hdf5_to_df(path) for path in data_h5_paths]
names = [os.path.splitext(os.path.basename(path))[0] for path in data_h5_paths]

fig_umap, axs = plt.subplots(nrows=1, ncols=len(dfs), figsize=(5*len(dfs), 5))
if len(dfs) == 1:
    axs = [axs]

for idx, (df, name) in enumerate(zip(dfs, names)):
    reducer = umap.UMAP()
    clean_df = df.copy()
    filenames = clean_df.index.get_level_values(0)
    diagnoses = [data_overview.get(f, {}).get("Diagnosis", "Unknown") for f in filenames]

    color_map = {"HC": 0, "MDD": 1, "SCHZ": 2}
    colors = [sns.color_palette()[color_map.get(d, 3)] for d in diagnoses]

    embedding = reducer.fit_transform(clean_df.values)
    axs[idx].set_title(f'UMAP: {name}')
    axs[idx].scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5)
    axs[idx].set_aspect('equal')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='HC', markerfacecolor=sns.color_palette()[0], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='MDD', markerfacecolor=sns.color_palette()[1], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='SCHZ', markerfacecolor=sns.color_palette()[2], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Unknown', markerfacecolor=sns.color_palette()[3], markersize=8),
]
axs[0].legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "umap_projection.png"), dpi=300)
plt.close()

# === 5. ROI-BOXPLOTS (OPTIONAL) ===
def plot_distribution(paths: list, measurement: str):
    for path in paths: 
        df = read_hdf5_to_df(str(path))
        df_t = df.T
        rois = df_t.columns

        fig = go.Figure()
        for roi in rois:
            fig.add_trace(go.Box(y=df_t[roi], name=roi))

        fig.update_layout(
            boxmode='group',
            title=f"ROI-Verteilung: {os.path.basename(path)}",
            yaxis_title=measurement
        )
        fig.update_traces(boxpoints='outliers', jitter=0.3)

        filename = os.path.basename(path).replace(".h5", f"_{measurement}_boxplot.html")
        fig.write_html(os.path.join(output_dir, filename))

# Aktivieren, wenn gewünscht:
# plot_distribution(data_h5_paths, measurement="Vgm")