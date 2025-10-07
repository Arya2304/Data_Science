from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_volcano(de: pd.DataFrame, p_adj_threshold: float = 0.05):
    fig, ax = plt.subplots(figsize=(7, 5))
    if de is None or de.empty:
        ax.text(0.5, 0.5, "No DE results", ha="center", va="center")
        return fig
    x = de["log2fc"].astype(float)
    y = -np.log10(de["p_adj"].astype(float).clip(lower=1e-12))
    sig = de["p_adj"].astype(float) <= p_adj_threshold
    ax.scatter(x[~sig], y[~sig], s=12, alpha=0.6, color="#999999", label="NS")
    ax.scatter(x[sig], y[sig], s=14, alpha=0.8, color="#d62728", label="FDR<=0.05")
    ax.axhline(-np.log10(p_adj_threshold), color="#333", ls="--", lw=1)
    ax.set_xlabel("log2 Fold-Change")
    ax.set_ylabel("-log10(FDR)")
    ax.legend()
    ax.set_title("Volcano Plot")
    return fig


def heatmap_expression(df: pd.DataFrame, top_genes: Optional[pd.Series] = None, max_genes: int = 50):
    pivot = (
        df.groupby(["gene_id", "patient_id"])["expression_post"].mean().unstack(fill_value=0)
    )
    if top_genes is not None and not top_genes.empty:
        pivot = pivot.loc[pivot.index.intersection(top_genes.head(max_genes))]
    else:
        pivot = pivot.sample(min(max_genes, len(pivot)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("Gene Expression Heatmap (post-radiation)")
    return fig


def dose_response_plot(df: pd.DataFrame, gene: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if gene is not None:
        sub = df[df["gene_id"] == gene]
    else:
        sub = df.sample(min(2000, len(df)), random_state=42)
    sns.scatterplot(data=sub, x="dose", y="expression_post", hue="time", palette="viridis", ax=ax)
    ax.set_title(f"Dose-Response{' - ' + gene if gene else ''}")
    return fig


def pca_plot(features: pd.DataFrame, labels: Optional[np.ndarray] = None):
    X = features.drop(columns=["outcome"]) if "outcome" in features else features.copy()
    # restrict to numeric columns only (exclude identifiers like patient_id)
    X = X.select_dtypes(include=[np.number])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", ax=ax)
    else:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D)")
    return fig


