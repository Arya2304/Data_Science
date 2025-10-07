from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def differential_expression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-test of expression_post vs expression_pre per gene across all patient/time/dose.
    Returns a DataFrame with gene_id, log2fc, t_stat, p_value, p_adj (BH-FDR).
    """
    # Aggregate per gene by patient/time/dose observations
    results = []
    for gene, sub in df.groupby("gene_id"):
        pre = sub["expression_pre"].astype(float)
        post = sub["expression_post"].astype(float)
        if len(pre) < 3:
            continue
        # paired t-test needs equal lengths; align on rows as-is (same sub)
        t_stat, p_val = stats.ttest_rel(post, pre, nan_policy="omit")
        log2fc = np.log2((post.mean() + 1.0) / (pre.mean() + 1.0))
        results.append({"gene_id": gene, "log2fc": log2fc, "t_stat": t_stat, "p_value": p_val})

    de = pd.DataFrame(results).dropna()
    if de.empty:
        return de

    # Benjamini-Hochberg FDR
    de = de.sort_values("p_value").reset_index(drop=True)
    m = len(de)
    ranks = np.arange(1, m + 1)
    de["p_adj"] = (de["p_value"] * m / ranks).clip(upper=1.0)
    de = de.sort_values("p_adj")
    return de


def top_biomarkers(de: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if de is None or de.empty:
        return pd.DataFrame(columns=["gene_id", "log2fc", "p_adj"])
    return de.nsmallest(k, "p_adj")[ ["gene_id", "log2fc", "p_adj"] ]


def biomarker_sensitivity_score(de: pd.DataFrame) -> pd.DataFrame:
    """Compute a heuristic sensitivity score: higher for large |log2fc| and small p_adj."""
    if de is None or de.empty:
        return pd.DataFrame(columns=["gene_id", "sensitivity_score"])
    de = de.copy()
    de["sensitivity_score"] = (np.abs(de["log2fc"]) * (1.0 - de["p_adj"].clip(0, 1))).astype(float)
    return de[["gene_id", "sensitivity_score"]].sort_values("sensitivity_score", ascending=False)


