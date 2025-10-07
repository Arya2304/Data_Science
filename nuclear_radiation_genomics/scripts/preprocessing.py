import os
import io
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def ensure_sample_dataset(path: str) -> str:
    """
    Create a synthetic multi-omics radiation dataset if it doesn't exist or is empty.

    Columns:
      - patient_id, time (h), dose (Gy), outcome (sensitive/resistant)
      - gene_id, expression_pre, expression_post
      - methylation_pre, methylation_post
      - mutation_count
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    should_generate = True
    if os.path.exists(path):
        try:
            if os.path.getsize(path) > 100:  # a few bytes indicates placeholder
                should_generate = False
        except OSError:
            should_generate = True

    if not should_generate:
        return path

    num_patients = 60
    patient_ids = [f"P{idx:03d}" for idx in range(1, num_patients + 1)]
    times = [0, 2, 6, 24]  # hours post-radiation
    doses = [0.0, 2.0, 4.0, 6.0]
    genes = [f"GENE{g:04d}" for g in range(1, 201)]  # 200 genes

    rows = []
    for pid in patient_ids:
        baseline_sensitivity = np.random.binomial(1, 0.5)
        outcome = "sensitive" if baseline_sensitivity == 1 else "resistant"
        patient_mutation_burden = np.random.poisson(5)
        for time in times:
            dose = float(np.random.choice(doses))
            # patient-level modulation of response
            dose_effect = 0.1 * dose * (1 if outcome == "sensitive" else -0.05)
            time_decay = np.exp(-time / 24.0)
            for gene in genes:
                # gene-specific baseline and response
                base_expr = np.random.normal(6.0, 0.8)
                noise = np.random.normal(0.0, 0.3)
                expr_pre = base_expr + noise
                # radiation induces changes depending on sensitivity, dose and time
                gene_radiation_effect = np.random.normal(0.0, 0.2) + dose_effect * time_decay
                expr_post = expr_pre + gene_radiation_effect

                # methylation inversely correlated with expression modestly
                meth_pre = np.clip(np.random.beta(2, 5), 0, 1)
                meth_post = np.clip(meth_pre + np.random.normal(-0.05 * dose_effect, 0.05), 0, 1)

                # mutation count per gene scaled from patient burden
                mut_count = max(0, int(np.random.poisson(0.1 + 0.02 * patient_mutation_burden)))

                rows.append(
                    {
                        "patient_id": pid,
                        "time": time,
                        "dose": dose,
                        "outcome": outcome,
                        "gene_id": gene,
                        "expression_pre": expr_pre,
                        "expression_post": expr_post,
                        "methylation_pre": meth_pre,
                        "methylation_post": meth_post,
                        "mutation_count": mut_count,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def load_dataset(
    file_or_buffer: Optional[io.BytesIO],
    default_path: str,
) -> pd.DataFrame:
    """Load CSV from upload buffer or default file path. Ensures sample dataset exists."""
    ensure_sample_dataset(default_path)
    if file_or_buffer is not None:
        return pd.read_csv(file_or_buffer)
    return pd.read_csv(default_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing via simple imputation."""
    df = df.copy()
    df = df.drop_duplicates()

    # Impute numerics with median, categoricals with mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
    return df


def log2_transform(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = np.log2(df[c].astype(float) + 1.0)
    return df


def normalize_expression(df: pd.DataFrame, cols: Tuple[str, ...]) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    df = df.copy()
    scalers: Dict[str, StandardScaler] = {}
    for c in cols:
        if c in df:
            scaler = StandardScaler()
            df[c] = scaler.fit_transform(df[[c]]).astype(float)
            scalers[c] = scaler
    return df, scalers


def encode_categoricals(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].astype("category").cat.codes
    return df


def integrate_modalities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a per-patient, per-time, per-dose feature set aggregating per-gene stats.
    Aggregations: mean expression_pre/post, delta expression, mean methylation delta, sum mutation_count.
    """
    df = df.copy()
    df["delta_expression"] = df["expression_post"] - df["expression_pre"]
    df["delta_methylation"] = df["methylation_post"] - df["methylation_pre"]

    aggregated = (
        df.groupby(["patient_id", "time", "dose", "outcome"], observed=True)
        .agg(
            mean_expr_pre=("expression_pre", "mean"),
            mean_expr_post=("expression_post", "mean"),
            mean_delta_expr=("delta_expression", "mean"),
            mean_delta_meth=("delta_methylation", "mean"),
            total_mutations=("mutation_count", "sum"),
        )
        .reset_index()
    )
    return aggregated


def preprocessing_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Full preprocessing for ML-ready features on aggregated data.
    """
    df_clean = clean_data(df)

    # Keep raw table for DE analysis; build aggregated features for ML
    agg = integrate_modalities(df_clean)
    agg = log2_transform(agg, ("mean_expr_pre", "mean_expr_post", "total_mutations"))
    agg, scalers = normalize_expression(agg, ("mean_expr_pre", "mean_expr_post", "mean_delta_expr", "mean_delta_meth", "total_mutations"))
    agg = encode_categoricals(agg, ("outcome",))
    return agg, scalers


