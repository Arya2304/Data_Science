import os
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from scripts.preprocessing import (
    ensure_sample_dataset,
    load_dataset,
    preprocessing_pipeline,
)
from scripts.differential_expression import differential_expression, top_biomarkers, biomarker_sensitivity_score
from scripts.pathway_analysis import enrich_pathways
from scripts.ml_models import train_classifier, train_regressor, cluster_patients
from scripts.visualization import plot_volcano, heatmap_expression, dose_response_plot, pca_plot


st.set_page_config(page_title="Genomic Response Analysis to Nuclear Radiation", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load(uploaded_file):
    default_path = os.path.join("data", "sample_data.csv")
    return load_dataset(uploaded_file, default_path)


def write_report_csv(summary: dict) -> bytes:
    df = pd.DataFrame([summary])
    return df.to_csv(index=False).encode("utf-8")


def try_make_pdf(summary: dict) -> bytes:
    try:
        from fpdf import FPDF
    except Exception:
        return b""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Genomic Response Analysis Report", ln=True)
    pdf.ln(5)
    for k, v in summary.items():
        pdf.multi_cell(0, 8, txt=f"{k}: {v}")
    out = pdf.output(dest='S').encode('latin1')
    return out


def main():
    st.title("Genomic Response Analysis to Nuclear Radiation Exposure in Cancer Patients")
    st.markdown("Analyze radiation-driven genomic responses, discover biomarkers, and predict outcomes.")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])  # expects long-format columns
        use_sample = st.checkbox("Use sample dataset", value=True)
        test_size = st.slider("Train/Test split", 0.1, 0.5, 0.2, 0.05)
        clf_choice = st.selectbox("Classifier", ["Random Forest", "SVM"]) 
        reg_choice = st.selectbox("Regressor", ["XGBoost", "Linear Regression"]) 
        st.divider()
        st.caption("Tip: If no file uploaded, a synthetic dataset will be used.")

    # Load data
    default_path = os.path.join("data", "sample_data.csv")
    ensure_sample_dataset(default_path)
    df = cached_load(uploaded if not use_sample and uploaded is not None else None)

    # Home Page
    st.subheader("Home")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("Data preview:")
        st.dataframe(df.head(50))
    with c2:
        st.write("Summary:")
        st.json({
            "rows": len(df),
            "patients": int(df['patient_id'].nunique()),
            "genes": int(df['gene_id'].nunique()),
            "times": sorted(df['time'].unique().tolist()),
            "doses": sorted(np.unique(df['dose']).tolist()),
        })

    st.divider()
    st.subheader("Analysis")
    run_pre = st.button("Run Preprocessing")
    if run_pre:
        feats, scalers = preprocessing_pipeline(df)
        st.session_state["features"] = feats
        st.success(f"Preprocessing complete. Feature rows: {len(feats)}")
        st.dataframe(feats.head())

    if st.button("Run Differential Expression"):
        de = differential_expression(df)
        st.session_state["de"] = de
        st.write("Top genes:")
        st.dataframe(top_biomarkers(de, 10))
        fig = plot_volcano(de)
        st.pyplot(fig)

    st.divider()
    st.subheader("Machine Learning")
    if "features" in st.session_state:
        feats = st.session_state["features"]
        # Classification
        if st.button("Train Classifier"):
            res = train_classifier(feats, target_col="outcome", model_name=clf_choice, test_size=test_size)
            st.session_state["clf_res"] = res
            st.write({"Accuracy": res["acc"], "AUC": res["auc"], "CV Acc": res["cv_acc"]})
            st.line_chart(pd.DataFrame({"fpr": res["fpr"], "tpr": res["tpr"]}).set_index("fpr"))
            st.write("Confusion matrix:")
            st.dataframe(pd.DataFrame(res["cm"], columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
            if res["feature_importances"] is not None:
                st.bar_chart(res["feature_importances"].head(20))

        # Regression
        if st.button("Train Regressor"):
            resr = train_regressor(feats, target_col="mean_delta_expr", model_name=reg_choice, test_size=test_size)
            st.session_state["reg_res"] = resr
            st.write({"MSE": resr["mse"]})

        # Clustering + PCA
        if st.button("Cluster Patients"):
            labels, km = cluster_patients(feats, n_clusters=3)
            st.session_state["cluster_labels"] = labels
            st.success("Clustering done.")
            st.pyplot(pca_plot(feats, labels))
    else:
        st.info("Run preprocessing to enable ML.")

    st.divider()
    st.subheader("Visualization")
    colA, colB = st.columns(2)
    with colA:
        if "de" in st.session_state:
            de = st.session_state["de"]
            st.pyplot(heatmap_expression(df, de.sort_values("p_adj")["gene_id"]))
        else:
            st.pyplot(heatmap_expression(df))
    with colB:
        st.pyplot(dose_response_plot(df))

    st.divider()
    st.subheader("Pathway Enrichment and Biomarkers")
    if "de" in st.session_state:
        de = st.session_state["de"]
        top = top_biomarkers(de, 10)
        st.write("Top 10 biomarkers:")
        st.dataframe(top)
        st.write("Biomarker sensitivity score:")
        st.dataframe(biomarker_sensitivity_score(de).head(10))
        terms = enrich_pathways(top["gene_id"].tolist())
        st.write("Enrichment:")
        st.dataframe(terms)
    else:
        st.info("Run differential expression to see enrichment results.")

    st.divider()
    st.subheader("Prediction")
    if "clf_res" in st.session_state and "features" in st.session_state:
        res = st.session_state["clf_res"]
        feats = st.session_state["features"]
        st.write("Provide new patient features (aggregated):")
        col1, col2, col3 = st.columns(3)
        with col1:
            mean_expr_pre = st.number_input("mean_expr_pre", value=float(feats["mean_expr_pre"].median()))
            mean_expr_post = st.number_input("mean_expr_post", value=float(feats["mean_expr_post"].median()))
        with col2:
            mean_delta_expr = st.number_input("mean_delta_expr", value=float(feats["mean_delta_expr"].median()))
            mean_delta_meth = st.number_input("mean_delta_meth", value=float(feats["mean_delta_meth"].median()))
        with col3:
            total_mutations = st.number_input("total_mutations", value=float(feats["total_mutations"].median()))
            dose = st.number_input("dose", value=float(feats["dose"].median()))
        new_X = pd.DataFrame([
            {
                "mean_expr_pre": mean_expr_pre,
                "mean_expr_post": mean_expr_post,
                "mean_delta_expr": mean_delta_expr,
                "mean_delta_meth": mean_delta_meth,
                "total_mutations": total_mutations,
                "dose": dose,
                "time": float(feats["time"].median()),
            }
        ])
        # Align features to trained model's expected columns and order
        expected_cols = list(res["X_test"].columns)
        new_X = new_X.reindex(columns=expected_cols, fill_value=0.0)
        new_X = new_X.astype(float)
        proba = res["model"].predict_proba(new_X)[:, 1][0]
        st.metric("Predicted Radiation Sensitivity Probability", f"{proba:.2f}")
        st.caption("Recommendation: Higher probability suggests consider dose de-escalation; lower suggests escalation. Consult clinician.")
    else:
        st.info("Train classifier to enable predictions.")

    st.divider()
    st.subheader("Report Generation")
    metrics_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "rows": len(df),
        "patients": int(df['patient_id'].nunique()),
        "genes": int(df['gene_id'].nunique()),
    }
    if "clf_res" in st.session_state:
        metrics_summary.update({
            "clf_accuracy": float(st.session_state["clf_res"]["acc"]),
            "clf_auc": float(st.session_state["clf_res"]["auc"]),
        })
    csv_bytes = write_report_csv(metrics_summary)
    st.download_button("Download CSV Report", data=csv_bytes, file_name="analysis_report.csv", mime="text/csv")
    pdf_bytes = try_make_pdf(metrics_summary)
    if pdf_bytes:
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="analysis_report.pdf", mime="application/pdf")
    else:
        st.caption("Install 'fpdf' to enable PDF export.")


if __name__ == "__main__":
    main()


