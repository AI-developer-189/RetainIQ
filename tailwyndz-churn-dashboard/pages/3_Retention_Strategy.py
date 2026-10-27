import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_data():
    predictions = pd.read_csv("data/churn_predictions_export.csv")

    # Try to read Excel sheets; generate sample data if files are missing
    try:
        churn_label = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Churn Label",
            header=1,
        )
    except FileNotFoundError:
        churn_label = pd.DataFrame({
            "CUSTOMER_ID": range(1, 101),
            "CHURNED": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        })

    # Ensure CUSTOMER_ID and CHURNED columns exist
    if "CUSTOMER_ID" in churn_label.columns and "CHURNED" in churn_label.columns:
        churn_label["CHURNED"] = pd.to_numeric(churn_label["CHURNED"], errors="coerce")
    # Ensure CUSTOMER_ID and CHURNED columns exist
    if "CUSTOMER_ID" in churn_label.columns and "CHURNED" in churn_label.columns:
        churn_label["CHURNED"] = pd.to_numeric(churn_label["CHURNED"], errors="coerce")
    return predictions, churn_label


def compute_retention_metrics(df, threshold, cost_per_customer, success_rate, ltv):
    """Compute retention strategy metrics given assumptions."""
    # Merge churn label if available
    if "CHURNED" in df.columns:
        actual_churners = df[df["CHURNED"] == 1]
    else:
        actual_churners = pd.DataFrame()

    # Customers targeted above threshold
    targeted = df[df["CHURN_PROBABILITY"] >= threshold]
    n_targeted = len(targeted)

    # Predictedabelabelabel: based on PREDICTED_CHURNED
    tp = int((targeted["PREDICTED_CHURNED"] == 1).sum())  # predicted churners who are targeted
    fp = int((targeted["PREDICTED_CHURNED"] == 0).sum())  # predicted non-churners who are targeted

    # Missed churners: actual churners with probability below threshold
    missed = int(len(actual_churners[actual_churners["CHURN_PROBABILITY"] < threshold])) if not actual_churners.empty else 0

    # Actual TP/FP against labeled test data
    if not actual_churners.empty and "PREDICTED_CHURNED" in targeted.columns:
        tp_actual = int(((targeted["PREDICTED_CHURNED"] == 1) & (targeted["CHURNED"] == 1)).sum())
        fp_actual = int(((targeted["PREDICTED_CHURNED"] == 1) & (targeted["CHURNED"] == 0)).sum())
    else:
        tp_actual = tp
        fp_actual = fp

    # Estimated campaign cost
    estimated_campaign_cost = n_targeted * cost_per_customer

    # Estimated gross value (true positives * success_rate * ltv)
    estimated_gross_value = tp_actual * success_rate * ltv

    # Estimated net value
    estimated_net_value = estimated_gross_value - estimated_campaign_cost

    return {
        "n_targeted": n_targeted,
        "tp": tp,
        "fp": fp,
        "missed_churners": missed,
        "tp_actual": tp_actual,
        "fp_actual": fp_actual,
        "campaign_cost": estimated_campaign_cost,
        "gross_value": estimated_gross_value,
        "net_value": estimated_net_value,
    }


def main():
    st.set_page_config(page_title="Tailwyndz Propel | Retention Strategy Simulator", layout="wide")

    st.title("Tailwyndz Propel | Retention Strategy Simulator")

    predictions, churn_label = load_data()

    # Merge churn label into predictions
    df = predictions.merge(churn_label[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID", how="left")

    st.sidebar.header("Assumptions")

    # Churn threshold: 0.3–0.8
    threshold = st.sidebar.slider("Churn threshold", 0.0, 1.0, 0.5, step=0.01)

    # Cost per targeted customer
    cost_per_customer = st.sidebar.number_input("Cost per targeted customer ($)", min_value=0, value=500)

    # Assumed intervention success rate
    success_rate = st.sidebar.slider("Assumed intervention success rate", 0.0, 1.0, 0.15, step=0.01)

    # Assumed customer lifetime value
    ltv = st.sidebar.number_input("Assumed customer lifetime value ($)", min_value=0, value=500)

    st.write("---")

    # Compute metrics
    metrics = compute_retention_metrics(df, threshold, cost_per_customer, success_rate, ltv)

    # Display results
    st.subheader("Retention Scenario Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers targeted", metrics["n_targeted"])
    with col2:
        st.metric("Estimated campaign cost ($)", f"{metrics['campaign_cost']:.0f}")
    with col3:
        st.metric("Estimated gross value ($)", f"{metrics['gross_value']:.0f}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("True positives", metrics["tp_actual"])
    with col5:
        st.metric("False positives", metrics["fp_actual"])
    with col6:
        st.metric("Missed churners", metrics["missed_churners"])

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated net value ($)", f"{metrics['net_value']:.0f}", delta="scenario value, not guaranteed ROI")
    with col2:
        st.caption(
            "Assumptions: The 15% success rate and $500 LTV are assumptions, not measured campaign results. "
            "Financial output is an estimated scenario value."
        )

    # Compare three thresholds
    st.subheader("Scenario Comparison: Thresholds 0.3, 0.4, 0.7")
    thresholds = [0.3, 0.4, 0.7]
    comparison = []
    for t in thresholds:
        m = compute_retention_metrics(df, t, cost_per_customer, success_rate, ltv)
        comparison.append({
            "Threshold": t,
            "Customers targeted": m["n_targeted"],
            "Campaign cost ($)": m["campaign_cost"],
            "Gross value ($)": m["gross_value"],
            "Net value ($)": m["net_value"],
            "True positives": m["tp_actual"],
            "False positives": m["fp_actual"],
            "Missed churners": m["missed_churners"],
        })
    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()