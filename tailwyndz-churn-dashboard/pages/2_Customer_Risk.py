import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


@st.cache_data
def load_data():
    predictions = pd.read_csv("data/churn_predictions_export.csv")

    # Try to read Excel sheets; generate sample data if files are missing
    try:
        customer_profile = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Customer Profile"
        )
    except FileNotFoundError:
        customer_profile = pd.DataFrame({
            "CUSTOMER_ID": range(1, 101),
            "TENURE": np.random.randint(1, 60, 100),
            "PREFERRED_CATEGORY": np.random.choice(["Electronics", "Clothing", "Home", "Food"], 100),
            "HOME_STORE_PRICE_TIER": np.random.choice(["Budget", "Standard", "Premium"], 100),
        })

    try:
        monthly_activity = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Monthly Activity"
        )
    except FileNotFoundError:
        months = pd.date_range("2023-01-01", periods=12, freq="ME")
        monthly_activity = pd.DataFrame({
            "CUSTOMER_ID": np.repeat(range(1, 101), 12),
            "MONTH": np.tile(months, 100),
            "SPEND": np.random.uniform(20, 200, 100 * 12),
            "TRANSACTION_COUNT": np.random.randint(1, 10, 100 * 12),
        })

    try:
        churn_label = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Churn Label"
        )
    except FileNotFoundError:
        churn_label = pd.DataFrame({
            "CUSTOMER_ID": range(1, 101),
            "CHURNED": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        })

    # Rename columns for easier merging
    monthly_activity.columns = [
        col.strip().replace(" ", "_") for col in monthly_activity.columns
    ]
    customer_profile.columns = [
        col.strip().replace(" ", "_") for col in customer_profile.columns
    ]

    # Take the latest monthly activity per customer
    latest_activity = monthly_activity.sort_values("MONTH").groupby("CUSTOMER_ID").last().reset_index()

    # Merge: predictions + profile + latest activity
    df = predictions.merge(customer_profile, on="CUSTOMER_ID", how="left")
    df = df.merge(latest_activity, on="CUSTOMER_ID", how="left")
    df = df.merge(churn_label[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID", how="left")

    # Derive risk level from churn probability if not present
    if "RISK_LEVEL" not in df.columns:
        bins = [0, 0.3, 0.6, 1.0]
        labels = ["Safe (No Action)", "Low Risk (Automated Email)", "Medium Risk (Targeted Offer)"]
        df["RISK_LEVEL"] = pd.cut(df["CHURN_PROBABILITY"], bins=bins, labels=labels, include_lowest=True).astype(str)

    # Derive key drivers
    if "SUPPORT_TICKETS" in df.columns:
        df["SUPPORT_TICKETS"] = pd.to_numeric(df["SUPPORT_TICKETS"], errors="coerce").fillna(0)
    else:
        df["SUPPORT_TICKETS"] = 0

    if "COMPLAINT_FLAG" in df.columns:
        df["COMPLAINT_FLAG"] = pd.to_numeric(df["COMPLAINT_FLAG"], errors="coerce").fillna(0)
    else:
        df["COMPLAINT_FLAG"] = 0

    if "PREFERRED_CATEGORY" not in df.columns:
        df["PREFERRED_CATEGORY"] = "Unknown"
    if "HOME_STORE_PRICE_TIER" not in df.columns:
        df["HOME_STORE_PRICE_TIER"] = "Unknown"

    return df


def main():
    st.set_page_config(page_title="Tailwyndz Propel | Customer Risk Explorer", layout="wide")

    st.title("Tailwyndz Propel | Customer Risk Explorer")

    df = load_data()

    st.sidebar.header("Filters")

    # Search field for customer ID
    search_id = st.sidebar.text_input("Search Customer ID", "")

    # Risk-level selector
    risk_levels = ["All"] + sorted(df["RISK_LEVEL"].unique().tolist())
    selected_risk = st.sidebar.selectbox("Risk Level Filter", risk_levels)

    # Minimum churn-probability slider
    min_prob = float(df["CHURN_PROBABILITY"].min())
    max_prob = float(df["CHURN_PROBABILITY"].max())
    prob_threshold = st.sidebar.slider("Minimum Churn Probability", min_prob, max_prob, 0.0)

    # Apply filters
    filtered = df.copy()
    if search_id:
        filtered = filtered[filtered["CUSTOMER_ID"].astype(str).str.contains(search_id)]
    if selected_risk != "All":
        filtered = filtered[filtered["RISK_LEVEL"] == selected_risk]
    filtered = filtered[filtered["CHURN_PROBABILITY"] >= prob_threshold]

    # Sortable table
    st.subheader("Customer Risk Table")
    sort_col = st.radio("Sort by", ["CUSTOMER_ID", "CHURN_PROBABILITY", "PREDICTED_CHURNED", "RISK_LEVEL"])
    filtered = filtered.sort_values(by=sort_col, ascending=True)

    # Display columns
    display_cols = ["CUSTOMER_ID", "CHURN_PROBABILITY", "PREDICTED_CHURNED", "RISK_LEVEL",
                    "SUPPORT_TICKETS", "COMPLAINT_FLAG", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]

    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

    # Key driver summaries for displayed customers
    if len(filtered) > 0:
        st.subheader("Key Risk Drivers Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_prob = filtered["CHURN_PROBABILITY"].mean()
            st.metric("Avg Churn Probability", f"{avg_prob:.2%}")
        with col2:
            total_support = int(filtered["SUPPORT_TICKETS"].sum())
            st.metric("Total Support Tickets", total_support)
        with col3:
            total_complaints = int(filtered["COMPLAINT_FLAG"].sum())
            st.metric("Total Complaints", total_complaints)

    # File upload for custom spreadsheet
    st.sidebar.header("Upload Customer Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with customer data", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if "CUSTOMER_ID" in uploaded_df.columns:
                st.sidebar.success("File uploaded successfully!")
                st.write("Uploaded data preview:")
                st.dataframe(uploaded_df.head(10), use_container_width=True)
                # Merge uploaded data with existing data if CUSTOMER_ID matches
                if "CHURN_PROBABILITY" in uploaded_df.columns:
                    filtered = filtered.merge(
                        uploaded_df[["CUSTOMER_ID", "CHURN_PROBABILITY"]],
                        on="CUSTOMER_ID",
                        how="left"
                    )
                    st.write("Updated churn probabilities from uploaded data.")
        except Exception as e:
            st.sidebar.error(f"Error uploading file: {e}")

    # Visualization: Churn Distribution Pie Chart
    st.subheader("Churn Distribution")
    fig_pie = px.pie(
        filtered,
        names="CHURNED",
        title="Churn Distribution (0 = Active, 1 = Churned)",
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Visualization: Churn Probability Distribution (histogram)
    st.subheader("Churn Probability Distribution")
    fig_hist = px.histogram(
        filtered,
        x="CHURN_PROBABILITY",
        nbins=20,
        title="Churn Probability Distribution",
        labels={"CHURN_PROBABILITY": "Probability"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Visualization: Average Churn Probability by Risk Level (bar chart)
    st.subheader("Churn Probability by Risk Level")
    risk_order = ["Safe (No Action)", "Low Risk (Automated Email)", "Medium Risk (Targeted Offer)"]
    fig_line = px.bar(
        filtered,
        x="RISK_LEVEL",
        y="CHURN_PROBABILITY",
        title="Average Churn Probability by Risk Level",
        color="RISK_LEVEL",
        category_orders={"RISK_LEVEL": risk_order},
        labels={"CHURN_PROBABILITY": "Avg Churn Probability", "RISK_LEVEL": "Risk Level"},
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # CSV download button
    st.download_button(
        label="Download Filtered Predictions as CSV",
        data=filtered[available_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_risk.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()