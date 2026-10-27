import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import numpy as np


@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_predictions_export.csv")

    # Try to add extra columns from Excel; fall back to defaults if missing
    try:
        customer_profile = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Customer Profile",
        )
        monthly_activity = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Monthly Activity",
        )
        churn_label = pd.read_excel(
            "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
            sheet_name="fb Churn Label",
            header=1,
        )

        # Derive SUPPORT_TICKETS and COMPLAINT_FLAG from activity/profile
        latest_activity = monthly_activity.sort_values("MONTH").groupby("CUSTOMER_ID").last().reset_index()
        df = df.merge(customer_profile, on="CUSTOMER_ID", how="left")
        df = df.merge(latest_activity, on="CUSTOMER_ID", how="left")
        df = df.merge(churn_label[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID", how="left")

        # Derive key drivers
        if "SUPPORT_TICKETS" in df.columns:
            df["SUPPORT_TICKETS"] = pd.to_numeric(df["SUPPORT_TICKETS"], errors="coerce").fillna(0)
        else:
            df["SUPPORT_TICKETS"] = 0

        if "COMPLAINT_FLAG" in df.columns:
            df["COMPLAINT_FLAG"] = pd.to_numeric(df["COMPLAINT_FLAG"], errors="coerce").fillna(0)
        else:
            df["COMPLAINT_FLAG"] = 0

    except FileNotFoundError:
        # Default columns if Excel not available
        df["SUPPORT_TICKETS"] = 0
        df["COMPLAINT_FLAG"] = 0

    return df


@st.cache_resource
def load_model():
    model_path = "models/churn_pipeline.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        return pipeline
    return None


def main():
    st.set_page_config(page_title="RetainIQ | Customer Churn Intelligence", layout="wide")

    st.title("RetainIQ | Customer Churn Intelligence")

    # Load data
    df = load_data()

    # Load model
    pipeline = load_model()

    # Sidebar
    st.sidebar.header("Navigation")
    view = st.sidebar.radio("Go to", ["Overview", "Metrics", "Charts", "Customer Risk", "Retention Strategy"])

    if view == "Overview":
        st.subheader("Customer Overview")
        st.write(f"Total customers: {len(df)}")

        st.dataframe(df.head(10), use_container_width=True)

        # Check missing values in important columns
        st.subheader("Missing Values in Key Columns")
        important_cols = ["CUSTOMER_ID", "CHURN_PROBABILITY", "PREDICTED_CHURNED", "RISK_LEVEL"]
        for col in important_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                st.write(f"- {col}: {missing} missing values")

        # Churn probability distribution
        st.subheader("Churn Probability Distribution")
        if "CHURN_PROBABILITY" in df.columns:
            st.write(f"Customers with predicted churn: {(df['PREDICTED_CHURNED'] == 1).sum()}")
            st.write(f"Customers not predicted to churn: {(df['PREDICTED_CHURNED'] == 0).sum()}")


    elif view == "Metrics":
        st.subheader("Key Performance Indicators")

        total_customers = len(df)
        predicted_high_risk = int((df["CHURN_PROBABILITY"] >= 0.5).sum())
        avg_churn_prob = df["CHURN_PROBABILITY"].mean()
        intervention_threshold = st.sidebar.slider("Intervention threshold", 0.0, 1.0, 0.5)
        customers_for_intervention = int((df["CHURN_PROBABILITY"] >= intervention_threshold).sum())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total customers", total_customers)
        with col2:
            st.metric("Predicted high-risk customers", predicted_high_risk)
        with col3:
            st.metric("Average predicted churn probability", f"{avg_churn_prob:.2%}")
        with col4:
            st.metric("Customers selected for intervention", customers_for_intervention)

        st.write("---")
        st.dataframe(df.head(20), use_container_width=True)

        # Model prediction info
        if pipeline is not None:
            st.sidebar.success("✅ Trained model loaded")
            st.info("Live model prediction available — input customer features below.")
        else:
            st.sidebar.warning("⚠️ No trained model found")
            st.info("Model pipeline not found. Dashboard uses prediction CSV. To enable live predictions, save the trained pipeline to `models/churn_pipeline.pkl`.")


    elif view == "Charts":
        st.subheader("Interactive Charts")

        # Risk distribution
        fig1 = px.pie(df, names="RISK_LEVEL", title="Customer Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        # Churn probability histogram
        fig2 = px.histogram(df, x="CHURN_PROBABILITY", nbins=20, title="Churn Probability Distribution")
        st.plotly_chart(fig2, use_container_width=True)

        # Predicted churn distribution
        fig3 = px.pie(df, names="PREDICTED_CHURNED", title="Predicted Churn Distribution",
                      labels={"0": "Not Churned", "1": "Churned"})
        st.plotly_chart(fig3, use_container_width=True)


    elif view == "Customer Risk":
        st.subheader("Customer Risk Explorer")

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

        display_cols = ["CUSTOMER_ID", "CHURN_PROBABILITY", "PREDICTED_CHURNED", "RISK_LEVEL",
                        "SUPPORT_TICKETS", "COMPLAINT_FLAG", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

        # Key driver summaries
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

        # CSV download button
        st.download_button(
            label="Download Filtered Predictions as CSV",
            data=filtered[available_cols].to_csv(index=False).encode("utf-8"),
            file_name="filtered_customer_risk.csv",
            mime="text/csv",
        )

        # Model prediction form
        st.write("---")
        st.subheader("Live Customer Prediction")
        st.caption("Enter customer features to predict churn using the trained model (if available).")

        if pipeline is not None:
            st.success("Model is available for live predictions")
            with st.form("customer_prediction_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    churn_prob = st.number_input("Churn Probability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                    risk_level = st.selectbox("Risk Level", 
                        ["Safe (No Action)", "Low Risk (Automated Email)", "Medium Risk (Targeted Offer)", "High Risk (High-Value Incentive)"])
                    support_tickets = st.number_input("Support Tickets", min_value=0, value=0, step=1)
                with col_b:
                    complaint_flag = st.number_input("Complaint Flag", min_value=0, max_value=1, value=0, step=1)
                    preferred_category = st.selectbox("Preferred Category", 
                        ["Frozen Foods", "Electronics", "Clothing", "Household", "Cosmetics"])
                    home_store_tier = st.selectbox("Home Store Price Tier", 
                        ["Value", "Standard", "Premium"])

                submitted = st.form_submit_button("Predict")
                if submitted:
                    # Prepare features for the model
                    # Encode risk level (simple mapping)
                    risk_map = {"Safe (No Action)": 0, "Low Risk (Automated Email)": 1, 
                                "Medium Risk (Targeted Offer)": 2, "High Risk (High-Value Incentive)": 3}
                    risk_encoded = risk_map.get(risk_level, 0)
                    
                    # Encode categorical features
                    cat_map = {"Frozen Foods": 0, "Electronics": 1, "Clothing": 2, "Household": 3, "Cosmetics": 4}
                    preferred_encoded = cat_map.get(preferred_category, 0)
                    
                    tier_map = {"Value": 0, "Standard": 1, "Premium": 2}
                    home_encoded = tier_map.get(home_store_tier, 0)
                    
                    features = [[churn_prob, risk_encoded, support_tickets, complaint_flag, preferred_encoded, home_encoded]]
                    
                    try:
                        pred_prob = m.predict_proba(features)[0][1]
                        pred_label = int(m.predict(features)[0])
                        risk_cat = "High Risk" if pred_prob >= 0.5 else "Low Risk"
                        
                        st.success(f"Predicted churn probability: {pred_prob:.2%}")
                        st.success(f"Predicted label: {'Churned' if pred_label == 1 else 'Not Churned'}")
                        st.info(f"Risk category: {risk_cat}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.info("No trained model loaded. Add the trained pipeline to `models/churn_pipeline.pkl` to enable live predictions.")
            st.caption("""
            To enable live predictions:
            1. Run the model training script to save the pipeline
            2. The pipeline expects features: CHURN_PROBABILITY (numeric) and RISK_LEVEL (categorical)
            3. The pipeline is saved via joblib at `models/churn_pipeline.pkl`
            """)


    elif view == "Retention Strategy":
        st.subheader("Retention Strategy Simulator")

        # Merge churn label if available (with fallback)
        try:
            churn_label = pd.read_excel(
                "data/FreshBasket_Loyalty_Churn_Dataset.xlsx",
                sheet_name="fb Churn Label",
                header=1,
            )
        except FileNotFoundError:
            churn_label = pd.DataFrame({
                "CUSTOMER_ID": df["CUSTOMER_ID"].values if "CUSTOMER_ID" in df.columns else range(1, len(df) + 1),
                "CHURNED": np.random.choice([0, 1], len(df), p=[0.7, 0.3]),
            })

        df_merged = df.merge(churn_label[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID", how="left")

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
        targeted = df_merged[df_merged["CHURN_PROBABILITY"] >= threshold]
        n_targeted = len(targeted)
        tp = int((targeted["PREDICTED_CHURNED"] == 1).sum())
        fp = int((targeted["PREDICTED_CHURNED"] == 0).sum())

        actual_churners = df_merged[df_merged["CHURNED"] == 1] if "CHURNED" in df_merged.columns else pd.DataFrame()
        missed = int(len(actual_churners[actual_churners["CHURN_PROBABILITY"] < threshold])) if not actual_churners.empty else 0

        if not actual_churners.empty:
            tp_actual = int(((targeted["PREDICTED_CHURNED"] == 1) & (targeted["CHURNED"] == 1)).sum())
            fp_actual = int(((targeted["PREDICTED_CHURNED"] == 1) & (targeted["CHURNED"] == 0)).sum())
        else:
            tp_actual = tp
            fp_actual = fp

        estimated_campaign_cost = n_targeted * cost_per_customer
        estimated_gross_value = tp_actual * success_rate * ltv
        estimated_net_value = estimated_gross_value - estimated_campaign_cost

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Customers targeted", n_targeted)
        with col2:
            st.metric("Estimated campaign cost ($)", f"{estimated_campaign_cost:.0f}")
        with col3:
            st.metric("Estimated gross value ($)", f"{estimated_gross_value:.0f}")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("True positives", tp_actual)
        with col5:
            st.metric("False positives", fp_actual)
        with col6:
            st.metric("Missed churners", missed)

        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated net value ($)", f"{estimated_net_value:.0f}", delta="scenario value, not guaranteed ROI")
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
            t_targeted = len(df_merged[df_merged["CHURN_PROBABILITY"] >= t])
            t_tp_actual = 0
            t_fp_actual = 0
            if "CHURNED" in df_merged.columns and not df_merged[df_merged["CHURNED"] == 1].empty:
                t_targeted_df = df_merged[df_merged["CHURN_PROBABILITY"] >= t]
                t_tp_actual = int(((t_targeted_df["PREDICTED_CHURNED"] == 1) & (t_targeted_df["CHURNED"] == 1)).sum())
                t_fp_actual = int(((t_targeted_df["PREDICTED_CHURNED"] == 1) & (t_targeted_df["CHURNED"] == 0)).sum())
            comparison.append({
                "Threshold": t,
                "Customers targeted": t_targeted,
                "Campaign cost ($)": t_targeted * cost_per_customer,
                "Gross value ($)": t_tp_actual * success_rate * ltv,
                "Net value ($)": t_tp_actual * success_rate * ltv - t_targeted * cost_per_customer,
                "True positives": t_tp_actual,
                "False positives": t_fp_actual,
                "Missed churners": int(len(df_merged[df_merged["CHURNED"] == 1][df_merged["CHURN_PROBABILITY"] < t])) if not df_merged[df_merged["CHURNED"] == 1].empty else 0,
            })
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()