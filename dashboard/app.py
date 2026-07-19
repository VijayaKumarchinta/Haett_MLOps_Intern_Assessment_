"""
HaettMLOps — Churn Prediction Monitoring Dashboard
Streamlit dashboard for model monitoring, predictions analysis, SHAP explainability,
and drift detection. Acts as a visual companion to the FastAPI prediction service.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Haett Churn Monitor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #8B5CF6;
        --primary-dark: #6D28D9;
        --primary-glow: rgba(139,92,246,0.25);
        --primary-subtle: rgba(139,92,246,0.08);
        --bg-page: #0F0F1A;
        --bg-card: #1A1A2E;
        --bg-card-hover: #1F1F35;
        --text-primary: #E2E8F0;
        --text-secondary: #94A3B8;
        --text-tertiary: #64748B;
        --border-light: rgba(255,255,255,0.06);
        --border-medium: rgba(255,255,255,0.1);
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
        --shadow-md: 0 8px 24px rgba(0,0,0,0.3);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --font: 'Inter', -apple-system, sans-serif;
    }

    html { font-size: 16px; }
    body { font-family: var(--font); background: var(--bg-page); color: var(--text-primary); }
    .main { background: var(--bg-page); }
    .main > .block-container { max-width: 1400px; margin: 0 auto; padding-top: 1.5rem; }

    h1, h2, h3 { font-family: var(--font); letter-spacing: -0.02em; color: var(--text-primary); }
    .main h1 { font-size: 2rem; font-weight: 700; }
    .main h2 { font-size: 1.4rem; font-weight: 600; }
    .main h3 { font-size: 1.1rem; font-weight: 600; color: var(--text-secondary); }

    div[data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1rem;
        box-shadow: var(--shadow-sm);
        transition: transform 200ms, box-shadow 200ms;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    div[data-testid="metric-container"] label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: var(--text-tertiary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(255,255,255,0.03);
        padding: 0.35rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 0.4rem 0.9rem;
        font-weight: 500;
        font-size: 0.82rem;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: all 150ms;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139,92,246,0.08);
        color: var(--primary);
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
    }

    .stDataFrame {
        border-radius: var(--radius-md);
        border: 1px solid var(--border-light);
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background: #1A1A2E;
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.6rem 0.75rem;
        border: none;
    }
    .stDataFrame tbody td {
        padding: 0.45rem 0.75rem;
        border-bottom: 1px solid var(--border-light);
        color: var(--text-secondary);
    }
    .stDataFrame tbody tr:hover { background: rgba(139,92,246,0.06); }

    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--border-medium), transparent); margin: 1.5rem 0; }
    .stAlert { border-radius: var(--radius-sm); border: 1px solid var(--border-light) !important; }
    .stButton button {
        border-radius: var(--radius-sm);
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid var(--border-medium);
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 0.35rem 0.9rem;
        transition: all 150ms;
    }
    .stButton button:hover {
        border-color: var(--primary);
        color: var(--primary);
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary), #A78BFA) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 14px var(--primary-glow) !important;
    }
    .stSpinner > div { border-top-color: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="🔮 Loading model metadata...")
def load_model_metadata():
    """Load model metadata JSON if available."""
    from src.utils.config import MODELS_DIR
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)

@st.cache_data(show_spinner="📊 Loading feature importance...")
def load_feature_importance():
    """Load SHAP feature importance CSV if available."""
    from src.utils.config import MODELS_DIR
    csv_paths = list(MODELS_DIR.glob("shap_importance_*.csv"))
    if not csv_paths:
        csv_paths = list(MODELS_DIR.glob("feature_importance_*.csv"))
    if not csv_paths:
        return None
    df = pd.read_csv(csv_paths[0])
    return df.sort_values("shap_importance" if "shap_importance" in df.columns else "importance", ascending=False)

@st.cache_data(show_spinner="📈 Loading drift reports...", ttl=300)
def load_drift_reports():
    """Load recent drift monitoring reports."""
    from src.utils.config import MODELS_DIR
    from src.monitoring.drift_detection import list_drift_reports
    return list_drift_reports()

@st.cache_data(show_spinner="🧠 Loading model comparison...")
def load_model_comparison():
    """Check MLflow for run history if available."""
    from src.utils.config import MODELS_DIR
    # Look for the comparison artifacts saved during training
    return None  # We'll show from metadata

# ─── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="text-align:center;padding:1rem 0.5rem">
    <div style="width:48px;height:48px;margin:0 auto 0.75rem;
                background:linear-gradient(135deg,#8B5CF6,#A78BFA);
                border-radius:12px;display:flex;align-items:center;justify-content:center;
                box-shadow:0 8px 24px rgba(139,92,246,0.3);">
        <span style="font-size:1.5rem">🔮</span>
    </div>
    <h3 style="margin:0;color:#E2E8F0;font-weight:700;font-size:1rem">Haett Churn Monitor</h3>
    <p style="font-size:0.65rem;color:#64748B;margin-top:0.2rem">ML Monitoring Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# Load metadata
metadata = load_model_metadata()

if metadata:
    st.sidebar.markdown("### 🧠 Model Summary")
    st.sidebar.markdown(f"""
    - **Best Model:** `{metadata.get('best_model_name', 'N/A')}`
    - **F1 Score:** {metadata.get('metrics', {}).get('f1_score', 'N/A')}
    - **ROC-AUC:** {metadata.get('metrics', {}).get('roc_auc', 'N/A')}
    - **PR-AUC:** {metadata.get('metrics', {}).get('pr_auc', 'N/A')}
    - **Threshold:** {metadata.get('optimal_threshold', 'N/A')}
    - **Features:** {metadata.get('n_features', 'N/A')}
    - **Train Samples:** {metadata.get('n_train', 'N/A')}
    """)

    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Model Info")
    st.sidebar.markdown(f"""
    - **Tuning:** {'✅' if metadata.get('hparams_tuning') else '❌'}
    - **Calibration:** {'✅' if metadata.get('calibration_enabled') else '❌'}
    - **Method:** {metadata.get('calibration_method', 'N/A')}
    - **Class Balance:** {metadata.get('class_balance', 'N/A'):.1%}
    """)
else:
    st.sidebar.warning("⚠️ No trained model found. Run the pipeline first:\n\n```\npython run.py\n```")

st.sidebar.divider()

# Drift report list
drift_reports = load_drift_reports()
if drift_reports is not None and len(drift_reports) > 0:
    st.sidebar.markdown("### 📡 Drift Reports")
    for _, row in drift_reports.head(5).iterrows():
        status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(row.get("status", ""), "❓")
        st.sidebar.markdown(f"{status_icon} **{row.get('report', '')}**  \n  {row.get('timestamp', '')}")
else:
    st.sidebar.markdown("### 📡 Drift Reports")
    st.sidebar.caption("No drift reports yet. Run drift detection after collecting new data.")

st.sidebar.divider()
st.sidebar.caption("Haett Churn Prediction v1.0.0")

# ─── Main Content ─────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div style="margin-bottom:1.5rem;padding:1.5rem;
            background:linear-gradient(135deg,#1A1A2E 0%,#16213E 100%);
            border-radius:12px;border:1px solid rgba(255,255,255,0.05);">
    <h1 style="margin:0;font-size:1.8rem;color:#E2E8F0;">🔮 Churn Prediction Monitor</h1>
    <p style="color:#94A3B8;margin:0.3rem 0 0 0;font-size:0.9rem;">
        Real-time model monitoring · Prediction analysis · SHAP explainability · Drift detection
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "📈 Prediction Analysis",
    "🧠 SHAP Explainability",
    "📡 Drift Monitoring",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Model Performance Overview")

    if metadata:
        metrics = metadata.get("metrics", {})
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}",
                      help="Area Under ROC Curve — 1.0 = perfect, 0.5 = random")
        with col2:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}",
                      help="Harmonic mean of precision and recall")
        with col3:
            st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}",
                      help="Precision-Recall AUC — better for imbalanced data")
        with col4:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}",
                      help="Of predicted churners, how many actually churned")
        with col5:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}",
                      help="Of actual churners, how many were caught")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}",
                      help="Overall correct predictions")
        with col2:
            st.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}",
                      help="Probability calibration (lower = better)")
        with col3:
            st.metric("Lift@10%", f"{metrics.get('lift_at_10pct', 0):.2f}x",
                      help="How much better than random at top 10% highest risk")
        with col4:
            st.metric("Lift@20%", f"{metrics.get('lift_at_20pct', 0):.2f}x",
                      help="How much better than random at top 20% highest risk")
        with col5:
            st.metric("Optimal Threshold", f"{metadata.get('optimal_threshold', 0.5):.3f}",
                      help="F1-maximizing decision threshold")

        # Model comparison bar chart
        st.markdown("### 📋 Model Comparison")

        # Build comparison from metadata
        comparison = pd.DataFrame([
            {"Model": metadata.get("best_model_name", "Best"), **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        ])

        # Try to find MLflow runs for more models
        fig_compare = px.bar(
            comparison.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model",
            barmode="group", title="Best Model Metrics",
            color_discrete_sequence=["#8B5CF6"],
        )
        fig_compare.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color="#94A3B8"),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_compare.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>")
        st.plotly_chart(fig_compare, use_container_width=True)

        # Feature info
        st.markdown("### 🧩 Feature Set")
        if metadata.get("n_features"):
            st.info(f"Model trained on **{metadata['n_features']} features** with "
                    f"{metadata.get('n_train', 0)} train / {metadata.get('n_val', 0)} val / "
                    f"{metadata.get('n_test', 0)} test samples. "
                    f"Class balance: {metadata.get('class_balance', 0):.1%} churn rate.")
    else:
        st.warning("No model metadata found. Run the pipeline first:\n\n```\npython run.py\n```")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Prediction Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🧪 Test Predictions & Scenario Explorer")

    st.markdown("""
    <p style="color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;">
        Adjust user features below to simulate different user profiles and see how the model predicts their churn risk.
    </p>
    """, unsafe_allow_html=True)

    # ── Auto-fill: when scenario changes, reset all number inputs ──
    # Map feature names to their widget keys + bounds for clamping
    SCENARIO_WIDGET_MAP = {
        "days_since_last_order":       ("dslo", 0, 365, int),
        "tenure_days":                 ("ten", 0, 1000, int),
        "total_orders":                ("to", 0, 200, int),
        "std_days_between_orders":     ("std", 0.0, 1000.0, float),
        "orders_last_30_days":         ("o30", 0, 50, int),
        "avg_order_value":             ("aov", 0.0, 200.0, float),
        "avg_rating":                  ("rat", 1.0, 5.0, float),
        "coupon_usage_count":          ("cc", 0, 50, int),
        "coupon_usage_rate":           ("cr", 0.0, 1.0, float),
        "n_plan_changes":              ("pc", 0, 20, int),
        "monthly_price":               ("mp", 0.0, 200.0, float),
        "subscription_tenure_days":    ("st", 0, 1000, int),
        "avg_app_logins":              ("al", 0.0, 30.0, float),
        "avg_meals_skipped":           ("ms", 0.0, 10.0, float),
        "total_support_tickets":       ("stt", 0, 50, int),
        "age":                         ("age", 18, 100, int),
        # "age_group_code" excluded — handled by selectbox index param directly
    }

    # User scenario selector
    scenario = st.selectbox(
        "Quick Scenarios:",
        options=["Custom", "Low Risk — Power User", "Medium Risk — Disengaged", "Medium Risk — New User", "High Risk — At Risk"],
        index=0,
        key="scenario_select",
    )

    # Track previous scenario to detect changes
    if "_prev_scenario" not in st.session_state:
        st.session_state._prev_scenario = scenario

    scenario_defaults = {
        "Custom": {},
        "Low Risk — Power User": {
            "days_since_last_order": 1.0, "tenure_days": 600, "total_orders": 80,
            "std_days_between_orders": 2.5, "orders_last_30_days": 10, "avg_order_value": 45.0,
            "avg_rating": 4.8, "coupon_usage_count": 0, "coupon_usage_rate": 0.0,
            "n_plan_changes": 0, "monthly_price": 99.99, "subscription_tenure_days": 580,
            "avg_app_logins": 15.0, "avg_meals_skipped": 0.0, "total_support_tickets": 0,
            "age": 45, "age_group_code": 2,
        },
        "Medium Risk — Disengaged": {
            "days_since_last_order": 60.0, "tenure_days": 75, "total_orders": 5,
            "std_days_between_orders": 15.0, "orders_last_30_days": 0, "avg_order_value": 28.0,
            "avg_rating": 2.1, "coupon_usage_count": 2, "coupon_usage_rate": 0.4,
            "n_plan_changes": 3, "monthly_price": 29.99, "subscription_tenure_days": 30,
            "avg_app_logins": 0.5, "avg_meals_skipped": 3.0, "total_support_tickets": 8,
            "age": 22, "age_group_code": 0,
        },
        "Medium Risk — New User": {
            "days_since_last_order": 10.0, "tenure_days": 15, "total_orders": 2,
            "std_days_between_orders": 999.0, "orders_last_30_days": 1, "avg_order_value": 22.0,
            "avg_rating": 2.5, "coupon_usage_count": 1, "coupon_usage_rate": 0.5,
            "n_plan_changes": 1, "monthly_price": 19.99, "subscription_tenure_days": 5,
            "avg_app_logins": 0.2, "avg_meals_skipped": 5.0, "total_support_tickets": 4,
            "age": 20, "age_group_code": 0,
        },
        "High Risk — At Risk": {
            "days_since_last_order": 90.0, "tenure_days": 120, "total_orders": 3,
            "std_days_between_orders": 30.0, "orders_last_30_days": 0, "avg_order_value": 18.0,
            "avg_rating": 1.5, "coupon_usage_count": 5, "coupon_usage_rate": 0.8,
            "n_plan_changes": 4, "monthly_price": 15.99, "subscription_tenure_days": 15,
            "avg_app_logins": 0.1, "avg_meals_skipped": 6.0, "total_support_tickets": 12,
            "age": 19, "age_group_code": 0,
        },
    }

    defaults = scenario_defaults.get(scenario, {})

    # ── Auto-fill: reset all widget values when scenario changes ──
    if scenario != st.session_state._prev_scenario:
        st.session_state._prev_scenario = scenario
        for feature_name, value in defaults.items():
            if feature_name in SCENARIO_WIDGET_MAP:
                key, min_val, max_val, cast_type = SCENARIO_WIDGET_MAP[feature_name]
                if key in st.session_state:
                    clamped = max(min_val, min(cast_type(value), max_val))
                    st.session_state[key] = clamped
        st.rerun()

    # Helper: enforce that default value stays within [min_val, max_val]
    def _clamp(val, min_val, max_val):
        return max(min_val, min(val, max_val))

    col1, col2, col3 = st.columns(3)
    with col1:
        days_since = st.number_input("Days Since Last Order", 0, 365, _clamp(int(defaults.get("days_since_last_order", 10)), 0, 365), key="dslo")
        tenure = st.number_input("Account Tenure (days)", 0, 1000, _clamp(int(defaults.get("tenure_days", 100)), 0, 1000), key="ten")
        total_orders = st.number_input("Total Orders", 0, 200, _clamp(int(defaults.get("total_orders", 10)), 0, 200), key="to")
        std_days = st.number_input("Order Consistency (std dev)", 0.0, 1000.0, _clamp(float(defaults.get("std_days_between_orders", 5.0)), 0.0, 1000.0), key="std")
        orders_30 = st.number_input("Orders in Last 30 Days", 0, 50, _clamp(int(defaults.get("orders_last_30_days", 2)), 0, 50), key="o30")

    with col2:
        avg_order = st.number_input("Avg Order Value ($)", 0.0, 200.0, _clamp(float(defaults.get("avg_order_value", 35.0)), 0.0, 200.0), key="aov")
        rating = st.slider("Avg Rating (1-5)", 1.0, 5.0, _clamp(float(defaults.get("avg_rating", 3.5)), 1.0, 5.0), 0.1, key="rat")
        coupon_count = st.number_input("Coupon Usage Count", 0, 50, _clamp(int(defaults.get("coupon_usage_count", 0)), 0, 50), key="cc")
        coupon_rate = st.slider("Coupon Usage Rate", 0.0, 1.0, _clamp(float(defaults.get("coupon_usage_rate", 0.0)), 0.0, 1.0), 0.05, key="cr")
        plan_changes = st.number_input("Plan Changes", 0, 20, _clamp(int(defaults.get("n_plan_changes", 0)), 0, 20), key="pc")

    with col3:
        monthly_price = st.number_input("Monthly Price ($)", 0.0, 200.0, _clamp(float(defaults.get("monthly_price", 49.99)), 0.0, 200.0), key="mp")
        sub_tenure = st.number_input("Subscription Tenure (days)", 0, 1000, _clamp(int(defaults.get("subscription_tenure_days", 60)), 0, 1000), key="st")
        app_logins = st.slider("App Logins/Week", 0.0, 30.0, _clamp(float(defaults.get("avg_app_logins", 5.0)), 0.0, 30.0), 0.5, key="al")
        meals_skipped = st.slider("Meals Skipped/Week", 0.0, 10.0, _clamp(float(defaults.get("avg_meals_skipped", 1.0)), 0.0, 10.0), 0.5, key="ms")
        support_tickets = st.number_input("Support Tickets", 0, 50, _clamp(int(defaults.get("total_support_tickets", 0)), 0, 50), key="stt")
        age = st.slider("Age", 18, 100, _clamp(int(defaults.get("age", 30)), 18, 100), 1, key="age")
        age_group = st.selectbox("Age Group", ["Young Adult", "Adult", "Middle Age", "Senior"],
                                 index=defaults.get("age_group_code", 1), key="ag")

    age_group_map = {"Young Adult": 0, "Adult": 1, "Middle Age": 2, "Senior": 3}

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            try:
                from src.models.predict import get_predictor
                predictor = get_predictor()

                features = pd.DataFrame([{
                    "days_since_last_order": days_since,
                    "tenure_days": tenure,
                    "total_orders": total_orders,
                    "std_days_between_orders": std_days,
                    "orders_last_30_days": orders_30,
                    "avg_order_value": avg_order,
                    "avg_rating": rating,
                    "coupon_usage_count": coupon_count,
                    "coupon_usage_rate": coupon_rate,
                    "n_plan_changes": plan_changes,
                    "monthly_price": monthly_price,
                    "subscription_tenure_days": sub_tenure,
                    "avg_app_logins": app_logins,
                    "avg_meals_skipped": meals_skipped,
                    "total_support_tickets": support_tickets,
                    "age": age,
                    "age_group_code": age_group_map[age_group],
                }])

                # Prepare features
                features_ordered = predictor.prepare_features(features)
                result = predictor.predict(features_ordered, explain=True)

                prob = result["churn_probability"]
                risk = result["risk_level"]
                rec = result["business_recommendation"]
                explanations = result.get("explanations", [])

                # Risk gauge
                risk_color = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}.get(risk, "#94A3B8")

                col1, col2 = st.columns([1, 2])
                with col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        title={"text": f"Churn Risk — {risk}", "font": {"size": 16, "color": risk_color}},
                        number={"suffix": "%", "font": {"size": 28, "color": risk_color}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                            "bar": {"color": risk_color},
                            "steps": [
                                {"range": [0, 10], "color": "rgba(16,185,129,0.15)"},
                                {"range": [10, 30], "color": "rgba(245,158,11,0.15)"},
                                {"range": [30, 70], "color": "rgba(239,68,68,0.1)"},
                                {"range": [70, 100], "color": "rgba(220,38,38,0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": risk_color, "width": 4},
                                "thickness": 0.75,
                                "value": prob * 100,
                            },
                        },
                    ))
                    fig_gauge.update_layout(
                        height=300,
                        margin=dict(t=40, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={"color": "#94A3B8"},
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Risk level badge
                    st.markdown(f"""
                    <div style="text-align:center;padding:0.75rem;border-radius:8px;
                                background:{risk_color}15;border:1px solid {risk_color}30;">
                        <span style="font-size:1.2rem;font-weight:700;color:{risk_color};">
                            {risk.upper()} RISK
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 💡 Business Recommendation")
                    st.info(rec, icon="💡")

                    if explanations:
                        st.markdown("#### 🔍 Top Feature Contributions")
                        expl_df = pd.DataFrame(explanations)
                        expl_df["direction"] = expl_df["impact"].apply(
                            lambda x: "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk"
                        )
                        expl_df["impact_display"] = expl_df["impact"].apply(lambda x: f"{x:+.4f}")

                        fig_shap = px.bar(
                            expl_df.sort_values("impact"),
                            x="impact", y="feature",
                            color="direction",
                            orientation="h",
                            color_discrete_map={
                                "⬆️ Increases Risk": "#EF4444",
                                "⬇️ Decreases Risk": "#10B981",
                            },
                            title="SHAP Feature Impact on Prediction",
                            labels={"impact": "Impact on Churn Probability", "feature": ""},
                        )
                        fig_shap.update_layout(
                            height=300,
                            showlegend=True,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=11, color="#94A3B8"),
                            xaxis_title="Impact on Churn Probability",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure the model is trained. Run `python run.py` first.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Global SHAP Feature Importance")

    st.markdown("""
    <p style="color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;">
        SHAP (SHapley Additive exPlanations) shows which features most influence
        the model's churn predictions globally across all users.
        <strong>Higher importance</strong> = feature has more influence on predictions.
    </p>
    """, unsafe_allow_html=True)

    # Load feature importance
    fi_df = load_feature_importance()

    if fi_df is not None and len(fi_df) > 0:
        importance_col = "shap_importance" if "shap_importance" in fi_df.columns else "importance"

        top_n = st.slider("Number of features to show:", 5, min(30, len(fi_df)), 15, key="shap_top_n")

        top_features = fi_df.head(top_n)

        fig_shap_global = px.bar(
            top_features,
            y="feature",
            x=importance_col,
            orientation="h",
            color=importance_col,
            color_continuous_scale="Purples",
            title=f"Top {top_n} Most Important Features (Global SHAP)",
            labels={importance_col: "Mean |SHAP Value|", "feature": ""},
        )
        fig_shap_global.update_layout(
            height=max(300, top_n * 25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color="#94A3B8"),
            coloraxis_showscale=False,
            xaxis_title="Mean Absolute SHAP Value",
            yaxis=dict(autorange="reversed"),
        )
        fig_shap_global.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            marker=dict(line=dict(color='rgba(139,92,246,0.3)', width=1)),
        )
        st.plotly_chart(fig_shap_global, use_container_width=True)

        # Feature value distribution
        st.markdown("### 📊 Feature Value Distributions")

        selected_feature = st.selectbox(
            "Select a feature to see its distribution:",
            options=top_features["feature"].tolist(),
            index=0,
            key="shap_feat_select",
        )

        # Show feature info from the signal map
        from src.utils.metrics import _FEATURE_SIGNAL_MAP
        feat_info = _FEATURE_SIGNAL_MAP.get(selected_feature, None)
        if feat_info:
            st.info(f"**{feat_info['label']}**: {feat_info['signal'].format(value=50)}", icon="ℹ️")

        st.markdown(f"<p style='color:#94A3B8;font-size:0.8rem;'>"
                    f"SHAP Importance: {top_features[top_features['feature'] == selected_feature][importance_col].values[0]:.4f}"
                    f"</p>", unsafe_allow_html=True)

        # Feature description table
        st.markdown("### 📋 Feature Legend")
        feature_legends = []
        for feat in top_features["feature"].tolist():
            info = _FEATURE_SIGNAL_MAP.get(feat, {})
            feature_legends.append({
                "Feature": feat,
                "Description": info.get("label", feat.replace("_", " ").title()),
                "Signal": info.get("signal", "—").format(value=50) if info.get("signal") else "—",
            })

        legend_df = pd.DataFrame(feature_legends)
        st.dataframe(legend_df, use_container_width=True, hide_index=True)

    else:
        st.warning("No SHAP importance data found. Run the full pipeline first:\n\n```\npython run.py\n```")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Drift Monitoring
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📡 Data Drift Monitoring")

    st.markdown("""
    <p style="color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;">
        Data drift monitors detect when the distribution of incoming features
        differs significantly from the training data — which can cause model
        performance degradation over time.
    </p>
    """, unsafe_allow_html=True)

    drift_reports = load_drift_reports()

    if drift_reports is not None and len(drift_reports) > 0:
        # Summary metrics
        latest_status = drift_reports.iloc[0].get("status", "unknown")
        status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(latest_status, "❓")
        latest_drift = drift_reports.iloc[0].get("drift_share", 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Status", f"{status_icon} {latest_status.upper()}")
        with col2:
            st.metric("Drift Share", f"{latest_drift:.1%}",
                      help="Fraction of features that have drifted")
        with col3:
            drifted = drift_reports.iloc[0].get("n_drifted", 0)
            st.metric("Drifted Features", f"{drifted}/{drift_reports.iloc[0].get('n_features', 0)}")
        with col4:
            st.metric("Reports Generated", f"{len(drift_reports)}")

        # Drift history chart
        st.markdown("### 📈 Drift History")

        drift_reports["drift_share"] = pd.to_numeric(drift_reports["drift_share"], errors="coerce")
        drift_reports["timestamp_dt"] = pd.to_datetime(drift_reports["timestamp"], errors="coerce")

        fig_drift = px.line(
            drift_reports.sort_values("timestamp_dt"),
            x="timestamp", y="drift_share",
            markers=True,
            color_discrete_sequence=["#8B5CF6"],
            title="Drift Share Over Time",
            labels={"timestamp": "Report Date", "drift_share": "Fraction of Drifted Features"},
            range_y=[0, 1],
        )
        fig_drift.add_hline(y=0.15, line_dash="dash", line_color="#F59E0B",
                            annotation_text="Warning Threshold", annotation_position="bottom right")
        fig_drift.add_hline(y=0.30, line_dash="dash", line_color="#EF4444",
                            annotation_text="Critical Threshold", annotation_position="bottom right")
        fig_drift.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color="#94A3B8"),
            hovermode="x unified",
        )
        fig_drift.update_traces(hovertemplate="<b>%{x}</b><br>Drift Share: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig_drift, use_container_width=True)

        # Drift reports table
        st.markdown("### 📋 Recent Drift Reports")
        st.dataframe(drift_reports, use_container_width=True, hide_index=True)

    else:
        st.warning("No drift monitoring reports found.", icon="⚠️")
        st.markdown("""
        To generate drift reports:

        1. First, save reference data by running the pipeline:
            ```bash
            python run.py
            ```

        2. Then run drift detection with new data:
            ```bash
            python scripts/check_drift.py
            ```

        The dashboard will update automatically once reports exist.
        """)

        # Quick drift test button
        if st.button("🧪 Run Quick Drift Test", type="primary", use_container_width=True):
            with st.spinner("Running drift detection on generated data..."):
                try:
                    import subprocess
                    import sys
                    from src.utils.config import PROJECT_ROOT

                    # Use subprocess to run the drift check script (avoids import issues)
                    result = subprocess.run(
                        [sys.executable, "scripts/check_drift.py"],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

                    if result.returncode == 0:
                        st.success("✅ Drift check completed successfully")
                        if result.stdout:
                            with st.expander("📄 Drift Report Output"):
                                st.code(result.stdout)
                    else:
                        st.error(f"Drift check failed:\n{result.stderr}")

                        # Suggest fallback option
                        st.info("💡 Make sure you've run `python run.py` first to generate training data.")

                except subprocess.TimeoutExpired:
                    st.error("Drift check timed out. The data generation may be too large.")
                except Exception as e:
                    st.error(f"Drift test failed: {e}")
                    st.info("Reference data needed. Run `python run.py` first, then try again.")

# ─── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;">
    <p style="color:#64748B;font-size:0.75rem;">
        Haett Churn Prediction Monitoring Dashboard · Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
