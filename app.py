from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = BASE_DIR / "influencer_buyer_intent_dataset.csv"

st.set_page_config(
    page_title="Influencer Recommendation Dashboard",
    layout="wide",
)

st.title("Influencer Recommendation Dashboard")
st.caption("Explore creator recommendations by campaign objective and audience type.")

objective = st.radio(
    "Campaign objective",
    ["Direct Sales", "Mass Brand Awareness"],
    horizontal=True,
)

if objective == "Direct Sales":
    recommendation_path = OUTPUT_DIR / "direct_sales_recommendation.csv"
else:
    recommendation_path = OUTPUT_DIR / "awareness_recommendations.csv"

st.subheader(f"{objective} recommendations")

if recommendation_path.exists():
    recommendations = pd.read_csv(recommendation_path)
    st.metric("Creators in this recommendation file", len(recommendations))
    st.dataframe(recommendations, use_container_width=True)
else:
    st.error(f"Recommendation file not found: {recommendation_path}")

st.divider()
st.subheader("Buyer vs. Watcher audience view")

audience_view = st.radio(
    "Choose an audience",
    ["High-Intent Buyers", "Passive Watchers"],
    horizontal=True,
)

if DATA_PATH.exists():
    creators = pd.read_csv(DATA_PATH)

    audience_data = creators[
        creators["Audience_Persona"] == audience_view
    ]

    if audience_data.empty:
        st.warning(f"No rows found for {audience_view}.")
    else:
        st.write(f"Creator profiles labeled **{audience_view}**: {len(audience_data)}")

        averages = audience_data[
            [
                "Save_Share_Rate",
                "Comment_Intent_Score",
                "Bot_Risk_Index",
                "CVI",
            ]
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Save/share rate", f"{averages['Save_Share_Rate']:.2f}%")
        col2.metric("Comment intent", f"{averages['Comment_Intent_Score']:.2f}")
        col3.metric("Bot risk", f"{averages['Bot_Risk_Index']:.2f}")
        col4.metric("CVI", f"{averages['CVI']:.2f}")

        st.caption("CVI is shown in the dataset’s original index scale.")
else:
    st.error(f"Dataset not found: {DATA_PATH}")