from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import recommend_creators, score_catalog


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "influencer_buyer_intent_dataset.csv"

st.set_page_config(
    page_title="Influencer Campaign Planner",
    page_icon="🎯",
    layout="wide",
)

st.title("Influencer Campaign Planner")
st.caption(
    "Enter campaign details to get a ranked shortlist. "
    "The full creator dataset is not shown on this page."
)

if not DATA_PATH.exists():
    st.error("The creator dataset file is missing from the project folder.")
    st.stop()


@st.cache_data(show_spinner="Preparing the recommendation model...")
def load_scored_catalog(data_path: str, modified_at: float) -> pd.DataFrame:
    # modified_at invalidates the cached result if the CSV changes.
    del modified_at
    return score_catalog(pd.read_csv(data_path))


try:
    catalog = load_scored_catalog(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Could not prepare recommendations: {error}")
    st.stop()

with st.form("campaign_inputs"):
    st.subheader("Campaign details")
    product_query = st.text_input(
        "What product or category are you promoting?",
        placeholder="For example: wireless earbuds, skincare, or fitness app",
    )

    first_row, second_row = st.columns(2)
    with first_row:
        objective = st.selectbox(
            "Campaign objective",
            ["Direct Sales", "Mass Brand Awareness"],
        )
        max_cost_per_creator = st.number_input(
            "Maximum budget per creator (USD)",
            min_value=75.0,
            max_value=1_000_000.0,
            value=3000.0,
            step=100.0,
        )
    with second_row:
        top_k = st.slider("How many recommendations?", 1, 25, 10)
        minimum_buyer_score = st.slider(
            "Minimum buyer score for Direct Sales (%)",
            min_value=50,
            max_value=95,
            value=80,
        )

    submitted = st.form_submit_button(
        "Get recommendations",
        type="primary",
        use_container_width=True,
    )

if submitted:
    try:
        results = recommend_creators(
            scored_catalog=catalog,
            product_query=product_query,
            objective=objective,
            max_cost_per_creator=max_cost_per_creator,
            top_k=top_k,
            minimum_buyer_score=float(minimum_buyer_score),
        )
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Could not generate recommendations: {error}")
    else:
        st.divider()
        st.subheader("Your recommendations")

        if results.empty:
            st.warning(
                "No creators matched these settings. Try a higher budget, "
                "a lower buyer-score cutoff, or a broader product description."
            )
        else:
            st.caption(
                "Buyer scores, reach, views, and sponsorship costs are estimates "
                "from the prototype dataset, not measured campaign outcomes."
            )
            st.metric("Creators recommended", len(results))
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download these recommendations",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="influencer_recommendations.csv",
                mime="text/csv",
            )
else:
    st.info(
        "Fill in the campaign details and select **Get recommendations**. "
        "The app will calculate a fresh shortlist from the saved creator data."
    )
