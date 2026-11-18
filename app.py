from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "influencer_buyer_intent_dataset.csv"

NICHE_TEXT = {
    "Tech": "technology electronics gadgets audio headphones earbuds wireless devices computers phones",
    "Beauty & Skincare": "beauty skincare cosmetics makeup haircare skin products",
    "Finance": "finance investing banking money personal finance wealth",
    "Gaming": "video games esports gaming streams consoles pc games",
    "Travel": "travel destinations tourism hotels trips vacations",
    "Fitness": "fitness workouts exercise training wellness health gym",
}

PLATFORM_REACH_RATE = {"Instagram": 0.20, "YouTube": 0.25, "TikTok": 0.30}
PLATFORM_VIEW_FACTOR = {"Instagram": 1.35, "YouTube": 1.75, "TikTok": 1.60}
PLATFORM_PRICE_FACTOR = {"Instagram": 1.00, "YouTube": 1.18, "TikTok": 0.88}
NICHE_CPM = {
    "Tech": 18,
    "Finance": 24,
    "Fitness": 14,
    "Beauty & Skincare": 15,
    "Gaming": 12,
    "Travel": 16,
}

CATEGORICAL_FEATURES = ["Platform", "Niche", "Tier"]
NUMERIC_FEATURES = [
    "Followers",
    "Engagement_Rate",
    "Save_Share_Rate",
    "Comment_Intent_Score",
    "Bot_Risk_Index",
]


@st.cache_data
def load_creator_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Preparing audience-intent estimates...")
def score_creators(data: pd.DataFrame) -> pd.DataFrame:
    """Attach out-of-fold persona probabilities to every creator row."""
    features = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    labels = data["Audience_Persona"].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        classifier, features, labels, cv=cv, method="predict_proba"
    )
    classes = np.sort(labels.unique())
    buyer_index = list(classes).index("High-Intent Buyers")

    scored = data.copy()
    scored["Buyer class score (%)"] = probabilities[:, buyer_index] * 100
    scored["Predicted Audience Persona"] = classes[probabilities.argmax(axis=1)]
    return scored


def semantic_match(product_text: str, category: str, niches: pd.Series) -> np.ndarray:
    category_words = NICHE_TEXT.get(category, category.lower())
    query = f"{product_text} {category} {category_words}"
    niche_documents = niches.astype(str).map(
        lambda niche: NICHE_TEXT.get(niche, niche.lower())
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([*niche_documents.tolist(), query])
    return cosine_similarity(matrix[-1], matrix[:-1]).ravel()


def minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(1.0, index=series.index)
    return (series - low) / (high - low)


def add_estimates(creators: pd.DataFrame) -> pd.DataFrame:
    result = creators.copy()
    result["Estimated Reach per Post"] = (
        result["Followers"] * result["Platform"].map(PLATFORM_REACH_RATE).fillna(0.20)
    )
    result["Estimated Views per Post"] = (
        result["Estimated Reach per Post"]
        * result["Platform"].map(PLATFORM_VIEW_FACTOR).fillna(1.0)
    )

    reference_engagement = max(float(result["Engagement_Rate"].median()), 0.01)
    engagement_factor = (
        0.8 + 0.2 * result["Engagement_Rate"] / reference_engagement
    ).clip(0.75, 1.35)
    result["Estimated Sponsorship Cost ($)"] = (
        result["Estimated Reach per Post"]
        / 1000
        * result["Niche"].map(NICHE_CPM).fillna(12)
        * result["Platform"].map(PLATFORM_PRICE_FACTOR).fillna(1.0)
        * engagement_factor
        + result["Followers"] / 1000 * 0.35
    ).clip(lower=75).round(2)

    result["Estimated Engagements"] = (
        result["Followers"] * result["Engagement_Rate"] / 100
    ).clip(lower=1)
    result["CPE ($)"] = (
        result["Estimated Sponsorship Cost ($)"] / result["Estimated Engagements"]
    ).round(2)
    result["CPM ($)"] = (
        result["Estimated Sponsorship Cost ($)"]
        / result["Estimated Reach per Post"]
        * 1000
    ).round(2)
    return result


st.set_page_config(page_title="Influencer Recommender", page_icon="🎯", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#20251f; --muted:#73786f; --paper:#f4f5ef; --panel:#fffefa; --line:#dfe2d8; --lime:#a7d64b; --deep:#20251f; }
    html, body, [class*="css"] { font-family:'Manrope', 'Aptos', sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(244,245,239,.86); }
    [data-testid="stToolbar"] { right:1rem; }
    section.main > div.block-container { max-width:1420px; padding-top:1.3rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.045em; }
    h1 { font-size:clamp(2.7rem,5.2vw,5rem)!important; line-height:.98!important; font-weight:700!important; }
    h2 { font-size:1.8rem!important; font-weight:700!important; }
    p, label, span { color:inherit; }
    .eyebrow { font:500 .68rem 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; color:#65705f; }
    .hero-wrap { display:grid; grid-template-columns:1.08fr .92fr; gap:2rem; align-items:center; margin:.5rem 0 2.3rem; }
    .hero-copy { padding:1.4rem .3rem 1.2rem 0; }
    .hero-copy h1 { margin:.9rem 0 1.1rem; }
    .hero-copy h1 em { color:#8bbd36; font-style:normal; }
    .hero-copy p { max-width:600px; color:#686f65; font-size:1rem; line-height:1.75; }
    .hero-art { min-height:260px; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden; background:#e9ebe2; border:1px solid #e0e3d9; }
    .orbit { width:230px; height:230px; border:1px solid #c5cbbd; border-radius:50%; display:flex; align-items:center; justify-content:center; position:relative; }
    .orbit:before,.orbit:after { content:''; position:absolute; border:1px dashed #cbd1c2; border-radius:50%; inset:22px; }
    .orbit:after { inset:48px; border-style:solid; border-color:#bed28f; }
    .orbit-core { z-index:1; width:108px; height:108px; border-radius:50%; background:var(--deep); color:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; font-size:.82rem; line-height:1.35; }
    .orbit-core small { color:var(--lime); font:500 .54rem 'DM Mono',monospace; letter-spacing:.12em; margin-bottom:.4rem; }
    .orbit-tag { position:absolute; background:#fbfcf7; border:1px solid #dfe2d8; padding:.42rem .62rem; font:500 .62rem 'DM Mono',monospace; color:#313a2b; z-index:2; }
    .orbit-tag.left { left:5%; top:32%; }.orbit-tag.right { right:4%; bottom:28%; }
    div[data-testid="stForm"] { background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:1.3rem 1.5rem 1.5rem; }
    div[data-testid="stForm"] h3 { font-size:1.05rem!important; letter-spacing:-.02em; }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, div[data-testid="stNumberInput"] input { background:#f7f8f3; border-color:#dfe2d8; border-radius:2px; }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#f7f8f3; border-color:#dfe2d8; border-radius:2px; }
    div[data-testid="stButton"] button[kind="primaryFormSubmit"], div[data-testid="stFormSubmitButton"] button { background:var(--deep); color:white; border:0; border-radius:2px; padding:.65rem 1.1rem; font-weight:700; }
    div[data-testid="stFormSubmitButton"] button:hover { background:#343c31; border:0; color:white; }
    div[data-testid="stMetric"] { background:var(--deep); color:#f8f9f3; border-radius:2px; padding:1rem 1.1rem; border:1px solid #343b32; min-height:112px; }
    div[data-testid="stMetricLabel"] p { color:#c0c8b6!important; font:500 .62rem 'DM Mono',monospace; letter-spacing:.09em; text-transform:uppercase; }
    div[data-testid="stMetricValue"] { color:#fff!important; font-weight:700; }
    .section-kicker { font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; color:#73786f; margin:.2rem 0 .65rem; }
    .creator-card { height:100%; background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--lime); padding:1.1rem 1.15rem; margin:.35rem 0 1rem; }
    .creator-card .rank { color:#879078; font:500 .62rem 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
    .creator-card h3 { font-size:1.1rem; margin:.55rem 0 .15rem; letter-spacing:-.035em; }
    .creator-card .meta { color:var(--muted); font-size:.78rem; margin-bottom:.85rem; }
    .creator-card .score { font-size:1.8rem; font-weight:800; letter-spacing:-.07em; }
    .creator-card .score-label { display:block; font:500 .6rem 'DM Mono',monospace; color:var(--muted); letter-spacing:.09em; text-transform:uppercase; }
    .creator-card .mini { border-top:1px solid var(--line); padding-top:.65rem; margin-top:.7rem; color:#454c43; font-size:.74rem; line-height:1.7; }
    .pill { display:inline-block; border:1px solid #dce3cf; color:#4e6c22; background:#f1f6e8; padding:.22rem .45rem; margin-top:.4rem; font:500 .58rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.04em; }
    [data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:2px; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); }
    [data-testid="stCaptionContainer"] { color:#788075; }
    @media(max-width:800px) { .hero-wrap { grid-template-columns:1fr; gap:1rem; }.hero-art { min-height:220px; }.orbit { transform:scale(.85); } }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-copy">
        <div class="eyebrow">● &nbsp; CREATOR INTELLIGENCE &nbsp; · &nbsp; CAMPAIGN PLANNING</div>
        <h1>Find the right<br>creator. <em>Move</em> the<br>right audience.</h1>
        <p>Build a campaign brief, compare creator fit, and see a budget-aware shortlist shaped around your objective.</p>
      </div>
      <div class="hero-art">
        <span class="orbit-tag left">signal · audience</span>
        <div class="orbit"><div class="orbit-core"><small>FROM DATA</small>to<br>direction</div></div>
        <span class="orbit-tag right">intent · reach</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-kicker">CAMPAIGN BUILDER &nbsp; / &nbsp; SET YOUR BRIEF</div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Creator dataset not found: {DATA_PATH}")
    st.stop()

try:
    creators = score_creators(load_creator_data(str(DATA_PATH)))
except Exception as exc:
    st.error(f"Could not prepare recommendations: {exc}")
    st.stop()

with st.form("campaign_form"):
    st.subheader("Campaign details")
    row1, row2 = st.columns(2)
    with row1:
        product_name = st.text_input("Product name", "Wireless Earbuds")
        product_description = st.text_area(
            "Product description or keywords",
            "Wireless audio, noise cancellation, music, consumer electronics",
            height=90,
        )
        categories = sorted(creators["Niche"].dropna().unique().tolist())
        default_category_index = categories.index("Tech") if "Tech" in categories else 0
        category = st.selectbox("Product category", categories, index=default_category_index)
    with row2:
        target_audience = st.text_input(
            "Target audience / region (optional)",
            placeholder="e.g., students in India, ages 18–30",
        )
        objective = st.radio(
            "Campaign objective",
            ["High Conversion / Direct Sales", "Mass Brand Awareness"],
        )
        budget = st.number_input(
            "Maximum sponsorship budget (USD)", min_value=0, value=3000, step=100
        )
        top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
    row3, row4 = st.columns(2)
    with row3:
        available_platforms = sorted(creators["Platform"].dropna().unique().tolist())
        selected_platforms = st.multiselect(
            "Platforms to include", available_platforms, default=available_platforms
        )
    with row4:
        max_bot_risk = st.slider(
            "Maximum bot-risk score", min_value=0.0, max_value=1.0, value=0.50, step=0.05
        )
    min_buyer_score = st.slider(
        "Minimum buyer-intent score for direct sales (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Only applied to Direct Sales campaigns.",
    )
    submitted = st.form_submit_button("Find recommended creators", type="primary")

if submitted:
    if not product_name.strip():
        st.warning("Enter a product name to generate recommendations.")
        st.stop()

    product_text = f"{product_name}. {product_description}. {category}"
    results = creators.copy()
    results["Semantic_Match_Score"] = semantic_match(
        product_text, category, results["Niche"]
    )
    results = add_estimates(results)
    results = results[
        results["Platform"].isin(selected_platforms)
        & (results["Bot_Risk_Index"] <= max_bot_risk)
    ].copy()

    if objective == "High Conversion / Direct Sales":
        comment_signal = ((results["Comment_Intent_Score"] + 1) / 2).clip(0, 1)
        cvi_signal = (results["CVI"] / 3.6).clip(0, 1)
        results["Match Score (%)"] = 100 * (
            0.40 * results["Buyer class score (%)"] / 100
            + 0.20 * comment_signal
            + 0.15 * cvi_signal
            + 0.15 * results["Semantic_Match_Score"]
            + 0.10 * (1 - results["Bot_Risk_Index"])
        )
        results["Marketing Suitability"] = np.select(
            [
                (results["Buyer class score (%)"] >= 80)
                & (results["Bot_Risk_Index"] <= 0.25),
                results["Semantic_Match_Score"] > 0,
            ],
            ["Highly Recommended for Direct Sales", "Review for Direct Sales"],
            default="Not Recommended",
        )
        results = results[results["Buyer class score (%)"] >= min_buyer_score].copy()
    else:
        results["Match Score (%)"] = 100 * (
            0.40 * minmax(results["Estimated Reach per Post"])
            + 0.40 * minmax(results["Estimated Views per Post"])
            + 0.10 * minmax(results["Semantic_Match_Score"])
            + 0.10 * (1 - results["Bot_Risk_Index"])
        )
        results["Marketing Suitability"] = "Best for Awareness Only"

    affordable = results[results["Estimated Sponsorship Cost ($)"] <= budget].copy()
    ranked = affordable.sort_values("Match Score (%)", ascending=False).head(top_k).copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))

    st.markdown('<div class="section-kicker">YOUR SHORTLIST &nbsp; / &nbsp; LIVE RESULTS</div>', unsafe_allow_html=True)
    st.subheader("Creators matched to your brief")
    st.caption(f"{len(ranked)} creators fit the selected filters and ${budget:,.0f} budget · {objective}")
    if ranked.empty:
        st.info("No creators match all selected filters and this budget. Try widening the platform, buyer-score, or bot-risk filters, or increase the budget.")
    else:
        spend = ranked["Estimated Sponsorship Cost ($)"].sum()
        average_buyer = ranked["Buyer class score (%)"].mean()
        average_match = ranked["Match Score (%)"].mean()
        summary = st.columns(4)
        summary[0].metric("CREATORS SHORTLISTED", f"{len(ranked)}")
        summary[1].metric("ESTIMATED SHORTLIST COST", f"${spend:,.0f}")
        summary[2].metric("AVG BUYER INTENT", f"{average_buyer:.1f}%")
        summary[3].metric("AVG MATCH SCORE", f"{average_match:.1f}%")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">TOP CREATOR MATCHES</div>', unsafe_allow_html=True)
        featured = ranked.head(min(3, len(ranked)))
        cards = st.columns(len(featured))
        for card, (_, creator) in zip(cards, featured.iterrows()):
            creator_id = escape(str(creator["Influencer_ID"]))
            creator_niche = escape(str(creator["Niche"]))
            creator_platform = escape(str(creator["Platform"]))
            persona = escape(str(creator["Predicted Audience Persona"]))
            card.markdown(
                f"""
                <div class="creator-card">
                  <div class="rank">MATCH #{int(creator['Rank']):02d} &nbsp; · &nbsp; {creator_platform}</div>
                  <h3>{creator_id}</h3>
                  <div class="meta">{creator_niche} &nbsp; / &nbsp; {escape(str(creator['Tier']))}</div>
                  <span class="score">{creator['Match Score (%)']:.1f}%</span>
                  <span class="score-label">campaign match</span>
                  <span class="pill">{persona}</span>
                  <div class="mini">
                    {creator['Followers']:,.0f} followers<br>
                    {creator['Estimated Reach per Post']:,.0f} est. reach<br>
                    ${creator['Estimated Sponsorship Cost ($)']:,.0f} est. post cost
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-kicker">MATCH SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
        st.bar_chart(ranked.set_index("Influencer_ID")[["Match Score (%)"]], height=260)

        display_columns = [
            "Rank",
            "Influencer_ID",
            "Platform",
            "Niche",
            "Tier",
            "Followers",
            "Estimated Reach per Post",
            "Engagement_Rate",
            "Bot_Risk_Index",
            "Buyer class score (%)",
            "Predicted Audience Persona",
            "Semantic_Match_Score",
            "Match Score (%)",
            "Estimated Sponsorship Cost ($)",
            "CPE ($)",
            "CPM ($)",
            "Marketing Suitability",
        ]
        with st.expander("View all creator metrics"):
            st.dataframe(
                ranked[display_columns].round(2),
                use_container_width=True,
                hide_index=True,
            )

    if target_audience.strip():
        st.info(
            "Target audience noted: "
            f"{target_audience.strip()}. The current dataset has no age, region, or demographic "
            "fields, so this information is not yet used to rank creators."
        )

    st.caption(
        "Audience probabilities are out-of-fold estimates from the project’s synthetic labels. "
        "Estimated reach, views, and sponsorship costs are rule-based proxies, not observed results."
    )

with st.expander("Compare buyer and watcher audiences"):
    st.markdown('<div class="section-kicker">AUDIENCE SIGNALS &nbsp; / &nbsp; BUYER VS. WATCHER</div>', unsafe_allow_html=True)
    persona_choice = st.radio(
        "Audience group",
        ["High-Intent Buyers", "Passive Watchers"],
        horizontal=True,
        key="persona_choice",
    )
    segment = creators[creators["Audience_Persona"] == persona_choice]
    if segment.empty:
        st.info(f"No creators are labeled {persona_choice} in the dataset.")
    else:
        st.write(f"Creator profiles in this group: **{len(segment):,}**")
        averages = segment[
            ["Save_Share_Rate", "Comment_Intent_Score", "Bot_Risk_Index", "CVI"]
        ].mean()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Save/share rate", f"{averages['Save_Share_Rate']:.2f}%")
        metric_cols[1].metric("Comment intent", f"{averages['Comment_Intent_Score']:.2f}")
        metric_cols[2].metric("Bot risk", f"{averages['Bot_Risk_Index']:.2f}")
        metric_cols[3].metric("CVI", f"{averages['CVI']:.2f}")
        st.caption("Metrics above show the selected group’s averages. CVI is on the dataset’s original scale.")

        group_means = creators.groupby("Audience_Persona")[[
            "Save_Share_Rate", "Comment_Intent_Score", "Bot_Risk_Index", "CVI"
        ]].mean()
        if {"High-Intent Buyers", "Passive Watchers"}.issubset(group_means.index):
            buyer = group_means.loc["High-Intent Buyers"]
            watcher = group_means.loc["Passive Watchers"]
            comparison = pd.DataFrame(
                {
                    "High-Intent Buyers": [
                        np.clip(buyer["Save_Share_Rate"] / 5 * 100, 0, 100),
                        np.clip((buyer["Comment_Intent_Score"] + 1) / 2 * 100, 0, 100),
                        np.clip(buyer["CVI"] / 3.6 * 100, 0, 100),
                        (1 - buyer["Bot_Risk_Index"]) * 100,
                    ],
                    "Passive Watchers": [
                        np.clip(watcher["Save_Share_Rate"] / 5 * 100, 0, 100),
                        np.clip((watcher["Comment_Intent_Score"] + 1) / 2 * 100, 0, 100),
                        np.clip(watcher["CVI"] / 3.6 * 100, 0, 100),
                        (1 - watcher["Bot_Risk_Index"]) * 100,
                    ],
                },
                index=["Save/share signal", "Comment intent", "CVI", "Bot safety"],
            )
            st.markdown('<div class="section-kicker">NORMALIZED SIGNAL COMPARISON &nbsp; / &nbsp; INDEXED 0–100</div>', unsafe_allow_html=True)
            st.bar_chart(comparison.round(1), height=300)
            st.caption("Signals are normalized for visual comparison; use the exact averages above for their original units.")

