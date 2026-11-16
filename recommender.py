"""Interactive influencer recommendation helpers.

The scores and prices are prototype estimates based on the supplied dataset.
They are not measured campaign outcomes or purchase probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


NICHE_DESCRIPTIONS = {
    "Tech": "technology electronics gadgets audio headphones earbuds wireless devices",
    "Beauty & Skincare": "beauty skincare cosmetics makeup haircare",
    "Finance": "finance investing banking money personal finance",
    "Gaming": "video games esports gaming streams consoles",
    "Travel": "travel destinations tourism hotels trip planning",
    "Fitness": "fitness workouts exercise training wellness",
}

PLATFORM_REACH_RATE = {"Instagram": 0.20, "YouTube": 0.25, "TikTok": 0.30}
PLATFORM_VIEWS_PER_REACH = {"Instagram": 1.35, "YouTube": 1.75, "TikTok": 1.60}
NICHE_CPM = {"Tech": 18, "Finance": 24, "Fitness": 14, "Beauty & Skincare": 15, "Gaming": 12, "Travel": 16}
PLATFORM_PRICE_FACTOR = {"Instagram": 1.00, "YouTube": 1.18, "TikTok": 0.88}

CATEGORICAL_FEATURES = ["Platform", "Niche", "Tier"]
NUMERIC_FEATURES = [
    "Followers",
    "Engagement_Rate",
    "Save_Share_Rate",
    "Comment_Intent_Score",
    "Bot_Risk_Index",
]


def score_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Add out-of-fold buyer scores and predicted audience personas."""
    required = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["Audience_Persona"])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    y = data["Audience_Persona"].astype(str)
    classes = np.array(sorted(y.unique()))
    if len(classes) < 2:
        raise ValueError("The dataset needs at least two audience persona classes.")

    features = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    class_counts = y.value_counts()
    smallest_class = int(class_counts.min())
    if smallest_class >= 2:
        folds = min(5, smallest_class)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        probabilities = cross_val_predict(
            model,
            features,
            y,
            cv=cv,
            method="predict_proba",
        )
    else:
        # Tiny custom datasets may have a persona represented by one row.
        # This fallback keeps the app usable, but its scores are in-sample.
        model.fit(features, y)
        probabilities = model.predict_proba(features)
        classes = model.named_steps["classifier"].classes_

    scored = data.copy()
    if "High-Intent Buyers" in classes:
        buyer_index = int(np.where(classes == "High-Intent Buyers")[0][0])
        scored["Buyer class score (%)"] = probabilities[:, buyer_index] * 100
    else:
        scored["Buyer class score (%)"] = 0.0
    scored["Predicted Audience Persona"] = classes[np.argmax(probabilities, axis=1)]
    return scored


def recommend_creators(
    scored_catalog: pd.DataFrame,
    product_query: str,
    objective: str,
    max_cost_per_creator: float,
    top_k: int = 10,
    minimum_buyer_score: float = 80.0,
) -> pd.DataFrame:
    """Rank creators for a submitted campaign form."""
    if not product_query.strip():
        raise ValueError("Enter a product or category to get recommendations.")

    candidates = scored_catalog.copy()
    niche_text = candidates["Niche"].astype(str).map(NICHE_DESCRIPTIONS)
    niche_text = niche_text.fillna(candidates["Niche"].astype(str)).tolist()

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        text_matrix = vectorizer.fit_transform(niche_text + [product_query.strip()])
        candidates["Product-Niche Match (%)"] = (
            cosine_similarity(text_matrix[-1], text_matrix[:-1]).ravel() * 100
        )
    except ValueError:
        # For queries made only of stop words or symbols, keep ranking usable.
        candidates["Product-Niche Match (%)"] = 0.0

    reach_rate = candidates["Platform"].map(PLATFORM_REACH_RATE).fillna(0.20)
    views_per_reach = candidates["Platform"].map(PLATFORM_VIEWS_PER_REACH).fillna(1.35)
    candidates["Estimated Reach per Post"] = candidates["Followers"] * reach_rate
    candidates["Estimated Views per Post"] = (
        candidates["Estimated Reach per Post"] * views_per_reach
    )

    reference_engagement = max(float(candidates["Engagement_Rate"].median()), 0.01)
    engagement_factor = (
        0.8 + 0.2 * candidates["Engagement_Rate"] / reference_engagement
    ).clip(0.75, 1.35)
    niche_cpm = candidates["Niche"].map(NICHE_CPM).fillna(12)
    platform_factor = candidates["Platform"].map(PLATFORM_PRICE_FACTOR).fillna(1.0)
    candidates["Estimated Sponsorship Cost ($)"] = (
        candidates["Estimated Reach per Post"] / 1000
        * niche_cpm
        * platform_factor
        * engagement_factor
        + candidates["Followers"] / 1000 * 0.35
    ).clip(lower=75).round(2)

    candidates["Estimated Engagements"] = (
        candidates["Followers"] * candidates["Engagement_Rate"] / 100
    ).clip(lower=1)
    candidates["CPE ($)"] = (
        candidates["Estimated Sponsorship Cost ($)"]
        / candidates["Estimated Engagements"]
    ).round(2)
    safe_reach = candidates["Estimated Reach per Post"].replace(0, np.nan)
    candidates["CPM ($)"] = (
        candidates["Estimated Sponsorship Cost ($)"] / safe_reach * 1000
    ).fillna(0).round(2)

    candidates["Affordable"] = (
        candidates["Estimated Sponsorship Cost ($)"] <= max_cost_per_creator
    )
    candidates["Reach Rank"] = candidates["Estimated Reach per Post"].rank(pct=True)
    candidates["Views Rank"] = candidates["Estimated Views per Post"].rank(pct=True)
    bot_safety = (1 - candidates["Bot_Risk_Index"]).clip(0, 1)
    match_fraction = candidates["Product-Niche Match (%)"] / 100

    if objective == "Direct Sales":
        candidates["Match Score (%)"] = 100 * (
            0.55 * candidates["Buyer class score (%)"] / 100
            + 0.25 * match_fraction
            + 0.20 * bot_safety
        )
        results = candidates[
            candidates["Affordable"]
            & (candidates["Buyer class score (%)"] >= minimum_buyer_score)
        ].sort_values("Match Score (%)", ascending=False)
        if not results.empty:
            results = results.copy()
            results["Marketing Suitability"] = np.where(
                (results["Predicted Audience Persona"] == "High-Intent Buyers")
                & (results["Bot_Risk_Index"] <= 0.20),
                "Recommended for Direct Sales",
                "Review Before Direct Sales",
            )
    elif objective == "Mass Brand Awareness":
        candidates["Match Score (%)"] = 100 * (
            0.40 * candidates["Views Rank"]
            + 0.30 * candidates["Reach Rank"]
            + 0.15 * match_fraction
            + 0.15 * bot_safety
        )
        results = candidates[candidates["Affordable"]].sort_values(
            "Match Score (%)",
            ascending=False,
        ).copy()
        if not results.empty:
            results["Marketing Suitability"] = "Recommended for Awareness"
    else:
        raise ValueError("Choose Direct Sales or Mass Brand Awareness.")

    results = results.head(int(top_k)).copy()
    if results.empty:
        return results

    results.insert(0, "Rank", range(1, len(results) + 1))
    columns = [
        "Rank",
        "Influencer_ID",
        "Platform",
        "Niche",
        "Followers",
        "Product-Niche Match (%)",
        "Predicted Audience Persona",
        "Buyer class score (%)",
        "Estimated Reach per Post",
        "Estimated Views per Post",
        "Bot_Risk_Index",
        "Match Score (%)",
        "Estimated Sponsorship Cost ($)",
        "CPE ($)",
        "CPM ($)",
        "Marketing Suitability",
    ]
    visible_columns = [column for column in columns if column in results.columns]
    return results[visible_columns].round(2)
