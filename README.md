# AI-Powered Influencer Recommendation System & Buyer Intent Estimator

A portfolio prototype that ranks influencer profiles for direct-sales or mass-awareness campaigns and presents the results in a Streamlit dashboard.

## What the project does

- Compares a product category with creator niches using TF-IDF and cosine similarity. The current dataset has niche labels but no creator bios, so matching is category-level.
- Estimates buyer persona scores with a Logistic Regression classifier using creator metadata and engagement signals.
- Ranks candidates differently by campaign objective:
  - **Direct Sales:** buyer-class score, product-niche similarity, and bot-risk score.
  - **Mass Brand Awareness:** estimated reach, estimated views, and bot-risk score.
- Estimates sponsorship cost, CPE, and CPM using transparent pricing rules.
- Displays saved recommendation tables and a Buyer vs. Watcher audience view in Streamlit.

## Project files

```text
INFLUENCER/
├── app.py
├── influencer.ipynb
├── influencer_buyer_intent_dataset.csv   # local input data; share only if its license permits
├── requirements.txt
└── outputs/
    ├── direct_sales_recommendation.csv
    ├── awareness_recommendations.csv
    └── buyer_vs_watcher_metrics.png
```

The Streamlit app reads the creator dataset from the project root. Visitors can enter product details, choose a niche, campaign objective, and budget, and receive recommendations ranked from the full creator dataset. Reach, views, and costs are recalculated in the app from project assumptions. The `outputs/` CSVs remain saved examples of the earlier notebook recommendations. The structured `influencer.ipynb` notebook contains the analysis workflow.

## Setup and run

Open a terminal in the project folder and activate the Python environment that has this project's dependencies installed. Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app includes an interactive campaign form for product name/description, category, target audience/region, objective, budget, platform, maximum bot risk, minimum buyer-intent score, and result count. It displays ranked creator cards, summary metrics, a match-score chart, and an expandable creator table. A buyer-versus-watcher section includes group metrics and a normalized comparison chart. Target audience/region is collected for context, but the current dataset has no demographic or region fields, so it does not affect ranking yet.

## Data fields used

The input file used in this prototype has 1,000 creator records and includes platform, niche, tier, followers, engagement rate, save/share rate, comment-intent score, bot-risk index, CVI, and audience persona.

## Model and demo results

The notebook uses a stratified train/validation split and five-fold out-of-fold predictions. In this dataset, the out-of-fold confusion matrix contained 957 correct persona predictions out of 1,000 (95.7% accuracy, approximately 97.0% balanced accuracy).

At the 0.80 buyer-score cutoff, 35 creators were selected; precision against the dataset's High-Intent Buyer labels was 0.886 and recall was 0.838. These are label-matching metrics, not purchase-conversion metrics.

## Important assumptions and limitations

- Persona labels align exactly with CVI bands in the supplied dataset. The classifier may be reproducing the rules used to create the labels; its scores do not establish real-world purchase intent.
- There are no actual purchases, attributed conversions, or campaign outcomes in the data.
- The dataset does not include creator bios or raw comments. TF-IDF uses a manually written niche vocabulary; the comment-intent value is supplied as a feature rather than trained from raw comments here.
- Measured reach and views are absent. The project estimates them from follower counts using platform-level assumptions.
- Sponsorship CPM values, platform adjustments, and the minimum per-post fee are demo assumptions, not market quotes.
- CPE uses estimated engagements from followers multiplied by the engagement-rate percentage.
- The dataset does not provide creator names, target-region audience shares, or age demographics.

Before publishing the repository, confirm that the dataset license allows redistribution. If it does not, omit the CSV and explain how users can obtain an authorized copy. Do not describe the estimated metrics as observed campaign performance.

## Next improvements

1. Replace proxy reach/views and heuristic prices with measured post and campaign data.
2. Add creator bios and labeled comment text for richer content matching and intent classification.
3. Collect click and conversion outcomes to train and evaluate buyer-intent models against real outcomes.
4. Calibrate probability scores and select decision thresholds on a dedicated validation set.
5. Add region and demographic audience attributes, and test ranking quality across groups.


