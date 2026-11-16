# AI-Powered Influencer Recommendation System & Buyer Intent Estimator

A portfolio prototype that ranks influencer profiles for direct-sales or mass-awareness campaigns. The Streamlit app now accepts campaign inputs and calculates a fresh shortlist from the creator dataset.

## What the project does

- Compares a product or category with creator niches using TF-IDF and cosine similarity. The dataset has niche labels but no creator bios, so matching is category-level.
- Estimates buyer persona scores with a Logistic Regression classifier using creator metadata and engagement signals.
- Ranks candidates differently by campaign objective:
  - **Direct Sales:** buyer-class score, product-niche similarity, and bot-risk score.
  - **Mass Brand Awareness:** estimated views, estimated reach, product-niche similarity, and bot-risk score.
- Estimates sponsorship cost, CPE, and CPM using transparent pricing rules.
- Shows a shortlist from the campaign form instead of displaying the full creator dataset or saved recommendation tables.

## Project files

\`\`\`text
INFLUENCER/
├── app.py
├── recommender.py
├── influencer.ipynb
├── influencer_buyer_intent_dataset.csv
├── requirements.txt
└── outputs/
    ├── direct_sales_recommendation.csv
    ├── awareness_recommendations.csv
    └── buyer_vs_watcher_metrics.png
\`\`\`

The notebook contains the exploratory analysis and can still generate the saved tables and chart in \`outputs/\`. The Streamlit app reads the creator CSV directly and calculates recommendations from the form, so you do not need to rerun the notebook for each campaign.

## Run the app

Install the dependencies, then run these commands from the project folder:

\`\`\`bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
\`\`\`

In the page, enter a product or category, choose the campaign objective, set the maximum budget per creator, and choose how many results to show. For Direct Sales, you can also set a minimum buyer score. Select **Get recommendations** to see the matching shortlist. The app displays only the results for that submission and lets you download them as a CSV.

## Data fields used

The input file has 1,000 creator records and includes platform, niche, tier, followers, engagement rate, save/share rate, comment-intent score, bot-risk index, CVI, and audience persona.

## Model and demo results

The notebook uses a stratified train/validation split and five-fold out-of-fold predictions. In this dataset, the out-of-fold confusion matrix contained 957 correct persona predictions out of 1,000 (95.7% accuracy, approximately 97.0% balanced accuracy).

At the 0.80 buyer-score cutoff, 35 creators were selected; precision against the dataset's High-Intent Buyer labels was 0.886 and recall was 0.838. These are label-matching metrics, not purchase-conversion metrics.

## Important assumptions and limitations

- Persona labels align exactly with CVI bands in the supplied dataset. The classifier may be reproducing the rules used to create the labels; its scores do not establish real-world purchase intent.
- There are no actual purchases, attributed conversions, or campaign outcomes in the data.
- The dataset does not include creator bios or raw comments. TF-IDF uses a manually written niche vocabulary; the comment-intent value is supplied as a feature rather than trained from raw comments.
- Measured reach and views are absent. The app estimates them from follower counts using platform-level assumptions.
- Sponsorship CPM values, platform adjustments, and the minimum per-post fee are demo assumptions, not market quotes.
- CPE uses estimated engagements from followers multiplied by the engagement-rate percentage.
- The dataset does not provide creator names, target-region audience shares, or age demographics.
- The app shows only a shortlist, but the dataset CSV itself is still included in the repository. Confirm that its license permits redistribution before sharing the repository publicly.

Do not describe the estimated metrics as observed campaign performance.

## Next improvements

1. Replace proxy reach/views and heuristic prices with measured post and campaign data.
2. Add creator bios and labeled comment text for richer content matching and intent classification.
3. Collect click and conversion outcomes to train and evaluate buyer-intent models against real outcomes.
4. Calibrate probability scores and select decision thresholds on a dedicated validation set.
5. Add region and demographic audience attributes, and test ranking quality across groups.
