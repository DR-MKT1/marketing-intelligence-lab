"""
Marketing Intelligence Lab — by Dr. Nik
A point-and-click teaching app covering the full marketing-analytics toolkit,
with novice-level Help on every element.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Dr. Nik's Marketing Intelligence Lab", page_icon="📊", layout="wide")

BRAND_NAME = "Dr. Nik"
BRAND_NOTICE = f"© 2026 {BRAND_NAME} · Marketing Intelligence Lab · All rights reserved."
DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------- helpers
@st.cache_data
def load_builtin(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

def num_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def cat_cols(df, max_unique=25):
    out = []
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) and df[c].nunique() <= max_unique:
            out.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() <= 10:
            out.append(c)
    return out

def text_cols(df):
    out = []
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            avg = df[c].dropna().astype(str).str.len().mean()
            if avg and avg > 30:
                out.append(c)
    return out

def date_cols(df):
    return [c for c in df.columns if "date" in c.lower()]

def classify_role(df, c):
    """Plain-English role of a column, for the Column info table."""
    if "date" in c.lower():
        return "📅 Date"
    if pd.api.types.is_numeric_dtype(df[c]):
        u = df[c].dropna().unique()
        if set(u) <= {0, 1} and len(u) == 2:
            return "🔀 Binary 0/1 (numeric, but acts categorical)"
        if df[c].dropna().nunique() <= 10:
            return "🔢 Numeric with few values (could be ordinal/categorical)"
        return "🔢 Numeric (continuous)"
    if pd.api.types.is_string_dtype(df[c]):
        avg = df[c].dropna().astype(str).str.len().mean() or 0
        if avg > 30:
            return "💬 Text (long free text)"
        if df[c].nunique() <= 25:
            return "🏷️ Categorical"
        return "🏷️ Categorical (many levels — e.g. an ID or name)"
    return "❓ Other"

def default_ix(options, preferred):
    for p in preferred:
        if p in options:
            return options.index(p)
    return 0

def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)

def metric_row(pairs):
    cols = st.columns(len(pairs))
    for col, (label, value) in zip(cols, pairs):
        col.metric(label, value)

def confusion_report(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    return cm, acc, sens, spec

def section(title, help_key=None):
    """A subheader with an ℹ️ Help popover next to it."""
    a, b = st.columns([0.8, 0.2])
    a.subheader(title)
    if help_key and help_key in HELP:
        with b.popover("ℹ️ Help"):
            st.markdown(HELP[help_key])

def instructions(md):
    with st.expander("📖  Step-by-step instructions — read me first"):
        st.markdown(md)

# ---------------------------------------------------------------- HELP LIBRARY
HELP = {

"missing": """**What are missing values?** Blank cells — the data was never recorded (a customer skipped a survey question, a system glitch, etc.).

**Why it matters:** most statistical methods silently drop incomplete rows, which can shrink your sample or bias it if the missingness isn't random.

**Your two options:**
- **Omission** — delete rows with blanks. Honest, but you lose data.
- **Imputation** — fill blanks with a best guess (mean or median). Keeps rows, but invents values.

**How to choose:** impute when missingness is small (a few %) and the column matters for your analysis; omit when a row is too incomplete to trust. Use the **median** for skewed variables like income or spend — one millionaire drags the *mean* up, but not the median.

**Caution:** never impute the variable you are trying to predict.""",

"filter": """**What is subsetting/filtering?** Keeping only the rows that match a condition — e.g. only customers in one region, or only orders above $100.

**Why marketers use it constantly:** most business questions are about *some* customers, not all ("how do our app users behave?").

**How to use it here:** pick a column, set the condition, check the row count preview, then Apply. The filter permanently changes the working data for **all tabs** — use the sidebar's *Reset data* to undo.

**Caution:** filtering before modeling changes what your conclusions apply to. If you keep only Texas, your regression describes Texas.""",

"binning": """**What is binning?** Converting a numeric variable into categories — e.g. age 18–30, 31–45, 46–60, 60+.

**Why do it:** categories are easier to communicate ("the 31–45 group spends the most" beats "spend rises 6.6 per year of age"), and some analyses (crosstabs, chi-square) need categories.

**How this works:** equal-width bins split the range into slices of the same size. The new column is added to your data with "_group" in the name.

**Caution:** binning throws away precision. Do it for reporting and crosstabs, not before a regression — regression prefers the raw number.""",

"dates": """**Why dates are secretly rich variables:** one date column contains a year, quarter, month, season, day of week, weekend-or-not, and — most useful for marketing — **recency** (days since that date, the R in RFM analysis).

**What each option gives you:**
- *Season / month / quarter* → seasonality analysis and campaign timing.
- *Day of week & weekend flag* → staffing, send-time optimization.
- *Recency (days since)* → who's gone quiet? A core churn signal.
- *Tenure (months since)* → how long someone has been a customer.
- *Holiday-season flag (Nov–Dec)* → the retail high season, isolated in one 0/1 column.

**Caution:** recency and tenure are computed relative to *today's* date on your computer.""",

"reduce": """**What is category reduction?** Combining rare categories into a single "Other" bucket — e.g. two sales channels that each cover under 5% of customers.

**Why:** statistics computed on tiny groups are unstable (5 customers can flip an average), and rare categories create noisy dummy variables in regression.

**How to use the result:** the new "_reduced" column keeps the frequent categories intact and pools the rare ones — use it instead of the original in models and charts.

**Caution:** only combine categories that make business sense together, and never hide a category a stakeholder specifically cares about.""",

"summary": """**What are summary measures?** Single numbers that describe a whole column: where its center is (mean, median, mode), how spread out it is (range, variance, standard deviation, MAD), and its shape (skewness, kurtosis).

**The one comparison to always make:** **mean vs. median.** If the mean is well above the median, a few large values are pulling the average up (right skew) — common for spend and income. Report the median as "typical" in that case.

**Quick glossary:** *Std. deviation* = typical distance from the mean. *CV* = std ÷ mean, for comparing volatility across different units. *Skewness* > 0 = long right tail. *Kurtosis* high = extreme values are more common than a bell curve predicts.""",

"outliers": """**What are outliers?** Values far from the rest — a $12,000 order among $50 orders.

**Why detect them:** they distort means, standard deviations, correlations, and regressions. But in marketing an outlier is often your **best customer**, not an error!

**How the two methods work:**
- **IQR rule:** flags values beyond 1.5× the interquartile range from the middle 50% box. Robust; what boxplot whiskers use.
- **z-score rule:** flags values more than 3 standard deviations from the mean. Assumes roughly bell-shaped data.
They can disagree — they draw the fence differently, and that's normal.

**What to do with them:** investigate before deleting. Data-entry errors → fix or drop. Real whales → keep them, maybe analyze them separately (or run your analysis with and without them and compare).""",

"boxplot": """**What is a boxplot?** A five-number picture: the box spans the middle 50% of values (Q1 to Q3) with a line at the median; whiskers reach to the most extreme non-outlier values; dots beyond are outliers by the IQR rule.

**Why it's useful:** it's the fastest way to *see* center, spread, skew, and outliers at once — and side-by-side boxplots are the fastest fair comparison of groups.

**How to read it:** a median line off-center in the box, or one long whisker, means skew. Dots = investigate.""",

"standardize": """**What is standardization?** Converting a variable into **z-scores**: z = (value − mean) ÷ standard deviation. The result always has mean 0 and standard deviation 1.

**Why it matters:** it puts variables with different units on one scale. A z of +2 means "unusually high" whether the raw variable was age or dollars — so you can compare apples to oranges.

**Where you'll need it:** any *distance-based* method — KNN and clustering — would otherwise be dominated by whichever variable has the biggest numbers. (This app standardizes automatically inside those tabs, but understanding why is on every exam.)

**How to read a z-score:** 0 = average; ±1 = fairly typical; beyond ±2 = notable; beyond ±3 = rare.

**Caution:** standardizing doesn't fix skew or outliers — an outlier just becomes a huge z-score.""",

"correlation": """**What is correlation?** A number from −1 to +1 measuring how two variables move together. +1 = perfect positive, 0 = no relationship, −1 = perfect negative.

**The three types and when to use each:**
- **Pearson** — the default. Measures *linear* relationships between two **continuous numeric** variables (spend, income). Sensitive to outliers.
- **Spearman** — works on **ranks**, so it suits **ordinal** variables (satisfaction 1–10, star ratings) or numeric data with outliers/curved-but-monotonic relationships.
- **Kendall's tau** — also rank-based like Spearman; more conservative, better for small samples with many tied values.

**Rules of thumb:** |r| < 0.3 weak, 0.3–0.7 moderate, > 0.7 strong (context matters!).

**Caution #1:** correlation ≠ causation — ice-cream sales correlate with drownings (both follow summer). **Caution #2:** Pearson can read a strong *curved* relationship as near-zero; check the scatterplot.""",

"barchart": """**What it shows:** the count (or share) of each category. Best for one categorical variable.

**Reading tips:** sort bars from largest to smallest so the reader doesn't have to; use *relative* frequencies (shares) when comparing datasets of different sizes.""",

"histogram": """**What it shows:** the shape of a numeric variable, by cutting its range into intervals (bins) and counting rows in each.

**The bin-width decision:** too few bins hides the shape; too many shows noise. Try a few settings — the shape that persists is real.

**What to look for:** symmetric vs. skewed, one hump or two (two humps often means two hidden customer segments!), gaps, and outliers.""",

"crosstab_viz": """**What it shows:** how two categorical variables relate — counts of every combination, e.g. region × churned.

**The key trick:** raw counts mislead when groups differ in size. Switch to **row %** to compare *rates* fairly ("what share of each region churned?").

**The stacked bar chart** is the same table as a picture; a 100% stacked version compares rates at a glance.""",

"scatter": """**What it shows:** the relationship between two numeric variables — each dot is one row.

**What to look for:** direction (up/down), strength (tight vs. loose cloud), shape (straight or curved), clusters of dots (hidden segments), and lone dots (outliers).

**Pro move:** color the dots by a categorical variable to see whether the relationship differs by group. Always scatter-plot before you regress.""",

"linechart": """**What it shows:** how a quantity moves over time — revenue by month, signups by week.

**What to look for:** trend (drifting up/down), seasonality (a repeating annual pattern — huge in retail), and one-off spikes (campaigns, holidays, outages).

**Caution:** be wary of the last data point — a partial month looks like a crash.""",

"ttest_one": """**What it tests:** whether the mean of one numeric variable differs from a specific benchmark number — e.g. "is our average satisfaction different from 7?"

**How to read it:** the p-value is the probability of seeing a gap this big if the true mean really equaled the benchmark. **p < 0.05 → conclude the mean differs.**

**Assumes:** roughly bell-shaped data or a decently large sample (n > 30 is usually fine).""",

"ttest_ind": """**What it tests:** whether two **separate groups** have different means — e.g. do app users spend more than non-users?

**Requirements:** a numeric outcome + a categorical variable with exactly two groups. The two groups contain *different* people.

**How to read it:** look at the two group means and the p-value; p < 0.05 → the difference is statistically significant. Welch's version (used by default here) doesn't require equal variances.

**Caution:** with big samples, tiny, commercially meaningless differences become "significant" — always ask if the difference is big enough to *matter*.""",

"ttest_paired": """**What it tests:** whether the mean changed for the **same people measured twice** — before vs. after a campaign, satisfaction in January vs. June.

**Requirements:** two numeric columns that are paired row by row (same customer in both). This is the "dependent samples" t-test.

**Why not the independent test?** Pairing removes person-to-person variation, making the test far more sensitive to real change.

**How to read it:** the mean *difference* and its p-value; p < 0.05 → a real average change.""",

"chisq": """**What it tests:** whether two categorical variables are related — e.g. is churn *independent* of region, or does churn rate genuinely differ by region?

**How it works:** compares the observed crosstab counts with the counts you'd *expect if the variables were unrelated*. Big gaps → big chi-square → small p-value.

**How to read it:** p < 0.05 → the variables are related. **Cramér's V** (0–1) tells you how strongly: <0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, >0.5 strong.

**Caution:** chi-square is unreliable when expected counts are small (rule of thumb: all expected counts ≥ 5 — combine rare categories first). And "related" doesn't say *which* cells drive it — read the row % table.""",

"anova": """**What is ANOVA?** *Analysis of Variance* — the test for comparing means across **three or more groups** (a t-test only handles two).

- **One-way ANOVA:** one grouping factor. "Does average spend differ across the four regions?"
- **Two-way ANOVA:** two factors at once — each factor's own effect (*main effects*) plus, optionally, their **interaction**: does the effect of one factor *depend on* the level of the other? (E.g., maybe email works better in the North but social works better in the South.)
- **N-way:** more factors; each added factor and interaction makes interpretation harder — with many factors, prefer regression with dummies.

**How to read the table:** each row is a factor (or interaction) with an F statistic and p-value. **p < 0.05 → that factor's group means are not all equal.**

**Crucial limitation:** a significant ANOVA says *at least one* group differs — it doesn't say **which**. That's what post-hoc tests are for (Help next to the post-hoc box).""",

"posthoc": """**Why post-hoc tests exist:** after a significant ANOVA, you want to know *which pairs of groups* differ. Running many ordinary t-tests inflates false positives (test 6 pairs at the 5% level and your real error rate is ~26%). Post-hoc procedures compare all pairs while keeping the overall error rate at 5%.

**The menu:**
- **Tukey HSD** — the standard choice for comparing *all pairs* with (roughly) equal group sizes.
- **Scheffé** — the most conservative; safest when group sizes are unequal or you're exploring many comparisons; flags fewer pairs.
- **Bonferroni** — simplest: ordinary pairwise tests with the p-value bar raised (0.05 ÷ number of pairs). Conservative when there are many pairs.

**How to read the output:** each row is a pair of groups; reject = TRUE or p < 0.05 → those two groups genuinely differ.""",

"regression": """**What is OLS regression?** A model of how a numeric outcome (the *dependent variable*, DV) relates to one or more predictors: DV = intercept + b₁·X₁ + b₂·X₂ + …

**How to read the output:**
- **R²** — share of the DV's variation the model explains (0–1). **Adjusted R²** penalizes useless extra predictors — use it when comparing models.
- **F-statistic** — tests whether the model as a whole beats just guessing the mean; its p-value should be < 0.05.
- **Each coefficient** — the expected change in the DV per one-unit increase in that predictor, *holding the others constant* (that "holding constant" is the whole magic). Its **p-value < 0.05** → the effect is statistically distinguishable from zero.
- **Categorical predictors** become dummy variables; each coefficient compares that category to the *baseline* (the category not shown).

**Caution:** regression finds association, not proof of causation; and never extrapolate far outside the range of your data.""",

"diagnostics": """**Why check assumptions?** The regression's p-values and intervals are only trustworthy if its assumptions roughly hold. The five classical checks:

1. **Linearity / correct form** — residuals-vs-fitted should be a patternless cloud. A *curve* → add a squared term or transform.
2. **No multicollinearity** — predictors shouldn't be near-copies of each other. **VIF < 10** is fine (< 5 comfortable). High VIF makes coefficients unstable.
3. **Constant error variance** — no funnel shape in residuals-vs-fitted; **Breusch-Pagan p > 0.05** is a pass.
4. **No autocorrelation** — errors shouldn't follow each other; **Durbin-Watson ≈ 2** is a pass (mostly a time-series concern).
5. **Normal errors** — Q-Q dots on the line; **Jarque-Bera p > 0.05** is a pass. Matters least with large samples.

**Perspective:** real data rarely passes all five perfectly. One borderline test is a footnote, not a crisis — a strong pattern is what demands action.""",

"quadratic": """**What a squared term does:** lets the relationship curve — rise then fall (inverted U) or fall then rise. Example: spending often rises with age to a peak, then declines.

**How to read it:** if the squared term's p-value < 0.05, the curve is real. The **turning point** = −b₁ ÷ (2·b₂) is where the curve peaks/bottoms — only meaningful if it falls inside your data's range.

**Marketing use:** diminishing returns! Ad spend usually helps at a decreasing rate — the turning point suggests where more spend stops paying.""",

"logs": """**Why log-transform?**
- **Log a predictor** → its coefficient becomes the effect of a *1% increase* in that predictor. Good for skewed inputs like income.
- **Log the DV** → every coefficient becomes an approximate *percentage change* in the outcome ("each extra visit lifts spend by ~0.8%"). Managers love percentage language; it also tames right-skewed outcomes.

**Caution:** logs need strictly positive values (the app checks). And remember to interpret in the transformed language — the numbers are no longer plain dollars.""",

"cv": """**Why validate?** A significant p-value says a variable matters *in this sample*. Prediction asks the harsher question: does the model work on data it has never seen?

- **Holdout:** train on 70% of rows, test on the untouched 30%, measure **RMSE** (typical prediction error, in the DV's units). Lower = better.
- **K-fold cross-validation:** repeats that five times with rotating test sets and averages — more trustworthy than a single lucky/unlucky split.

**How to use it:** compare two candidate models; prefer the one with lower CV RMSE. If a fancier model wins in-sample (higher R²) but loses in CV, it's *overfitting* — memorizing noise.""",

"logistic": """**What is logistic regression?** The regression for a **yes/no outcome** (churned or not, upgraded or not). It models the *probability* of a 1, via an S-shaped curve that stays between 0 and 1 — a straight line would happily predict 140%.

**How to read it:** raw coefficients are hard to read, so look at **odds ratios** (OR = e^coefficient):
- OR = 2.7 → that factor multiplies the odds of a "1" by 2.7.
- OR = 0.72 → each unit *cuts* the odds by 28%.
- OR ≈ 1 with p > 0.05 → no detectable effect.

**Pseudo R²** is only loosely like regular R² — don't expect big values; 0.2–0.4 is often a good model.

**Key difference from OLS:** effects are *not constant* — the same one-unit change moves probability most near 50% and barely at the extremes. Use the probability calculator to feel this.""",

"cutoff": """**What is a cutoff?** The model outputs probabilities; the cutoff turns them into yes/no decisions (probability ≥ cutoff → predict "1").

**The three scores:**
- **Accuracy** — share of all predictions that are right. Misleading when one class is rare!
- **Sensitivity (recall)** — share of *actual 1s* you caught. The churn-catcher's metric.
- **Specificity** — share of actual 0s correctly left alone.

**The trade-off:** lowering the cutoff catches more 1s (sensitivity ↑) at the cost of false alarms (specificity ↓).

**How to choose — it's a business decision, not a statistical one:** compare the cost of a miss vs. a false alarm. If a lost customer costs $200 and a retention email costs $0.10, use a LOW cutoff and happily accept "worse accuracy". Accuracy is not the goal; money is.""",

"knn": """**K-Nearest Neighbors (KNN):** classifies each customer by a vote of the *k most similar* customers ("judgment by peer group"). No formula is learned — the data itself is the model.

**Why features must be standardized:** "similar" means *distance*, and distance is meaningless if one feature is in thousands of dollars and another is 1–10. (Done automatically here.)

**Choosing k:** small k = flexible but noisy; large k = smooth but blurry. Odd k avoids tied votes.""",

"nb": """**Naive Bayes:** flips the question — given what class-1 profiles typically look like vs. class-0 profiles, which is *this* profile more consistent with? Uses Bayes' rule.

**Why "naive":** it assumes the features don't interact — almost always wrong, yet the method is fast and often surprisingly competitive, especially as a baseline.""",

"tree": """**Classification/regression tree:** a flowchart of if-then splits (e.g. "satisfaction ≤ 6.5?"), each chosen to make the resulting groups as pure as possible.

**Why managers love it:** you can *read* the rules directly off the plot — no statistics degree needed.

**The depth setting:** deeper = more detailed but more likely to memorize noise (overfit). Depth 3–5 is usually the sweet spot for interpretability.""",

"rf": """**Random forest:** hundreds of slightly-different trees, each trained on a random slice of the data and features, voting together. Individual trees overfit in different ways; the vote cancels their mistakes — wisdom of crowds.

**Feature importance:** measures how much each feature contributed to the forest's decisions — a quick "what drives this outcome?" chart.

**Trade-off vs. one tree:** usually more accurate, but you lose the readable flowchart.""",

"clustering": """**What is clustering?** Finding natural groups in data **without** a target variable — the algorithmic version of market segmentation. No answer key exists; the algorithm proposes groups and *you* judge whether they're meaningful.

**The two classic methods (both available here):**
- **K-means** — you pick k; it finds k cluster centers and assigns each row to the nearest. Fast, scales to big data. Downside: you must choose k (use the elbow plot) and it prefers round, similar-sized clusters.
- **Hierarchical (agglomerative)** — starts with every row as its own cluster and repeatedly merges the closest pair, producing the **dendrogram** (the full merge tree). You see the structure at *every* number of clusters and cut where the gaps are tallest. Downside: slow on large data (this app uses a sample for the dendrogram).

**Does the choice matter?** For well-separated data they usually agree (compare the profiles!). K-means is the production workhorse; hierarchical is the exploration tool that helps you *choose* k in the first place.

**Standardization is essential** (done automatically): clustering runs on distance, and a big-scale variable would otherwise dominate.""",

"elbow": """**The elbow plot:** k-means "inertia" (total within-cluster distance) for k = 1, 2, 3… Inertia always falls as k rises, but with diminishing returns — the **elbow** is where the curve bends from steep to flat: adding another cluster stops buying much. That's your k.

**The silhouette score** (−1 to 1) rates how well-separated the final clusters are: ~0 = no real structure, 0.3–0.5 = reasonable, > 0.5 = solid. Use both together.""",

"dendrogram": """**How to read a dendrogram:** it's the merge history of hierarchical clustering drawn as a tree. Height = how *far apart* two clusters were when merged.

**The rule:** cut horizontally where the vertical gaps are tallest — big gaps mean the merge joined genuinely different groups. Count the branches below your cut: that's your number of segments.""",

"textmining": """**Why clean text first?** Raw text is messy: "Shipping", "shipping!", "shipped" should count as one idea. Cleaning = lowercase → remove punctuation → remove **stopwords** (the, a, and…) → **stem** (ship/shipping/shipped → "ship").

**The word cloud** sizes words by frequency — the crowd's volume knob. It's a starting point, not an analysis: follow up with sentiment and topics.""",

"sentiment": """**What is sentiment analysis?** Scoring text by tone. TextBlob gives each review a **polarity** from −1 (very negative) to +1 (very positive) using a dictionary of scored words.

**Sanity-check the tool:** if your data has star ratings, correlate polarity with stars — a healthy correlation (say > 0.6) means the scores are credible for *your* text.

**Caution:** dictionary methods miss sarcasm, negation subtleties, and domain slang ("this coffee is sick!"). Treat scores as a strong signal, not gospel.""",

"aspects": """**Aspect-based sentiment — the money slide.** Overall sentiment hides the story: customers can love your product AND hate your shipping. Splitting sentiment *by aspect* (price, quality, shipping, service…) tells you exactly what to fix and what to advertise.

**How this works here:** each aspect has a keyword list (editable — tailor it to your product!). Reviews mentioning a keyword count toward that aspect; we average their polarity.

**How to use the output:** negative bars = operational to-do list, ranked. Positive bars = strengths for your next campaign's copy.""",

"lda": """**Topic modeling (LDA)** finds recurring themes across many documents with **no labels at all** — it clusters words that co-occur. You get word lists; *you* supply the human-readable topic names.

**How to read it:** scan each topic's top words and name the theme ("shipping complaints", "price comparisons"). If topics look muddled, try a different number of topics.

**Same division of labor as clustering:** the algorithm finds structure; the marketer gives it meaning.""",

"image": """**Why analyze images?** Your brand *is* partly its visuals. Batch-analyzing social posts or customer photos reveals whether your feed is visually consistent and on-palette.

**What each metric means:**
- **Dominant colors** — the k most common colors (found by running k-means on the pixels — yes, the same clustering algorithm, reused on colors!). Compare against your brand palette.
- **Brightness** (0–255) — dark and moody vs. light and airy. Consistency matters more than the level.
- **Colorfulness** — a standard measure (Hasler–Süsstrunk) of how saturated/varied the colors are; muted vs. vivid.

**How marketers use it:** audit a month of posts — if dominant palettes and brightness jump around wildly, the feed reads as inconsistent; if a competitor's palette is creeping into your user-generated content, you'll see it.""",

"addvar": """**"Add to data" buttons** save a *derived* column into the working dataset so every other tab can use it:
- **z-score** — the standardized version (see the Standardization tab's Help).
- **Outlier flag** — 1 if the row is an IQR-rule outlier on this variable, else 0. Useful for filtering or as a robustness check.
- **Percentile rank** — where each row sits from 0–100 (a value at the 90th percentile beats 90% of rows). Great for building scores and tiers (e.g. "top-decile spenders").""",
}

# ---------------------------------------------------------------- chart style helper
PALETTES = ["copper", "viridis", "tab10", "Set2", "plasma", "coolwarm", "Pastel1", "cividis"]

def chart_options(key, single_series=True):
    """Customization expander; returns a dict of style options."""
    with st.expander("🎨 Customize this chart (colors, titles, labels)"):
        c1, c2, c3 = st.columns(3)
        opts = {}
        opts["title"] = c1.text_input("Chart title", value="", key=f"{key}_t")
        opts["xlabel"] = c2.text_input("X-axis label", value="", key=f"{key}_x")
        opts["ylabel"] = c3.text_input("Y-axis label", value="", key=f"{key}_y")
        c4, c5, c6 = st.columns(3)
        if single_series:
            opts["color"] = c4.color_picker("Bar/line/point color", "#6B3F1D", key=f"{key}_c")
        opts["palette"] = c5.selectbox("Color palette (multi-series charts)", PALETTES, key=f"{key}_p")
        opts["rot"] = c6.slider("Rotate x labels (°)", 0, 90, 30, key=f"{key}_r")
        c7, c8, c9 = st.columns(3)
        opts["grid"] = c7.checkbox("Show gridlines", value=True, key=f"{key}_g")
        opts["sort"] = c8.checkbox("Sort bars by size", value=True, key=f"{key}_s")
        opts["labels"] = c9.checkbox("Show value labels on bars", value=False, key=f"{key}_l")
    return opts

def apply_style(ax, opts, default_title="", default_x="", default_y=""):
    ax.set_title(opts.get("title") or default_title)
    ax.set_xlabel(opts.get("xlabel") or default_x)
    ax.set_ylabel(opts.get("ylabel") or default_y)
    if opts.get("grid"):
        ax.grid(axis="y", alpha=.3)
    plt.setp(ax.get_xticklabels(), rotation=opts.get("rot", 30), ha="right" if opts.get("rot", 30) else "center")

def bar_value_labels(ax):
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:,.0f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown(
    """
    <div style="background:linear-gradient(135deg,#6B3F1D 0%,#8a5a2e 100%);
                border-radius:14px;padding:18px 16px;margin-bottom:14px;
                text-align:center;color:white;">
      <div style="width:58px;height:58px;border-radius:50%;background:white;
                  color:#6B3F1D;font-weight:800;font-size:22px;line-height:58px;
                  margin:0 auto 8px auto;font-family:Georgia,serif;">DN</div>
      <div style="font-size:22px;font-weight:800;letter-spacing:.5px;">Dr.&nbsp;Nik</div>
      <div style="font-size:13px;opacity:.92;margin-top:2px;">Marketing Intelligence Lab</div>
    </div>
    """,
    unsafe_allow_html=True,
)

BUILTINS = {
    "Marketing customers (lecture data)": "marketing_customers.csv",
    "NovaGlow reviews (lecture NLP data)": "novaglow_reviews.csv",
    "BrewBox transactions (Assignment 1)": "brewbox_transactions.csv",
    "BrewBox ad markets (Assignment 2)": "brewbox_ad_markets.csv",
    "BrewBox upgrades (Assignment 2)": "brewbox_upgrades.csv",
    "BrewBox customers (Assignment 3)": "brewbox_customers.csv",
    "BrewBox reviews (Assignment 3)": "brewbox_reviews.csv",
}
src = st.sidebar.radio("Where is your data?", ["Course dataset", "Upload my own CSV"])
if src == "Course dataset":
    choice = st.sidebar.selectbox("Pick a course dataset", list(BUILTINS.keys()))
    df_raw = load_builtin(BUILTINS[choice]).copy()
    st.sidebar.success(f"Loaded **{BUILTINS[choice]}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
else:
    up = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if up is None:
        st.sidebar.info("Waiting for a file… (or switch to a course dataset)")
        df_raw = load_builtin("marketing_customers.csv").copy()
        st.sidebar.caption("Showing the lecture dataset until you upload.")
    else:
        df_raw = pd.read_csv(up)
        st.sidebar.success(f"Loaded your file — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

if "work_key" not in st.session_state or st.session_state.get("work_src") != (src, str(df_raw.shape)):
    st.session_state.work = df_raw.copy()
    st.session_state.work_src = (src, str(df_raw.shape))
    st.session_state.work_key = True
if st.sidebar.button("🔄 Reset data (undo all wrangling)"):
    st.session_state.work = df_raw.copy()
    st.sidebar.success("Data reset to the original file.")
df = st.session_state.work

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How this app works**\n\n"
    "1. Pick a dataset (left).\n"
    "2. Pick a tab for the analysis you need.\n"
    "3. Open the 📖 instructions and the ℹ️ Help buttons inside the tab.\n"
    "4. Choose your options and click the ▶ Run button.\n\n"
    "Changes you make in the *Data & Wrangling* tab carry over to every other tab, "
    "until you press **Reset data**."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:11px;color:#8a8a8a;text-align:center;'>{BRAND_NOTICE}<br>"
    "Do not reuse or redistribute without permission.</div>",
    unsafe_allow_html=True,
)

st.title("Marketing Intelligence Lab")
st.markdown("<div style='margin-top:-10px;color:#6B3F1D;font-weight:600;'>by Dr. Nik</div>", unsafe_allow_html=True)
st.caption("The full marketing-analytics toolkit — no coding required, with ℹ️ Help on every element. "
           f"Data currently loaded: **{df.shape[0]:,} rows × {df.shape[1]} columns**")

tabs = st.tabs(["🧹 Data & Wrangling", "📋 Summary Measures", "⚖️ Standardization", "🔗 Correlation",
                "📊 Visualization", "🧪 t-Tests & Chi-Square", "🔬 ANOVA", "📈 Regression",
                "🎯 Logistic", "🤖 Data Mining", "🧩 Clustering", "💬 Text & NLP", "🖼️ Image Analysis"])

# ================================================================ TAB 1: WRANGLING
with tabs[0]:
    instructions("""
**What this tab does:** inspect your data and clean it — the analyst's first hour on any job.
Work top-to-bottom: inspect → fix missing values → filter if needed → create transformed columns.
Every ℹ️ Help button explains the concept next to it in plain language.

⚠️ Everything you do here **changes the working data used by all other tabs** — that's the point!
Use the sidebar's *Reset data* button to start over.
""")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Preview (first 10 rows)")
        st.dataframe(df.head(10), width="stretch")
    with c2:
        st.subheader("Column info")
        info = pd.DataFrame({
            "type": df.dtypes.astype(str),
            "role (plain English)": [classify_role(df, c) for c in df.columns],
            "missing": df.isna().sum(),
            "unique": df.nunique(),
        })
        st.dataframe(info, width="stretch")
        st.caption("**role** tells you how to treat the column: numeric columns feed regressions and "
                   "correlations; categorical columns feed bar charts, crosstabs, and ANOVA factors.")

    st.markdown("---")
    w1, w2 = st.columns(2)

    with w1:
        section("💧 Missing values", "missing")
        miss = [c for c in df.columns if df[c].isna().any()]
        if not miss:
            st.success("No missing values in the working data. Nothing to do here!")
        else:
            mcol = st.selectbox("Column with missing values", miss, key="miss_col")
            method = st.radio("How to handle it", ["Impute with median", "Impute with mean", "Drop those rows (omission)"],
                              key="miss_method", horizontal=True)
            if st.button("▶ Apply missing-data fix", key="btn_miss"):
                if method == "Drop those rows (omission)":
                    before = len(df)
                    st.session_state.work = df.dropna(subset=[mcol]).reset_index(drop=True)
                    st.success(f"Dropped {before - len(st.session_state.work):,} rows with missing **{mcol}**.")
                else:
                    if pd.api.types.is_numeric_dtype(df[mcol]):
                        val = df[mcol].median() if "median" in method else df[mcol].mean()
                        st.session_state.work[mcol] = df[mcol].fillna(val)
                        st.success(f"Filled blanks in **{mcol}** with {val:,.2f}.")
                    else:
                        st.error("That column isn't numeric — use omission instead.")
                st.rerun()

        section("🔪 Filter / subset", "filter")
        fcol = st.selectbox("Filter on which column?", df.columns.tolist(), key="f_col")
        if pd.api.types.is_numeric_dtype(df[fcol]):
            op = st.selectbox("Condition", ["≥", "≤", "="], key="f_op")
            val = st.number_input("Value", value=float(df[fcol].median() if df[fcol].notna().any() else 0), key="f_val")
            mask = {"≥": df[fcol] >= val, "≤": df[fcol] <= val, "=": df[fcol] == val}[op]
        else:
            keep = st.multiselect("Keep rows where value is one of…", sorted(df[fcol].dropna().unique().tolist()), key="f_keep")
            mask = df[fcol].isin(keep) if keep else pd.Series(True, index=df.index)
        st.caption(f"This filter would keep **{int(mask.sum()):,}** of {len(df):,} rows.")
        if st.button("▶ Apply filter", key="btn_filter"):
            st.session_state.work = df[mask].reset_index(drop=True)
            st.rerun()

        section("🧩 Combine rare categories into 'Other'", "reduce")
        ccols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and 2 < df[c].nunique() <= 25]
        if ccols:
            rcol = st.selectbox("Categorical column", ccols,
                                index=default_ix(ccols, ["acquisition_channel", "channel"]), key="r_col")
            counts = df[rcol].value_counts()
            st.dataframe(counts.rename("count"), width="stretch")
            n_rare = st.slider("Combine the N least frequent categories", 1, max(1, len(counts) - 2), 2, key="r_n")
            rare = counts.tail(n_rare).index.tolist()
            st.caption(f"Will combine: {', '.join(map(str, rare))}")
            if st.button("▶ Create reduced column", key="btn_reduce"):
                st.session_state.work[f"{rcol}_reduced"] = df[rcol].replace({r: "Other" for r in rare})
                st.success(f"Created **{rcol}_reduced**.")
                st.rerun()
        else:
            st.caption("No suitable categorical column found.")

    with w2:
        section("🗂️ Bin a numeric column into categories", "binning")
        bincol_opts = num_cols(df)
        if bincol_opts:
            bcol = st.selectbox("Numeric column to bin", bincol_opts,
                                index=default_ix(bincol_opts, ["age"]), key="b_col")
            nbins = st.slider("Number of equal-width bins", 2, 8, 4, key="b_n")
            if st.button("▶ Create binned column", key="btn_bin"):
                st.session_state.work[f"{bcol}_group"] = pd.cut(df[bcol], bins=nbins).astype(str)
                st.success(f"Created **{bcol}_group**.")
                st.rerun()

        section("📅 Date wrangling toolbox", "dates")
        datelike = date_cols(df)
        if datelike:
            dcol = st.selectbox("Date column", datelike, key="d_col")
            dops = st.multiselect(
                "What to extract (each becomes a new column)",
                ["Season", "Month name", "Quarter", "Year", "Day of week", "Weekend flag (0/1)",
                 "Recency: days since this date", "Tenure: months since this date",
                 "Holiday-season flag (Nov–Dec, 0/1)"],
                default=["Season"], key="d_ops")
            if st.button("▶ Create date column(s)", key="btn_season"):
                d = pd.to_datetime(df[dcol], errors="coerce")
                today = pd.Timestamp.today().normalize()
                smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                        6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
                made = []
                for op in dops:
                    if op == "Season":
                        st.session_state.work[f"{dcol}_season"] = d.dt.month.map(smap); made.append(f"{dcol}_season")
                    elif op == "Month name":
                        st.session_state.work[f"{dcol}_month"] = d.dt.month_name(); made.append(f"{dcol}_month")
                    elif op == "Quarter":
                        st.session_state.work[f"{dcol}_quarter"] = "Q" + d.dt.quarter.astype("Int64").astype(str); made.append(f"{dcol}_quarter")
                    elif op == "Year":
                        st.session_state.work[f"{dcol}_year"] = d.dt.year; made.append(f"{dcol}_year")
                    elif op == "Day of week":
                        st.session_state.work[f"{dcol}_dayofweek"] = d.dt.day_name(); made.append(f"{dcol}_dayofweek")
                    elif op.startswith("Weekend"):
                        st.session_state.work[f"{dcol}_weekend"] = (d.dt.dayofweek >= 5).astype(int); made.append(f"{dcol}_weekend")
                    elif op.startswith("Recency"):
                        st.session_state.work[f"{dcol}_days_since"] = (today - d).dt.days; made.append(f"{dcol}_days_since")
                    elif op.startswith("Tenure"):
                        st.session_state.work[f"{dcol}_months_since"] = ((today - d).dt.days / 30.44).round(1); made.append(f"{dcol}_months_since")
                    elif op.startswith("Holiday"):
                        st.session_state.work[f"{dcol}_holiday_season"] = d.dt.month.isin([11, 12]).astype(int); made.append(f"{dcol}_holiday_season")
                st.success("Created: " + ", ".join(f"**{m}**" for m in made))
                st.rerun()
        else:
            st.caption("No column with 'date' in its name was found in this dataset.")

# ================================================================ TAB 2: SUMMARY MEASURES
with tabs[1]:
    instructions("""
**What this tab does:** the full set of descriptive statistics for any numeric column, plus outlier
detection and boxplots. Use the ℹ️ Help buttons for what each measure means, and the ➕ **Add to data**
buttons to save derived columns (z-scores, outlier flags, percentile ranks) for use in other tabs.
""")
    ncols = num_cols(df)
    if not ncols:
        st.warning("This dataset has no numeric columns.")
    else:
        v = st.selectbox("Numeric variable to summarize", ncols,
                         index=default_ix(ncols, ["annual_spend", "monthly_spend", "sales_k"]), key="s_var")
        x = df[v].dropna()
        section(f"Summary measures for `{v}`", "summary")
        mode_val = x.mode().iloc[0] if not x.mode().empty else np.nan
        stats_tbl = pd.DataFrame({
            "Measure": ["Count", "Mean", "Median", "Mode", "25th percentile", "75th percentile", "Range",
                        "Mean Absolute Deviation", "Variance", "Std. deviation",
                        "Coefficient of Variation", "Skewness", "Kurtosis"],
            "Value": [f"{len(x):,}", f"{x.mean():,.2f}", f"{x.median():,.2f}", f"{mode_val:,.2f}",
                      f"{x.quantile(.25):,.2f}", f"{x.quantile(.75):,.2f}", f"{x.max()-x.min():,.2f}",
                      f"{(x-x.mean()).abs().mean():,.2f}", f"{x.var():,.2f}", f"{x.std():,.2f}",
                      f"{x.std()/x.mean():.3f}" if x.mean() != 0 else "—",
                      f"{x.skew():.2f}", f"{x.kurtosis():.2f}"],
        })
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(stats_tbl, width="stretch", hide_index=True)
        with c2:
            section("Boxplot", "boxplot")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.boxplot(x, vert=False)
            ax.set_title(f"Boxplot of {v}"); ax.set_xlabel(v)
            show_fig(fig)
            g = st.selectbox("Compare boxplots by group (optional)", ["(none)"] + cat_cols(df), key="s_group")
            if g != "(none)":
                fig, ax = plt.subplots(figsize=(5, 3))
                lvls = df[g].dropna().unique()
                ax.boxplot([df.loc[df[g] == lv, v].dropna() for lv in lvls], tick_labels=[str(l) for l in lvls])
                ax.set_title(f"{v} by {g}"); plt.xticks(rotation=30)
                show_fig(fig)

        st.markdown("---")
        section("🚨 Outlier detection", "outliers")
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = df[(df[v] < lo) | (df[v] > hi)]
        z = (x - x.mean()) / x.std()
        z_out = df.loc[z[abs(z) > 3].index]
        metric_row([("IQR-rule outliers", f"{len(iqr_out):,}"), ("|z| > 3 outliers", f"{len(z_out):,}"),
                    ("Upper IQR fence", f"{hi:,.1f}")])
        if len(iqr_out):
            st.dataframe(iqr_out.head(15), width="stretch")

        st.markdown("---")
        section("➕ Add derived columns to the data", "addvar")
        st.caption("Each button saves a new column to the working data, usable in every other tab.")
        a1, a2, a3 = st.columns(3)
        if a1.button(f"▶ Add z-score:  {v}_z", key="btn_std"):
            st.session_state.work[f"{v}_z"] = (df[v] - df[v].mean()) / df[v].std()
            st.success(f"Created **{v}_z**."); st.rerun()
        if a2.button(f"▶ Add outlier flag:  {v}_outlier", key="btn_outflag"):
            st.session_state.work[f"{v}_outlier"] = ((df[v] < lo) | (df[v] > hi)).astype(int)
            st.success(f"Created **{v}_outlier** (1 = IQR outlier)."); st.rerun()
        if a3.button(f"▶ Add percentile rank:  {v}_pctile", key="btn_pctile"):
            st.session_state.work[f"{v}_pctile"] = (df[v].rank(pct=True) * 100).round(1)
            st.success(f"Created **{v}_pctile** (0–100)."); st.rerun()

# ================================================================ TAB 3: STANDARDIZATION
with tabs[2]:
    instructions("""
**What this tab does:** converts numeric variables into z-scores and lets you *see* what standardization
does. Open the ℹ️ Help for the concept — it's one of the most-used ideas in the whole course
(KNN and clustering depend on it).
""")
    section("⚖️ Standardize variables (z-scores)", "standardize")
    ncols3 = num_cols(df)
    if ncols3:
        pick = st.multiselect("Variables to standardize", ncols3,
                              default=[c for c in ["age", "income_k", "annual_spend", "monthly_spend"] if c in ncols3][:2],
                              key="std_vars")
        if pick:
            zdf = pd.DataFrame({c: (df[c] - df[c].mean()) / df[c].std() for c in pick})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before (original units)** — different scales, hard to compare:")
                st.dataframe(df[pick].describe().loc[["mean", "std", "min", "max"]].round(2), width="stretch")
            with c2:
                st.markdown("**After (z-scores)** — every variable now has mean 0, sd 1:")
                st.dataframe(zdf.describe().loc[["mean", "std", "min", "max"]].round(2), width="stretch")
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            for c in pick:
                axes[0].hist(df[c].dropna(), bins=30, alpha=.55, label=c)
                axes[1].hist(zdf[c].dropna(), bins=30, alpha=.55, label=c)
            axes[0].set_title("Original scales"); axes[1].set_title("Standardized (z-scores)")
            axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
            show_fig(fig)
            st.caption("Left: the variables live on totally different scales. Right: after standardization "
                       "they're directly comparable — a z of +2 is 'unusually high' for any of them.")
            if st.button("▶ Add these z-score columns to the data", key="btn_std_tab"):
                for c in pick:
                    st.session_state.work[f"{c}_z"] = zdf[c]
                st.success("Added: " + ", ".join(f"**{c}_z**" for c in pick))
                st.rerun()
    else:
        st.warning("No numeric columns in this dataset.")

# ================================================================ TAB 4: CORRELATION
with tabs[3]:
    instructions("""
**What this tab does:** measures how variables move together, three ways. Read the ℹ️ Help first —
choosing the right correlation type for your variable types is exactly the kind of judgment this
course trains: **Pearson** for two continuous numerics, **Spearman** for ordinal variables (like 1–10
satisfaction or star ratings) or when outliers/curves are a worry, **Kendall** for small samples with ties.
""")
    section("🔗 Correlation", "correlation")
    ncols4 = num_cols(df)
    if len(ncols4) >= 2:
        method = st.radio("Correlation type",
                          ["Pearson (two continuous numeric variables)",
                           "Spearman (ordinal variables, or numeric with outliers/curves)",
                           "Kendall (like Spearman; small samples, many ties)"],
                          key="cor_method")
        mkey = method.split(" ")[0].lower()
        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Variable 1", ncols4, index=default_ix(ncols4, ["income_k", "monthly_spend"]), key="cor_a")
            b = st.selectbox("Variable 2", [c for c in ncols4 if c != a],
                             index=default_ix([c for c in ncols4 if c != a], ["annual_spend", "satisfaction"]), key="cor_b")
            both = df[[a, b]].dropna()
            r = both[a].corr(both[b], method=mkey)
            st.metric(f"{mkey.title()} correlation of {a} & {b}", f"{r:.3f}")
            absr = abs(r)
            verdict = "very weak / none" if absr < .1 else "weak" if absr < .3 else "moderate" if absr < .7 else "strong"
            st.caption(f"Strength: **{verdict}** · Direction: **{'positive' if r >= 0 else 'negative'}** · "
                       "Remember: correlation is not causation.")
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.scatter(both[a], both[b], s=9, alpha=.5, color="#6B3F1D")
            ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title("Always look at the scatterplot too")
            show_fig(fig)
        with c2:
            st.markdown("**Correlation matrix** (all numeric variables, same method)")
            sel = st.multiselect("Variables for the matrix", ncols4,
                                 default=[c for c in ncols4 if c != "customer_id"][:6], key="cor_matrix_vars")
            if len(sel) >= 2:
                M = df[sel].corr(method=mkey).round(2)
                fig, ax = plt.subplots(figsize=(5.4, 4.2))
                im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(sel)), sel, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(sel)), sel, fontsize=8)
                for i in range(len(sel)):
                    for j in range(len(sel)):
                        ax.text(j, i, f"{M.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
                fig.colorbar(im, shrink=.8)
                ax.set_title(f"{mkey.title()} correlation heatmap")
                show_fig(fig)
    else:
        st.warning("Need at least two numeric columns.")

# ================================================================ TAB 5: VISUALIZATION
with tabs[4]:
    instructions("""
**What this tab does:** every core chart type, now with a full 🎨 customization panel — palettes,
your own colors, titles, axis labels, sorting, and value labels — so charts are presentation-ready.
Pick a chart type, use its ℹ️ Help to know what it's for, and style it.
""")
    kind = st.radio("Chart type", ["Bar chart (one categorical)", "Histogram (one numeric)",
                                   "Contingency table + stacked bars (two categoricals)",
                                   "Scatterplot (two numerics)", "Line chart over time"],
                    horizontal=True, key="viz_kind")

    if kind == "Bar chart (one categorical)":
        section("Bar chart & frequency table", "barchart")
        opts_v = chart_options("viz_bar_opts")
        opts_list = cat_cols(df)
        if opts_list:
            c = st.selectbox("Categorical column", opts_list, key="viz_bar")
            counts = df[c].value_counts()
            if not opts_v.get("sort", True):
                counts = counts.sort_index()
            t1, t2 = st.columns([1, 2])
            with t1:
                st.dataframe(pd.DataFrame({"count": counts, "share": (counts / counts.sum()).round(3)}),
                             width="stretch")
            with t2:
                fig, ax = plt.subplots(figsize=(6.5, 3.4))
                counts.plot(kind="bar", ax=ax, color=opts_v.get("color", "#6B3F1D"))
                apply_style(ax, opts_v, f"Frequency of {c}", c, "count")
                if opts_v.get("labels"):
                    bar_value_labels(ax)
                show_fig(fig)
        else:
            st.warning("No categorical columns found.")

    elif kind == "Histogram (one numeric)":
        section("Histogram", "histogram")
        opts_v = chart_options("viz_hist_opts")
        c = st.selectbox("Numeric column", num_cols(df), key="viz_hist")
        bins = st.slider("Number of bins (intervals)", 5, 80, 30, key="viz_bins")
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.hist(df[c].dropna(), bins=bins, color=opts_v.get("color", "#6B3F1D"), edgecolor="white")
        apply_style(ax, opts_v, f"Histogram of {c}", c, "frequency")
        show_fig(fig)
        sk = df[c].skew()
        st.caption(f"Skewness = {sk:.2f} → " + ("right-skewed: a long tail of large values." if sk > 0.5 else
                   "left-skewed: a long tail of small values." if sk < -0.5 else "roughly symmetric."))

    elif kind.startswith("Contingency"):
        section("Contingency table & stacked bars", "crosstab_viz")
        opts_v = chart_options("viz_ct_opts", single_series=False)
        opts_list = cat_cols(df)
        if len(opts_list) >= 2:
            a = st.selectbox("Rows (categorical)", opts_list, key="viz_ct_a")
            b = st.selectbox("Columns (categorical)", [c for c in opts_list if c != a], key="viz_ct_b")
            pct = st.toggle("Show 100% stacked (compare rates, not counts)", key="viz_ct_pct")
            ct = pd.crosstab(df[a], df[b])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Counts**"); st.dataframe(ct, width="stretch")
            with c2:
                st.markdown("**Row %**")
                st.dataframe((pd.crosstab(df[a], df[b], normalize="index") * 100).round(1), width="stretch")
            plot_df = ct.div(ct.sum(axis=1), axis=0) * 100 if pct else ct
            fig, ax = plt.subplots(figsize=(7.5, 3.4))
            plot_df.plot(kind="bar", stacked=True, ax=ax, colormap=opts_v.get("palette", "copper"))
            apply_style(ax, opts_v, f"{a} by {b}" + (" (100% stacked)" if pct else " (stacked counts)"),
                        a, "%" if pct else "count")
            ax.legend(fontsize=8, title=b)
            show_fig(fig)
        else:
            st.warning("Need at least two categorical columns.")

    elif kind == "Scatterplot (two numerics)":
        section("Scatterplot", "scatter")
        opts_v = chart_options("viz_sc_opts", single_series=False)
        ncs = num_cols(df)
        a = st.selectbox("X axis", ncs, index=default_ix(ncs, ["income_k", "social_spend_k"]), key="viz_sc_x")
        b = st.selectbox("Y axis", [c for c in ncs if c != a],
                         index=default_ix([c for c in ncs if c != a], ["annual_spend", "sales_k"]), key="viz_sc_y")
        color_opt = st.selectbox("Color points by (optional)", ["(none)"] + cat_cols(df), key="viz_sc_c")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        if color_opt != "(none)":
            cmap = plt.get_cmap(opts_v.get("palette", "tab10"))
            for i, lv in enumerate(df[color_opt].dropna().unique()):
                sub = df[df[color_opt] == lv]
                ax.scatter(sub[a], sub[b], s=10, alpha=.55, label=str(lv), color=cmap(i % 10))
            ax.legend(fontsize=8, title=color_opt)
        else:
            ax.scatter(df[a], df[b], s=10, alpha=.55, color="#6B3F1D")
        apply_style(ax, opts_v, f"{b} vs {a}", a, b)
        show_fig(fig)

    else:
        section("Line chart over time", "linechart")
        opts_v = chart_options("viz_line_opts")
        datelike = date_cols(df)
        if datelike:
            dcol = st.selectbox("Date column", datelike, key="viz_line_d")
            val = st.selectbox("What to plot per month", ["Row count"] + num_cols(df), key="viz_line_v")
            d = pd.to_datetime(df[dcol], errors="coerce")
            grp = df.assign(_m=d.dt.to_period("M").astype(str))
            series = grp.groupby("_m").size() if val == "Row count" else grp.groupby("_m")[val].sum()
            fig, ax = plt.subplots(figsize=(8.5, 3.4))
            series.plot(ax=ax, marker="o", color=opts_v.get("color", "#6B3F1D"))
            apply_style(ax, opts_v, f"{val} by month", "month", val)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            show_fig(fig)
        else:
            st.warning("No date column found in this dataset.")

# ================================================================ TAB 6: T-TESTS & CHI-SQUARE
with tabs[5]:
    instructions("""
**What this tab does:** the classic hypothesis tests for comparing means and checking whether two
categorical variables are related. Pick the test that matches your question and variable types —
each ℹ️ Help explains when that test applies and how to read the result. The universal reading rule:
**p < 0.05 → the difference/relationship is statistically significant.**
""")
    from scipy import stats as sps
    test = st.radio("Which test?", ["One-sample t-test", "Independent-samples t-test",
                                    "Paired (dependent) samples t-test", "Crosstab + Chi-square"],
                    horizontal=True, key="tt_kind")
    ncols6 = num_cols(df)

    if test == "One-sample t-test":
        section("One-sample t-test", "ttest_one")
        v = st.selectbox("Numeric variable", ncols6, index=default_ix(ncols6, ["satisfaction"]), key="tt1_v")
        mu = st.number_input("Benchmark value to test against", value=float(df[v].median()), key="tt1_mu")
        if st.button("▶ Run one-sample t-test", key="btn_tt1", type="primary"):
            x = df[v].dropna()
            t, p = sps.ttest_1samp(x, mu)
            metric_row([("Sample mean", f"{x.mean():.3f}"), ("Benchmark", f"{mu:g}"),
                        ("t-statistic", f"{t:.3f}"), ("p-value", f"{p:.4f}")])
            st.success(f"Conclusion: the mean of **{v}** {'**differs** from' if p < .05 else 'is **not significantly different** from'} "
                       f"{mu:g} (p = {p:.4f}).")

    elif test == "Independent-samples t-test":
        section("Independent-samples t-test", "ttest_ind")
        v = st.selectbox("Numeric outcome", ncols6, index=default_ix(ncols6, ["annual_spend", "monthly_spend"]), key="tt2_v")
        gopts = [c for c in cat_cols(df) if df[c].nunique() >= 2]
        g = st.selectbox("Grouping variable", gopts, key="tt2_g")
        lvls = sorted(df[g].dropna().unique().tolist())
        pick = st.multiselect("Pick exactly TWO groups to compare", lvls, default=lvls[:2], key="tt2_lvls")
        if st.button("▶ Run independent t-test", key="btn_tt2", type="primary"):
            if len(pick) != 2:
                st.error("Pick exactly two groups.")
            else:
                a = df.loc[df[g] == pick[0], v].dropna()
                b = df.loc[df[g] == pick[1], v].dropna()
                t, p = sps.ttest_ind(a, b, equal_var=False)  # Welch
                metric_row([(f"Mean — {pick[0]}", f"{a.mean():,.2f}"), (f"Mean — {pick[1]}", f"{b.mean():,.2f}"),
                            ("t-statistic (Welch)", f"{t:.3f}"), ("p-value", f"{p:.4f}")])
                st.success(f"Conclusion: the mean of **{v}** {'**differs**' if p < .05 else 'does **not** significantly differ'} "
                           f"between {pick[0]} and {pick[1]} (p = {p:.4f}).")
                fig, ax = plt.subplots(figsize=(5.5, 3))
                ax.boxplot([a, b], tick_labels=[str(pick[0]), str(pick[1])])
                ax.set_title(f"{v} by {g}")
                show_fig(fig)

    elif test == "Paired (dependent) samples t-test":
        section("Paired (dependent) samples t-test", "ttest_paired")
        st.caption("Needs two numeric columns measured on the SAME rows — e.g. before vs. after a campaign.")
        v1 = st.selectbox("Measurement 1 (e.g. before)", ncols6, key="tt3_a")
        v2 = st.selectbox("Measurement 2 (e.g. after)", [c for c in ncols6 if c != v1], key="tt3_b")
        if st.button("▶ Run paired t-test", key="btn_tt3", type="primary"):
            both = df[[v1, v2]].dropna()
            t, p = sps.ttest_rel(both[v1], both[v2])
            diff = (both[v2] - both[v1]).mean()
            metric_row([(f"Mean — {v1}", f"{both[v1].mean():,.2f}"), (f"Mean — {v2}", f"{both[v2].mean():,.2f}"),
                        ("Mean change", f"{diff:+,.2f}"), ("p-value", f"{p:.4f}")])
            st.success(f"Conclusion: the average {'**changed**' if p < .05 else 'did **not** significantly change'} "
                       f"between the two measurements (p = {p:.4f}).")
            st.caption("Note: this test only makes sense if the two columns are truly paired (same customer in both).")

    else:
        section("Crosstab + Chi-square test of independence", "chisq")
        copts = cat_cols(df)
        if len(copts) >= 2:
            a = st.selectbox("Variable 1 (rows)", copts, key="chi_a")
            b = st.selectbox("Variable 2 (columns)", [c for c in copts if c != a], key="chi_b")
            if st.button("▶ Run chi-square test", key="btn_chi", type="primary"):
                ct = pd.crosstab(df[a], df[b])
                chi2, p, dof, exp = sps.chi2_contingency(ct)
                n = ct.values.sum()
                cramer = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Observed counts**"); st.dataframe(ct, width="stretch")
                with c2:
                    st.markdown("**Expected counts if unrelated**")
                    st.dataframe(pd.DataFrame(exp, index=ct.index, columns=ct.columns).round(1), width="stretch")
                metric_row([("Chi-square", f"{chi2:.2f}"), ("p-value", f"{p:.4f}"), ("Cramér's V", f"{cramer:.3f}")])
                strength = "negligible" if cramer < .1 else "weak" if cramer < .3 else "moderate" if cramer < .5 else "strong"
                st.success(f"Conclusion: **{a}** and **{b}** are {'**related**' if p < .05 else '**not** significantly related'} "
                           f"(p = {p:.4f}); association strength: **{strength}**.")
                if (exp < 5).any():
                    st.warning("Some expected counts are below 5 — the test is less reliable. "
                               "Consider combining rare categories first (Data & Wrangling tab).")
        else:
            st.warning("Need at least two categorical columns.")

# ================================================================ TAB 7: ANOVA
with tabs[6]:
    instructions("""
**What this tab does:** compares a numeric outcome across the groups of one or more categorical
factors. Start with **one-way** (one factor). **Two-way** adds a second factor and, optionally, their
**interaction** — read the ℹ️ Help to understand what an interaction means before adding one.
**N-way** is available, but each added factor makes interpretation harder — that's not a software
limitation, it's statistics. After a significant result, run a **post-hoc test** to find *which*
groups differ.
""")
    import statsmodels.formula.api as smfa
    import statsmodels.api as sma
    section("🔬 ANOVA — comparing means across groups", "anova")
    ncols7 = num_cols(df)
    dv7 = st.selectbox("Numeric outcome (dependent variable)", ncols7,
                       index=default_ix(ncols7, ["annual_spend", "monthly_spend", "sales_k"]), key="an_dv")
    fopts = [c for c in cat_cols(df) if 2 <= df[c].nunique() <= 12 and c != dv7]
    mode7 = st.radio("ANOVA type", ["One-way (1 factor)", "Two-way (2 factors)", "N-way (3+ factors — advanced)"],
                     horizontal=True, key="an_mode")
    nfac = 1 if mode7.startswith("One") else 2 if mode7.startswith("Two") else 3
    factors = st.multiselect(f"Categorical factor(s) — pick {'exactly ' + str(nfac) if nfac < 3 else '3 or more'}",
                             fopts, default=[c for c in ["region"] if c in fopts][:1] if nfac == 1 else [], key="an_factors")
    interact = False
    if nfac >= 2:
        interact = st.checkbox("Include interaction terms (does one factor's effect depend on the other?)",
                               key="an_inter")
        if nfac >= 3:
            st.warning("⚠️ N-way ANOVA with interactions multiplies fast: 3 factors → 3 main effects, "
                       "3 two-way interactions, and 1 three-way interaction. Higher-order interactions are "
                       "genuinely hard to interpret and need lots of data in every combination of groups. "
                       "For many factors, a regression with dummy variables is usually the wiser tool.")
    if st.button("▶ Run ANOVA", key="btn_anova", type="primary"):
        need = nfac if nfac < 3 else 3
        if len(factors) < need or (nfac < 3 and len(factors) != nfac):
            st.error(f"Pick {'exactly' if nfac < 3 else 'at least'} {need} factor(s).")
        else:
            dat = df[[dv7] + factors].dropna()
            for c in factors:
                dat[c] = dat[c].astype(str)
            joiner = " * " if interact else " + "
            formula = f"{dv7} ~ " + joiner.join(f"C({c})" for c in factors)
            model = smfa.ols(formula, data=dat).fit()
            aov = sma.stats.anova_lm(model, typ=2).round(4)
            aov["significant (p<0.05)"] = np.where(aov["PR(>F)"] < 0.05, "✅", "—")
            st.session_state.anova = (aov, dat, factors, dv7, formula)
    if "anova" in st.session_state and st.session_state.anova[3] == dv7:
        aov, dat, factors_u, dv_u, formula_u = st.session_state.anova
        st.code(f"Model: {formula_u}   (n = {len(dat):,})")
        st.dataframe(aov, width="stretch")
        st.caption("Each row is a factor (or interaction, shown as A:B). p < 0.05 → that factor's group "
                   "means are not all equal.")
        gm = dat.groupby(factors_u)[dv_u].agg(["mean", "count"]).round(2)
        st.markdown("**Group means**")
        st.dataframe(gm, width="stretch")
        if len(factors_u) == 1:
            fig, ax = plt.subplots(figsize=(6.5, 3))
            lvls = dat[factors_u[0]].unique()
            ax.boxplot([dat.loc[dat[factors_u[0]] == l, dv_u] for l in lvls], tick_labels=lvls)
            ax.set_title(f"{dv_u} by {factors_u[0]}"); plt.xticks(rotation=30)
            show_fig(fig)

        st.markdown("---")
        section("🔎 Post-hoc tests — WHICH groups differ?", "posthoc")
        ph_factor = st.selectbox("Run post-hoc on which factor?", factors_u, key="ph_factor")
        ph_method = st.radio("Procedure", ["Tukey HSD", "Scheffé", "Bonferroni"], horizontal=True, key="ph_method")
        if st.button("▶ Run post-hoc test", key="btn_posthoc"):
            sub = dat[[dv_u, ph_factor]].dropna()
            if ph_method == "Tukey HSD":
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                res = pairwise_tukeyhsd(sub[dv_u], sub[ph_factor])
                out = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
                st.dataframe(out, width="stretch", hide_index=True)
                st.caption("reject = True → that pair of groups genuinely differs (family error kept at 5%).")
            else:
                import scikit_posthocs as sp
                fn = sp.posthoc_scheffe if ph_method == "Scheffé" else \
                     (lambda **k: sp.posthoc_ttest(**k, p_adjust="bonferroni", equal_var=False))
                pm = fn(a=sub, val_col=dv_u, group_col=ph_factor).round(4)
                st.markdown(f"**{ph_method} — pairwise p-values** (p < 0.05 → that pair differs)")
                st.dataframe(pm.style.map(lambda v: "background-color:#e8f5e9" if isinstance(v, float) and v < .05 else ""),
                             width="stretch")
                if ph_method == "Scheffé":
                    st.caption("Scheffé is the most conservative procedure — if it flags a pair, you can be confident.")

# ================================================================ TAB 8: REGRESSION
with tabs[7]:
    instructions("""
**What this tab does:** OLS regression with the full statistical table, all five assumption checks,
curve and log options, and out-of-sample model comparison. Workflow: pick the DV and predictors →
Run → read the table (ℹ️ Help decodes every number) → check Diagnostics → optionally compare models
with cross-validation.
""")
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    section("📈 OLS Regression", "regression")
    ncs = num_cols(df)
    dv = st.selectbox("Dependent variable (numeric)", ncs,
                      index=default_ix(ncs, ["annual_spend", "sales_k", "monthly_spend"]), key="reg_dv")
    xnum = st.multiselect("Numeric predictors", [c for c in ncs if c != dv],
                          default=[c for c in ["age", "income_k", "web_visits_month",
                                               "social_spend_k", "search_spend_k", "podcast_spend_k"] if c in ncs and c != dv],
                          key="reg_xnum")
    xcat = st.multiselect("Categorical predictors", [c for c in cat_cols(df) if c != dv],
                          default=[c for c in ["region"] if c in df.columns], key="reg_xcat")
    o1, o2, o3 = st.columns(3)
    sq = o1.selectbox("Add squared term for…", ["(none)"] + xnum, key="reg_sq",
                      help="Tests a curved (rise-then-fall) relationship. See Help above.")
    logx = o2.selectbox("Log-transform predictor…", ["(none)"] + xnum, key="reg_logx")
    logy = o3.checkbox("Log-transform the DV", key="reg_logy")
    h1, h2 = st.columns(2)
    with h1.popover("ℹ️ Help: squared terms"):
        st.markdown(HELP["quadratic"])
    with h2.popover("ℹ️ Help: log transforms"):
        st.markdown(HELP["logs"])

    def build_formula():
        terms = []
        for c in xnum:
            terms.append(f"np.log({c})" if c == logx else c)
        if sq != "(none)":
            terms.append(f"I({sq}**2)")
        terms += [f"C({c})" for c in xcat]
        lhs = f"np.log({dv})" if logy else dv
        return f"{lhs} ~ " + " + ".join(terms) if terms else None

    if st.button("▶ Run regression", key="btn_reg", type="primary"):
        f = build_formula()
        if not f:
            st.error("Pick at least one predictor.")
        else:
            data = df[[dv] + xnum + xcat].dropna()
            if (logy and (data[dv] <= 0).any()) or (logx != "(none)" and (data[logx] <= 0).any()):
                st.error("Log transforms need strictly positive values — that column has zeros or negatives.")
            else:
                m = smf.ols(f, data=data).fit()
                st.session_state.reg_model = (m, f, data)
    if "reg_model" in st.session_state:
        m, f, data = st.session_state.reg_model
        st.code(f"Model: {f}   (n = {int(m.nobs):,})")
        metric_row([("R²", f"{m.rsquared:.3f}"), ("Adj. R²", f"{m.rsquared_adj:.3f}"),
                    ("F-statistic", f"{m.fvalue:,.1f}"), ("F p-value", f"{m.f_pvalue:.2e}")])
        tbl = m.summary2().tables[1].round(4)
        tbl["significant (p<0.05)"] = np.where(tbl["P>|t|"] < 0.05, "✅", "—")
        st.dataframe(tbl, width="stretch")
        if sq != "(none)":
            b1 = m.params.get(sq); b2 = m.params.get(f"I({sq} ** 2)")
            if b1 is not None and b2 is not None and b2 != 0:
                st.info(f"**Turning point:** the effect of `{sq}` peaks/bottoms at {(-b1/(2*b2)):,.1f} "
                        "(only meaningful if inside the data's range).")

        section("Diagnostics — the five OLS assumptions", "diagnostics")
        d1, d2, d3 = st.columns(3)
        with d1:
            fig, ax = plt.subplots(figsize=(4.4, 3))
            ax.scatter(m.fittedvalues, m.resid, s=8, alpha=.5, color="#6B3F1D")
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel("fitted values"); ax.set_ylabel("residuals"); ax.set_title("1&3. Residuals vs fitted")
            show_fig(fig)
        with d2:
            fig = sm.qqplot(m.resid, line="45", fit=True)
            fig.set_size_inches(4.4, 3); fig.axes[0].set_title("5. Q-Q plot of residuals")
            show_fig(fig)
        with d3:
            fig, ax = plt.subplots(figsize=(4.4, 3))
            ax.hist(m.resid, bins=30, color="#6B3F1D", edgecolor="white")
            ax.set_title("5. Histogram of residuals")
            show_fig(fig)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
        from scipy import stats as sps2
        rows = []
        if len(xnum) >= 2:
            Xv = sm.add_constant(data[xnum])
            for i, cname in enumerate(Xv.columns):
                if cname != "const":
                    rows.append(("2. VIF — " + cname, f"{variance_inflation_factor(Xv.values, i):.2f}", "OK if < 10"))
        bp = het_breuschpagan(m.resid, m.model.exog)
        rows.append(("3. Breusch-Pagan p-value", f"{bp[1]:.3f}", "p > 0.05 → constant variance OK"))
        rows.append(("4. Durbin-Watson", f"{durbin_watson(m.resid):.2f}", "≈ 2 → no autocorrelation"))
        jb = sps2.jarque_bera(m.resid)
        rows.append(("5. Jarque-Bera p-value", f"{jb.pvalue:.3f}", "p > 0.05 → Normal residuals OK"))
        st.dataframe(pd.DataFrame(rows, columns=["Check", "Value", "How to read it"]),
                     width="stretch", hide_index=True)

        section("Model comparison — does the squared term improve prediction?", "cv")
        if sq == "(none)":
            st.caption("Pick a squared term above and rerun to enable the comparison.")
        elif st.button("▶ Compare with vs without the squared term (holdout + 5-fold CV)", key="btn_cv"):
            from sklearn.model_selection import train_test_split, KFold
            from sklearn.metrics import mean_squared_error
            fB = f
            fA = fB.replace(f" + I({sq}**2)", "").replace(f"I({sq}**2) + ", "")
            tr, te = train_test_split(data, test_size=0.3, random_state=42)
            out = []
            for nm, ff in [("Without squared term", fA), ("With squared term", fB)]:
                mm = smf.ols(ff, data=tr).fit()
                hold = np.sqrt(mean_squared_error(te[dv] if not logy else np.log(te[dv]), mm.predict(te)))
                kf = KFold(5, shuffle=True, random_state=42); r = []
                for a, b in kf.split(data):
                    mk = smf.ols(ff, data=data.iloc[a]).fit()
                    yt = data.iloc[b][dv] if not logy else np.log(data.iloc[b][dv])
                    r.append(np.sqrt(mean_squared_error(yt, mk.predict(data.iloc[b]))))
                out.append((nm, f"{hold:.2f}", f"{np.mean(r):.2f}"))
            st.dataframe(pd.DataFrame(out, columns=["Model", "Holdout RMSE (70/30)", "5-fold CV RMSE"]),
                         width="stretch", hide_index=True)

# ================================================================ TAB 9: LOGISTIC
with tabs[8]:
    instructions("""
**What this tab does:** logistic regression for a yes/no outcome, plus two things a plain output can't
give you: a live **probability calculator** (feel the S-curve with sliders) and a **cutoff slider**
showing the accuracy / sensitivity / specificity trade-off that makes targeting a business decision.
""")
    import statsmodels.formula.api as smf2
    section("🎯 Logistic Regression", "logistic")
    bin_opts = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and set(df[c].dropna().unique()) <= {0, 1} and df[c].nunique() == 2]
    if not bin_opts:
        st.warning("No binary (0/1) column found in this dataset. Try the customers or upgrades datasets.")
    else:
        tgt = st.selectbox("Binary target (0/1)", bin_opts,
                           index=default_ix(bin_opts, ["churned", "upgraded_premium"]), key="log_tgt")
        preds = st.multiselect("Predictors (numeric)", [c for c in num_cols(df) if c not in bin_opts + ["customer_id"]],
                               default=[c for c in ["satisfaction", "annual_spend", "loyalty_years",
                                                    "months_subscribed", "avg_monthly_spend", "support_tickets"]
                                        if c in df.columns], key="log_preds")
        flag_preds = st.multiselect("Predictors that are 0/1 flags (optional)",
                                    [c for c in bin_opts if c != tgt],
                                    default=[c for c in ["app_user"] if c in bin_opts and c != tgt], key="log_flags")
        allp = preds + flag_preds
        if st.button("▶ Run logistic regression", key="btn_logit", type="primary"):
            if not allp:
                st.error("Pick at least one predictor.")
            else:
                data = df[[tgt] + allp].dropna()
                ml = smf2.logit(f"{tgt} ~ " + " + ".join(allp), data=data).fit(disp=0)
                st.session_state.logit_model = (ml, data, tgt, allp)
        if "logit_model" in st.session_state and st.session_state.logit_model[2] == tgt:
            ml, data, tgt, allp = st.session_state.logit_model
            metric_row([("Base rate of 1s", f"{data[tgt].mean():.1%}"),
                        ("Pseudo R²", f"{ml.prsquared:.3f}"), ("n", f"{int(ml.nobs):,}")])
            tbl = ml.summary2().tables[1].round(4)
            tbl["Odds ratio"] = np.exp(ml.params).round(3)
            tbl["significant"] = np.where(tbl["P>|z|"] < 0.05, "✅", "—")
            st.dataframe(tbl, width="stretch")

            st.markdown("#### 🎛️ Probability calculator")
            vals = {}
            cols = st.columns(min(4, len(allp)))
            for i, c in enumerate(allp):
                with cols[i % len(cols)]:
                    if c in flag_preds:
                        vals[c] = 1.0 if st.toggle(f"{c} = 1", value=True, key=f"pc_{c}") else 0.0
                    else:
                        lo_, hi_ = float(data[c].min()), float(data[c].max())
                        vals[c] = st.slider(c, lo_, hi_, float(data[c].median()), key=f"pc_{c}")
            prob = float(ml.predict(pd.DataFrame([vals]))[0])
            st.metric("Predicted probability for this customer", f"{prob:.1%}")

            section("✂️ Cutoff analysis (70/30 stratified holdout, random_state=42)", "cutoff")
            cut = st.slider("Classification cutoff", 0.05, 0.9, 0.5, 0.05, key="log_cut")
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(data, test_size=0.3, random_state=42, stratify=data[tgt])
            m2 = smf2.logit(f"{tgt} ~ " + " + ".join(allp), data=tr).fit(disp=0)
            pr = (m2.predict(te) >= cut).astype(int)
            cm, acc, sens, spec = confusion_report(te[tgt], pr)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(cm, width="stretch")
            with c2:
                metric_row([("Accuracy", f"{acc:.3f}"), ("Sensitivity", f"{sens:.3f}"), ("Specificity", f"{spec:.3f}")])

# ================================================================ TAB 10: DATA MINING
with tabs[9]:
    instructions("""
**What this tab does:** four machine-learning classifiers — KNN, Naive Bayes, a classification tree,
and a random forest — trained on the same target with the same 70/30 split (random_state=42) so you can
compare them fairly. Each model has its own ℹ️ Help. Compare **sensitivity**, not just accuracy —
accuracy flatters any model when one class is rare.
""")
    section("🤖 Supervised Data Mining — four classifiers, one fair fight", None)
    hh = st.columns(4)
    for col, key, label in zip(hh, ["knn", "nb", "tree", "rf"],
                               ["ℹ️ KNN", "ℹ️ Naive Bayes", "ℹ️ Trees", "ℹ️ Random forest"]):
        with col.popover(label):
            st.markdown(HELP[key])
    bin_opts6 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                 and set(df[c].dropna().unique()) <= {0, 1} and df[c].nunique() == 2]
    if not bin_opts6:
        st.warning("No binary (0/1) target found. Try the BrewBox customers dataset.")
    else:
        tgt6 = st.selectbox("Binary target", bin_opts6, index=default_ix(bin_opts6, ["churned"]), key="dm_tgt")
        feat_opts = [c for c in num_cols(df) if c != tgt6 and c != "customer_id"]
        feats = st.multiselect("Features", feat_opts,
                               default=[c for c in feat_opts if c not in bin_opts6][:8], key="dm_feats")
        kval = st.slider("k for KNN", 3, 25, 9, 2, key="dm_k")
        depth = st.slider("Max depth for the tree", 2, 8, 4, key="dm_depth")
        if st.button("▶ Train all four models", key="btn_dm", type="primary"):
            if len(feats) < 2:
                st.error("Pick at least two features.")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                d6 = df[[tgt6] + feats].dropna()
                Xtr, Xte, ytr, yte = train_test_split(d6[feats], d6[tgt6], test_size=0.3,
                                                      random_state=42, stratify=d6[tgt6])
                sc = StandardScaler().fit(Xtr)
                results, models = [], {}
                for nm, mod, scaled in [
                        (f"KNN (k={kval}, standardized)", KNeighborsClassifier(kval), True),
                        ("Gaussian Naive Bayes", GaussianNB(), False),
                        (f"Classification tree (depth {depth})", DecisionTreeClassifier(max_depth=depth, random_state=1), False),
                        ("Random forest (200 trees)", RandomForestClassifier(n_estimators=200, random_state=1), False)]:
                    a, b = (sc.transform(Xtr), sc.transform(Xte)) if scaled else (Xtr, Xte)
                    mod.fit(a, ytr)
                    pr = mod.predict(b)
                    _, acc, sens, spec = confusion_report(yte, pr)
                    results.append((nm, f"{acc:.3f}", f"{sens:.3f}", f"{spec:.3f}"))
                    models[nm] = mod
                st.session_state.dm = (results, models, feats, depth, kval)
        if "dm" in st.session_state:
            results, models, feats_used, depth_u, k_u = st.session_state.dm
            st.dataframe(pd.DataFrame(results, columns=["Model", "Accuracy", "Sensitivity", "Specificity"]),
                         width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**The classification tree's rules**")
                from sklearn.tree import plot_tree
                tree_model = [m for n, m in models.items() if "tree" in n][0]
                fig, ax = plt.subplots(figsize=(9, 4.5))
                plot_tree(tree_model, feature_names=feats_used, class_names=["0", "1"],
                          filled=True, fontsize=7, ax=ax, impurity=False)
                show_fig(fig)
            with c2:
                st.markdown("**Random-forest feature importance**")
                rf_model = [m for n, m in models.items() if "forest" in n][0]
                imp = pd.Series(rf_model.feature_importances_, index=feats_used).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.6))
                imp.plot(kind="barh", ax=ax, color="#6B3F1D")
                ax.set_title("What drives the prediction?")
                show_fig(fig)

# ================================================================ TAB 11: CLUSTERING
with tabs[10]:
    instructions("""
**What this tab does:** market segmentation by algorithm. Choose your method — **K-means**,
**Hierarchical**, or **run both and compare** (they often agree on clean data; comparing them is itself
a robustness check). Use the elbow plot and dendrogram to choose the number of segments, then read the
profile table and *name* your segments — the algorithm finds groups, the marketer gives them meaning.
""")
    section("🧩 Clustering — find your customer segments", "clustering")
    ncs7 = [c for c in num_cols(df) if c != "customer_id"]
    cvars = st.multiselect("Variables to cluster on (2–4, standardized automatically)", ncs7,
                           default=[c for c in ["monthly_spend", "orders_per_year", "app_sessions_month",
                                                "income_k", "web_visits_month", "annual_spend"] if c in ncs7][:3],
                           key="cl_vars")
    method_c = st.radio("Method", ["K-means", "Hierarchical (Ward)", "Both — compare them"],
                        horizontal=True, key="cl_method")
    if len(cvars) >= 2:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        d7 = df[cvars].dropna()
        Z = StandardScaler().fit_transform(d7)
        e1, e2 = st.columns([3, 1])
        with e2.popover("ℹ️ Help: elbow & silhouette"):
            st.markdown(HELP["elbow"])
        with e1:
            with st.spinner("Computing elbow plot…"):
                inert = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).inertia_ for k in range(1, 8)]
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.plot(range(1, 8), inert, marker="o", color="#6B3F1D")
            ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("inertia"); ax.set_title("Elbow plot")
            show_fig(fig)
        kpick = st.slider("Number of clusters (k)", 2, 7, 3, key="cl_k")
        if st.button("▶ Run clustering", key="btn_cluster", type="primary"):
            out_models = {}
            if method_c in ("K-means", "Both — compare them"):
                km = KMeans(n_clusters=kpick, n_init=10, random_state=42).fit(Z)
                out_models["K-means"] = km.labels_
            if method_c in ("Hierarchical (Ward)", "Both — compare them"):
                hc = AgglomerativeClustering(n_clusters=kpick, linkage="ward").fit(Z)
                out_models["Hierarchical (Ward)"] = hc.labels_
            for nm, labels in out_models.items():
                sil = silhouette_score(Z, labels)
                st.markdown(f"### {nm}")
                metric_row([("k", kpick), ("Silhouette score", f"{sil:.3f}"),
                            ("Cluster sizes", " / ".join(str(s) for s in np.bincount(labels)))])
                prof = d7.groupby(labels).mean().round(1)
                prof.index.name = "cluster"
                st.markdown("**Cluster profiles (original units) — name these segments!**")
                st.dataframe(prof, width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                labels0 = list(out_models.values())[0]
                fig, ax = plt.subplots(figsize=(5.5, 4))
                for lab in np.unique(labels0):
                    sub = d7[labels0 == lab]
                    ax.scatter(sub[cvars[0]], sub[cvars[1]], s=8, alpha=.5, label=f"cluster {lab}")
                ax.set_xlabel(cvars[0]); ax.set_ylabel(cvars[1]); ax.legend(fontsize=8)
                ax.set_title(f"Clusters in 2-D ({list(out_models)[0]})")
                show_fig(fig)
            with c2:
                dd1, dd2 = st.columns([3, 1])
                with dd2.popover("ℹ️ Help: dendrogram"):
                    st.markdown(HELP["dendrogram"])
                from scipy.cluster.hierarchy import linkage, dendrogram
                samp = d7.sample(min(len(d7), 1200), random_state=42)
                Zs = StandardScaler().fit_transform(samp)
                L = linkage(Zs, method="ward")
                fig, ax = plt.subplots(figsize=(5.5, 4))
                dendrogram(L, truncate_mode="lastp", p=30, ax=ax, no_labels=True, color_threshold=0)
                ax.set_title("Hierarchical dendrogram (Ward, sample ≤1,200)")
                show_fig(fig)
    else:
        st.info("Pick at least two numeric variables.")

# ================================================================ TAB 12: TEXT & NLP
with tabs[11]:
    instructions("""
**What this tab does:** turns thousands of reviews into marketing insight: cleaned word frequencies
and a word cloud, sentiment scoring, **aspect-based sentiment** (edit the keywords for your product —
this is the money slide), and topic modeling. Load a reviews dataset from the sidebar or upload your own
CSV with a text column.
""")
    tcols = text_cols(df)
    if not tcols:
        st.warning("No long-text column found. Load a reviews dataset from the sidebar.")
    else:
        tc = st.selectbox("Text column", tcols, key="nlp_col")
        texts = df[tc].dropna().astype(str)
        import re as _re
        from collections import Counter
        STOP = set("""the a an and my is was for of to in from at this that with such very i me it be so
                      you your we our they them he she his her its are were been being do does did have has
                      had will would can could should may might must not no nor but or if then than too""".split())
        c1, c2 = st.columns(2)
        with c1:
            section("Most frequent words (cleaned & stemmed)", "textmining")
            try:
                from nltk.stem import PorterStemmer
                ps = PorterStemmer(); stem = ps.stem
            except Exception:
                stem = lambda w: w
            words = Counter()
            for t in texts:
                for w in _re.findall(r"[a-z]+", t.lower()):
                    if w not in STOP and len(w) > 2:
                        words[stem(w)] += 1
            st.dataframe(pd.DataFrame(words.most_common(15), columns=["word (stemmed)", "count"]),
                         width="stretch", hide_index=True)
        with c2:
            st.markdown("**Word cloud**")
            try:
                from wordcloud import WordCloud
                wc = WordCloud(width=560, height=340, background_color="white",
                               colormap="copper").generate_from_frequencies(words)
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.imshow(wc); ax.axis("off")
                show_fig(fig)
            except Exception as e:
                st.caption(f"(word cloud unavailable: {e})")

        st.markdown("---")
        s1, s2 = st.columns(2)
        with s1:
            section("Sentiment (TextBlob polarity)", "sentiment")
            if st.button("▶ Score sentiment", key="btn_sent", type="primary"):
                from textblob import TextBlob
                st.session_state.nlp_pol = texts.apply(lambda t: TextBlob(t).sentiment.polarity)
            if "nlp_pol" in st.session_state and len(st.session_state.nlp_pol) == len(texts):
                pol = st.session_state.nlp_pol
                metric_row([("Positive (> 0.05)", f"{(pol > .05).mean():.1%}"),
                            ("Negative (< −0.05)", f"{(pol < -.05).mean():.1%}"),
                            ("Mean polarity", f"{pol.mean():.3f}")])
                fig, ax = plt.subplots(figsize=(6, 2.6))
                ax.hist(pol, bins=30, color="#6B3F1D", edgecolor="white")
                ax.set_title("Distribution of sentiment"); ax.set_xlabel("polarity")
                show_fig(fig)
                stars = [c for c in num_cols(df) if "star" in c.lower() or "rating" in c.lower()]
                if stars:
                    sr = df.loc[texts.index, stars[0]]
                    st.metric(f"Correlation of polarity with {stars[0]}", f"{pol.corr(sr):.3f}")
        with s2:
            section("Aspect-based sentiment", "aspects")
            default_aspects = "flavor: flavor, taste\nshipping: shipping, delivery, arrived, delayed\nprice: price, expensive, overpriced\npackaging: packaging, bags, box"
            asp_text = st.text_area("aspect: keyword, keyword, …  (one aspect per line — edit for YOUR product)",
                                    value=default_aspects, height=120, key="nlp_aspects")
            if st.button("▶ Score aspects", key="btn_asp"):
                if "nlp_pol" not in st.session_state or len(st.session_state.nlp_pol) != len(texts):
                    from textblob import TextBlob
                    st.session_state.nlp_pol = texts.apply(lambda t: TextBlob(t).sentiment.polarity)
                pol = st.session_state.nlp_pol
                rows = []
                low = texts.str.lower()
                for line in asp_text.splitlines():
                    if ":" not in line:
                        continue
                    name, kws = line.split(":", 1)
                    kws = [k.strip() for k in kws.split(",") if k.strip()]
                    mask = low.apply(lambda t: any(k in t for k in kws))
                    if mask.any():
                        rows.append((name.strip(), int(mask.sum()), round(float(pol[mask].mean()), 3)))
                if rows:
                    asp_df = pd.DataFrame(rows, columns=["Aspect", "Reviews mentioning it", "Mean polarity"]) \
                                .sort_values("Mean polarity")
                    st.dataframe(asp_df, width="stretch", hide_index=True)
                    fig, ax = plt.subplots(figsize=(6, 2.6))
                    colors = ["#B3261E" if v < 0 else "#2E7D32" for v in asp_df["Mean polarity"]]
                    ax.barh(asp_df["Aspect"], asp_df["Mean polarity"], color=colors)
                    ax.axvline(0, color="k", lw=1); ax.set_title("Sentiment by aspect")
                    show_fig(fig)

        st.markdown("---")
        section("Topic modeling (LDA)", "lda")
        ntop = st.slider("Number of topics", 2, 6, 3, key="nlp_k")
        if st.button("▶ Find topics", key="btn_lda"):
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            vec = CountVectorizer(stop_words="english", min_df=5)
            Xt = vec.fit_transform(texts)
            lda = LatentDirichletAllocation(n_components=ntop, random_state=42).fit(Xt)
            vocab = vec.get_feature_names_out()
            rows = [(f"Topic {i+1}", ", ".join(vocab[j] for j in comp.argsort()[-8:][::-1]))
                    for i, comp in enumerate(lda.components_)]
            st.dataframe(pd.DataFrame(rows, columns=["Topic", "Top words — YOU name the theme"]),
                         width="stretch", hide_index=True)

# ================================================================ TAB 13: IMAGE ANALYSIS
with tabs[12]:
    instructions("""
**What this tab does:** batch analysis of marketing images — upload a set of social posts, product
shots, or customer photos (PNG/JPG). For each image you get its **dominant color palette** (found with
k-means on the pixels — the same clustering algorithm from the Clustering tab, reused on colors!),
**brightness**, and **colorfulness**. The summary row tells you whether your feed is visually consistent.
""")
    section("🖼️ Simple image analysis for marketers", "image")
    imgs = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"],
                            accept_multiple_files=True, key="img_up")
    n_colors = st.slider("Dominant colors per image", 3, 8, 5, key="img_k")
    if imgs and st.button("▶ Analyze images", key="btn_img", type="primary"):
        from PIL import Image
        from sklearn.cluster import KMeans
        summary = []
        for up in imgs[:12]:
            try:
                im = Image.open(up).convert("RGB")
                im_small = im.resize((120, 120))
                arr = np.asarray(im_small).reshape(-1, 3).astype(float)
                gray = arr.mean(axis=1)
                brightness = gray.mean()
                rg = arr[:, 0] - arr[:, 1]
                yb = 0.5 * (arr[:, 0] + arr[:, 1]) - arr[:, 2]
                colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
                km = KMeans(n_clusters=n_colors, n_init=4, random_state=42).fit(arr)
                centers = km.cluster_centers_.astype(int)
                shares = np.bincount(km.labels_) / len(km.labels_)
                order = np.argsort(-shares)
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.image(im, caption=up.name, width="stretch")
                with c2:
                    fig, ax = plt.subplots(figsize=(5, 1.1))
                    left = 0
                    for j in order:
                        ax.barh(0, shares[j], left=left, color=centers[j] / 255)
                        left += shares[j]
                    ax.set_xlim(0, 1); ax.axis("off"); ax.set_title("Dominant colors (share of pixels)", fontsize=9)
                    show_fig(fig)
                    st.caption(" · ".join("#%02x%02x%02x (%.0f%%)" % (*centers[j], shares[j] * 100) for j in order[:3]))
                with c3:
                    st.metric("Brightness (0–255)", f"{brightness:.0f}")
                    st.metric("Colorfulness", f"{colorfulness:.0f}")
                summary.append((up.name, brightness, colorfulness))
                st.markdown("---")
            except Exception as e:
                st.error(f"Couldn't read {up.name}: {e}")
        if len(summary) >= 2:
            sdf = pd.DataFrame(summary, columns=["image", "brightness", "colorfulness"])
            st.markdown("#### 📋 Batch consistency summary")
            metric_row([("Images", len(sdf)),
                        ("Brightness mean ± sd", f"{sdf.brightness.mean():.0f} ± {sdf.brightness.std():.0f}"),
                        ("Colorfulness mean ± sd", f"{sdf.colorfulness.mean():.0f} ± {sdf.colorfulness.std():.0f}")])
            st.caption("A large standard deviation relative to the mean = a visually inconsistent feed. "
                       "Compare the dominant palettes above against your brand colors.")
    elif not imgs:
        st.info("Upload images above, then click Analyze. Tip: try a batch of your course's or a brand's "
                "recent social posts.")

st.markdown("---")
st.caption(BRAND_NOTICE + " · Fixed seeds (random_state=42) wherever randomness is involved, so results are reproducible.")
