# Assignment 3 — Movie Review Sentiment (3-class)

Kaggle course assignment: classify review text as negative (0), neutral (1) or positive (2). Metric: accuracy.
**Kaggle score:** _add_ · **Cutoff:** _add_ · **Competition:** _add link_

## Data
7,000 training phrases, 1,700 test phrases, plus three anonymous numeric features. See [`problem_statement.txt`](problem_statement.txt).

## Approach
- De-duplicated training phrases (7,000 → 6,823).
- **Text features:** TF-IDF, 1–2-grams, sublinear TF, `min_df=2`.
- **Hand-built features:** VADER polarity scores, word count, `!`/`?` counts, ALL-CAPS words, contrast words (*but, however…*), positive/negative keyword flags.
- Dense SVD (100 components) variant for KNN and tree models; MinMax-scaled variant for Complement Naive Bayes, which rejects negatives.
- Compared 7 models, then tuned the top 3.

## Results (15 % stratified validation)

| Model | Val accuracy |
|-------|-------------|
| **Calibrated LinearSVC (C = 0.6)** | **0.679** |
| Complement NB | 0.657 |
| Logistic Regression | 0.634 |
| Random Forest | 0.610 |
| KNN (k = 7) | 0.578 |
| Decision Tree | 0.482 |
| SGD (modified Huber) | 0.230 |

Tuning didn't help LinearSVC (C = 5 scored 0.671 on validation vs 0.679 untuned), so the untuned C = 0.6 model was retrained on all data and submitted.

## Limitation
The model predicts "neutral" for only 2.2 % of test rows. Neutral is the hardest class and sits between two stronger ones. Class weighting or ordinal modelling would be the next thing to try.

## Notebook
[`movie_sentiment.ipynb`](movie_sentiment.ipynb) (ydata-profiling HTML output removed to keep the file small).
