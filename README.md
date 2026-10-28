# Machine Learning Practice — IIT Madras BS (Diploma in Data Science)

Coursework from the **Machine Learning Practice (MLP)** course at IIT Madras: three graded Kaggle assignments and one end-to-end capstone project (assessed through two 1-hour vivas).

| # | Project | Task | Type | Best model | Result |
|---|---------|------|------|-----------|--------|
| 1 | [Taxi fare prediction](assignments/01-taxi-fare-regression) | Predict `total_amount` of NYC taxi rides | Regression | Random Forest | Local R² 0.957 · Kaggle score: _add_ |
| 2 | [Mushroom classification](assignments/02-mushroom-classification) | Edible vs poisonous | Binary classification | Logistic Regression | Val accuracy 1.00 · Kaggle score: _add_ |
| 3 | [Movie review sentiment](assignments/03-movie-sentiment) | Negative / neutral / positive | Multi-class NLP | Calibrated LinearSVC (TF-IDF + VADER) | Val accuracy 0.679 · Kaggle score: _add_ |
| ★ | [**Heavy equipment price prediction**](project-heavy-equipment-price-prediction) | Predict auction sale price | Regression (RMSLE) | CatBoost | **Leaderboard RMSLE 0.198** |

★ = capstone project. Start there.

## Repo layout

```
assignments/
  01-taxi-fare-regression/        notebook, problem statement, plots
  02-mushroom-classification/
  03-movie-sentiment/
project-heavy-equipment-price-prediction/
  README.md                       full write-up
  heavy_equipment_price_prediction.ipynb
  data/README.md                  how to get the data
  assets/                         figures used in the write-up
requirements.txt
```

## Data

Datasets are **not** included: they belong to course-run Kaggle competitions, and the capstone training file is 42 MB. Each folder's README links to the competition page; download the files there and place them as described in `project-heavy-equipment-price-prediction/data/README.md`.

## Running the notebooks

The notebooks were written and run on Kaggle, so the data paths look like `/kaggle/input/competitions/...`. To run locally, download the data and edit the path at the top of each notebook.

```bash
pip install -r requirements.txt
jupyter lab
```

## Skills demonstrated

Data cleaning and outlier handling · structural-missingness analysis · feature engineering · target encoding without leakage (out-of-fold) · tree ensembles and gradient boosting (RF, XGBoost, LightGBM, CatBoost) · TF-IDF text pipelines · hyperparameter search · metric-aligned training (log-target for RMSLE).

## Author

Yash Jadhav · [GitHub](https://github.com/) · [LinkedIn](https://www.linkedin.com/)
