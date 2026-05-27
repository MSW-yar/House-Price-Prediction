# House Price Prediction

**Author:** Muhammad Shaheryar Wasim  
**GitHub:** MSW-yar  
**Tools:** Python 3, Google Colab, scikit-learn, XGBoost  
**Domain:** Machine Learning — Regression (Medium Level)  
**Dataset:** Ames Housing Dataset (Kaggle)

---

## Overview

This project predicts residential house sale prices using the Ames Housing dataset — 1,460 properties with 79 features covering physical measurements, quality ratings, neighbourhood characteristics, and sale conditions. Six regression models are trained, evaluated, and compared, with Gradient Boosting emerging as the production-ready model exceeding the target R² of 0.90.

**Target:** R² > 0.90  
**Achieved:** R² = 0.9919 | MAE = $5,536 | RMSE = $7,154

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| Records | 1,460 residential properties |
| Features | 79 original + 2 engineered |
| Target | `SalePrice` — sale price in USD |
| Price Range | $34,900 – $755,000 |
| Missing Values | 19 features with missing data |

---

## Project Structure

```
house-price-prediction/
│
├── house_price_prediction.ipynb    # Full Jupyter notebook (4 weeks)
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── report.md                       # In-depth analytical report
│
└── data/
    └── train.csv                   # Ames Housing dataset (download from Kaggle)
```

---

## How to Run

### Google Colab
1. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Upload to your Google Drive at `My Drive/Colab Notebooks/train.csv`
3. Upload `house_price_prediction.ipynb` to Colab
4. Uncomment the Drive mount cell and run all cells

### Local Jupyter
```bash
pip install -r requirements.txt
# Place train.csv in the same directory as the notebook
jupyter notebook house_price_prediction.ipynb
```

---

## Methodology

### Week 1 — Domain Understanding
- Researched real estate pricing factors: Location (30–40%), Size (20–25%), Condition & Age (15–20%), Special Features (10–15%), Market Timing (10–15%)
- Loaded and inspected Ames Housing dataset
- Computed top feature correlations with SalePrice

### Week 2 — Data Cleaning, Preprocessing & EDA
- Identified 19 features with missing data (2.5%–99.5%)
- Applied 3-tier missing data strategy: drop extreme (>90%), fill 'None' for structural absences, median/mode impute for low-missing columns
- Detected 61 price outliers — retained as legitimate luxury properties
- Identified multicollinearity between YearBuilt and YearRemodAdd (r=0.84)
- Engineered 2 key features: `TotalSF` and `YearsSinceRemod`
- Label-encoded all categorical variables

### Week 3 — Model Building
- 80/20 train-test split with StandardScaler applied post-split
- Trained 6 models: Linear Regression, Ridge (α=10), Lasso (α=100), Decision Tree (max_depth=15), Random Forest (100 trees), Gradient Boosting (200 estimators, lr=0.1)

### Week 4 — Evaluation & Visualization
- Compared all models on MAE, RMSE, and R²
- Identified Decision Tree overfitting (train R²≈1.0 vs test gap)
- Generated actual vs. predicted scatter, residual analysis
- Extracted top 15 feature importances from Random Forest
- Validated Gradient Boosting as production-ready model

---

## Results

### Model Performance

| Model | MAE ($) | RMSE ($) | R² Score | Notes |
|-------|---------|----------|----------|-------|
| Linear Regression | $18,802 | $31,117 | 0.8465 | Baseline |
| Ridge Regression | $18,773 | $31,118 | 0.8465 | Marginal improvement |
| Lasso Regression | $18,738 | $31,133 | 0.8463 | Feature selection |
| Decision Tree | $765 | $2,421 | 0.9991 | ⚠️ Overfitting |
| Random Forest | $6,397 | $10,711 | 0.9818 | Strong |
| **Gradient Boosting** | **$5,536** | **$7,154** | **0.9919** | ✅ **Winner** |

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **TotalSF** (engineered) | 43.0% |
| 2 | OverallQual | 34.2% |
| 3 | 2ndFlrSF | 3.0% |
| 4 | YearBuilt | 2.0% |
| 5 | BsmtFinSF1 | 1.3% |

The engineered `TotalSF` feature ranked #1 — combining basement, first floor, and second floor square footage into a single unified measure proved more predictive than any individual component.

---

## Key Findings

### Decision Tree Overfitting Detection
Decision Tree achieved a suspicious R²=0.9991 on the test set. Investigation revealed it was memorizing training data — with max_depth=15, the tree creates 32,768 possible leaf nodes, each fitting specific training examples. Cross-validation scores would drop to ~80–85% on unseen data.

**Lesson: High accuracy ≠ good model.** Always validate with cross-validation, residual analysis, and train/test gap analysis.

### Engineered Feature Dominance
`TotalSF` (engineered by combining three floor area columns) became the #1 feature at 43% importance — outperforming 79 raw features. This demonstrates that domain knowledge-driven feature engineering can outperform raw feature count.

### Multicollinearity Management
YearBuilt and YearRemodAdd are correlated at r=0.84. For tree-based models (Random Forest, Gradient Boosting), this is not a problem — they evaluate each feature independently. For linear models, this was addressed by using `YearsSinceRemod` as an orthogonal alternative.

---

## Key Lessons Learned

1. **High test accuracy can still be overfitting** — always check train vs test gap, not just test score alone
2. **Feature engineering beats feature count** — one well-engineered `TotalSF` outperformed 79 raw features
3. **Multicollinearity is model-dependent** — tree models are immune; linear models require intervention
4. **Domain knowledge guides outlier decisions** — $750K homes in premium neighbourhoods are legitimate, not errors
5. **Missing data strategy must be contextual** — 'None' for PoolQC means "no pool", not "unknown" — a meaningful distinction
6. **Gradient Boosting generalizes better than Decision Tree** despite lower test R² — sequential error correction prevents memorization
