# Customer Lifetime Value Prediction

**Team:** The Starks

| Member | Student ID | GitHub |
|--------|------------|--------|
| Othmane Zizi | 261255341 | `othmane-zizi-pro` |
| Fares Joni | 261254593 | `FaresJ81` |
| Tanmay Giri | 261272443 | `tanmaysgiri` |

## Project Overview

This project predicts **Customer Lifetime Value (CLV)** using machine learning techniques applied to the Online Retail II dataset. CLV represents the total revenue a business can expect from a single customer throughout their relationship.

### Hypothesis

> "Historical purchase patterns (recency, frequency, and monetary value) can accurately predict a customer's future lifetime value, enabling businesses to optimize marketing spend and retention strategies."

## Dataset

**Source:** [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

- **Records:** ~541,909 transactions
- **Time Period:** December 2009 - December 2011
- **Domain:** UK-based online giftware retailer

## Project Structure

```
ent_assignment/
├── README.md                           # Project overview
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/                            # Original dataset
│   │   └── online_retail_II.xlsx
│   └── processed/                      # Cleaned and feature-engineered data
│       ├── cleaned_retail.csv
│       ├── customer_features.csv
│       ├── clv_predictions.csv
│       └── customer_segments.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Initial EDA
│   ├── 02_data_cleaning.ipynb          # Data preprocessing
│   ├── 03_feature_engineering.ipynb    # RFM + behavioral features
│   ├── 04_modeling.ipynb               # Model training & evaluation
│   └── 05_segmentation.ipynb           # Customer segmentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading utilities
│   ├── features.py                     # Feature engineering functions
│   └── models.py                       # Model training utilities
└── reports/
    ├── figures/                        # Visualizations
    ├── model_results.csv
    ├── cv_results.csv
    └── feature_importance.csv
```

## Methodology

### 1. Data Preparation
- Remove transactions with missing Customer ID (~20%)
- Filter out cancelled transactions
- Remove negative quantities and prices
- Calculate transaction amounts

### 2. Feature Engineering

**RFM Features:**
- `Recency`: Days since last purchase
- `Frequency`: Total number of transactions
- `Monetary`: Total customer spend

**Behavioral Features:**
- `Tenure`: Days since first purchase
- `AvgTimeBetweenPurchases`: Purchase cadence
- `NumUniqueProducts`: Product diversity
- `AvgBasketSize`: Items per transaction
- `AvgOrderValue`: Spend per transaction

**Target Variable:**
- `CLV`: Total revenue in next 6 months

### 3. Modeling

Models compared:
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

Evaluation metrics: RMSE, MAE, R-squared

### 4. Customer Segmentation

Segmentation methods:
- Quantile-based CLV tiers
- RFM scoring
- K-Means clustering

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/FaresJ81/Nerds-With-Attitude.git
cd Nerds-With-Attitude

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

Execute notebooks in order:
1. `01_data_exploration.ipynb` - Explore raw data
2. `02_data_cleaning.ipynb` - Clean and preprocess
3. `03_feature_engineering.ipynb` - Create features
4. `04_modeling.ipynb` - Train and evaluate models
5. `05_segmentation.ipynb` - Segment customers

## Key Results

### Model Performance
- Best model identified through cross-validation
- Feature importance analysis shows Monetary and Frequency as top predictors

### Business Impact
- Model identifies high-value customers with significant lift over random selection
- Enables targeted marketing strategies for each customer segment

### Segment Recommendations
| Segment | Strategy | Key Actions |
|---------|----------|-------------|
| High Value | Retention | VIP treatment, exclusive offers, loyalty rewards |
| Medium-High | Growth | Cross-selling, volume discounts, referrals |
| Medium-Low | Engagement | Email campaigns, bundle deals, promotions |
| Low Value | Activation | Win-back campaigns, entry offers, free shipping |

## References

- [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- [CLV Prediction Research (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2405844023005911)
- [CLV in B2B SaaS (Springer)](https://link.springer.com/article/10.1057/s41270-023-00234-6)

## License

This project is for educational purposes as part of McGill University coursework.
