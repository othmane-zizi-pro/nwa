# Customer Lifetime Value Prediction
## 5-Minute Progress Presentation

---

## Slide 1: Team (30 sec)

**Group Name:** The Starks

| Member | Student ID | GitHub |
|--------|------------|--------|
| Othmane Zizi | 261255341 | @othmane-zizi-pro |
| Fares Joni | 261254593 | @FaresJ81 |
| Tanmay Giri | 261272443 | @tanmaysgiri |

**Repository:** github.com/FaresJ81/Nerds-With-Attitude

---

## Slide 2: Context (1.5 min)

**What is Customer Lifetime Value (CLV)?**
- Total revenue a business expects from a customer over their entire relationship

**Why does it matter?**
- Acquiring new customers costs **5-25x more** than retaining existing ones
- Companies using CLV see **5-8% improvement** in acquisition efficiency
- Enables smarter allocation of marketing budgets

**Business Applications:**
- Identify high-value customers for VIP treatment
- Target at-risk customers with retention campaigns
- Optimize marketing spend by focusing on profitable segments

---

## Slide 3: Hypothesis (45 sec)

> **"Historical purchase patterns (recency, frequency, and monetary value) can accurately predict a customer's future lifetime value."**

**What we're testing:**
1. Can RFM features predict future customer spending?
2. Which ML model performs best? (Linear vs. Random Forest vs. XGBoost)
3. Can we segment customers into actionable groups based on predicted CLV?

**Success Metrics:**
- Model accuracy: RMSE, MAE, R-squared
- Business value: Lift in identifying top 20% customers

---

## Slide 4: Data (1.5 min)

**Dataset:** Online Retail II (UCI Machine Learning Repository)

| Attribute | Details |
|-----------|---------|
| Source | UCI / Kaggle |
| Records | ~541,909 transactions |
| Time Period | Dec 2009 - Dec 2011 (2 years) |
| Domain | UK online giftware retailer |

**Original Columns:**
- Invoice, StockCode, Description, Quantity
- InvoiceDate, Price, Customer ID, Country

**Features We'll Engineer:**

| RFM Features | Behavioral Features |
|--------------|---------------------|
| Recency (days since last purchase) | Tenure (customer age) |
| Frequency (# of orders) | Avg time between purchases |
| Monetary (total spend) | # unique products bought |
| | Avg basket size |

**Target:** CLV = Total spend in next 6 months

---

## Slide 5: GitHub Project (45 sec)

**Repository Structure:**
```
Nerds-With-Attitude/
├── data/raw/          # Original dataset
├── data/processed/    # Cleaned data
├── notebooks/         # Analysis notebooks (01-05)
├── src/               # Reusable Python modules
├── reports/figures/   # Visualizations
└── README.md
```

**Project Board (Kanban):**
| Backlog | To Do | In Progress | Done |
|---------|-------|-------------|------|
| | Feature Engineering | Data Cleaning | Repo Setup |
| | Modeling | EDA | Data Download |
| | Segmentation | | |

**Notebooks:**
1. Data Exploration
2. Data Cleaning
3. Feature Engineering
4. Modeling (Linear, RF, XGBoost)
5. Customer Segmentation

---

## Summary

- **Problem:** Predict which customers will be most valuable
- **Data:** 500K+ retail transactions over 2 years
- **Approach:** RFM features + ML models + Customer segmentation
- **Goal:** Enable targeted marketing strategies

**Questions?**
