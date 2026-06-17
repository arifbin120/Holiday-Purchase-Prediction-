# 🏖️ Holiday Package Purchase Prediction using Random Forest

A machine learning project to predict whether a customer will purchase a holiday/wellness travel package, enabling **Trips & Travel.Com** to optimize their marketing spend through targeted outreach.

---

## 📌 Problem Statement

**Trips & Travel.Com** offers 5 travel packages — *Basic, Standard, Deluxe, Super Deluxe, and King*. Historical data shows only **18% of customers** purchased packages, yet marketing costs were high because customers were contacted randomly.

The company is now launching a new **Wellness Tourism Package** and wants to use existing customer data to:
- Identify customers most likely to purchase
- Reduce unnecessary marketing expenditure
- Make data-driven outreach decisions

---

## 📂 Dataset

- **Source:** [Kaggle – Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- **Size:** 4,888 rows × 20 columns
- **Target Variable:** `ProdTaken` (1 = purchased, 0 = not purchased)

### Key Features

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `TypeofContact` | How the customer was contacted |
| `CityTier` | Tier of customer's city |
| `DurationOfPitch` | Duration of sales pitch (minutes) |
| `Gender` | Customer gender |
| `NumberOfFollowups` | Number of follow-up calls |
| `PreferredPropertyStar` | Star rating of preferred hotel |
| `NumberOfTrips` | Trips taken in the past year |
| `MonthlyIncome` | Monthly income |
| `MaritalStatus` | Marital status |

---

## 🔧 Project Pipeline

### 1. Data Cleaning
- Identified and corrected inconsistent category labels (e.g., `'Fe Male'` → `'Female'`, `'Single'` → `'Unmarried'`)
- Removed duplicate `CustomerID` column

### 2. Handling Missing Values

| Column | Imputation Strategy |
|---|---|
| `Age` | Median |
| `TypeofContact` | Mode |
| `DurationOfPitch` | Median |
| `NumberOfFollowups` | Mode |
| `PreferredPropertyStar` | Mode |
| `NumberOfTrips` | Median |
| `NumberOfChildrenVisiting` | Mode |
| `MonthlyIncome` | Median |

### 3. Feature Engineering
- Created a new feature: `TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting`
- Dropped the two source columns after combining

### 4. Preprocessing
- **Categorical features:** One-Hot Encoding (`drop='first'` to avoid multicollinearity)
- **Numerical features:** Standard Scaling
- Applied via `ColumnTransformer` with 80/20 train-test split

---

## 🤖 Models Compared

Four classifiers were evaluated before selecting the best performer:

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Simple tree-based classifier |
| **Random Forest** ✅ | Ensemble of decision trees — selected |
| Gradient Boosting | Sequential ensemble method |

---

## 🌲 Random Forest – Hyperparameter Tuning

`RandomizedSearchCV` with 3-fold cross-validation was used to find the optimal parameters.

**Search Space:**

```python
rf_params = {
    "max_depth": [5, 8, 15, None, 10],
    "max_features": [5, 7, "auto", 8],
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators": [100, 200, 500, 1000]
}
```

**Best Parameters Found:**

```python
RandomForestClassifier(
    n_estimators=1000,
    min_samples_split=2,
    max_features=7,
    max_depth=None
)
```

---

## 📊 Model Evaluation Metrics

The tuned Random Forest was evaluated on:

- **Accuracy**
- **F1 Score** (weighted)
- **Precision**
- **Recall**
- **ROC-AUC Score** (~0.83)

A **ROC-AUC curve** was plotted and saved as `auc.png`.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Run the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction) and save it as `Travel.csv`.
2. Open the notebook in **Google Colab** or **Jupyter Notebook**.
3. Run all cells sequentially.

> **Colab users:** Upload `Travel.csv` to `/content/` before running data loading cells.

---

## 📁 Project Structure

```
├── HOLIDAY_PURCHASE_PREDICTION_USING_RANDOM_FOREST.ipynb   # Main notebook
├── Travel.csv                                               # Dataset (download from Kaggle)
├── auc.png                                                  # ROC-AUC curve output
└── README.md                                               # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Static visualizations |
| Plotly | Interactive visualizations |
| Scikit-learn | ML models, preprocessing, evaluation |

---

## 📈 Key Takeaway

By predicting which customers are likely to purchase the Wellness Tourism Package, **Trips & Travel.Com** can focus marketing efforts on high-probability leads — significantly reducing costs while improving conversion rates compared to the previous random-contact approach.
