# ML-Group08
Cars4You Machine Learning Project
# Cars 4 You - Car Price Prediction Project

## Project Overview
This project was developed for the "Cars 4 You" company to address the growing demand for car evaluations. The main goal is to build a Machine Learning regression model capable of accurately predicting the resale price of a vehicle based on its characteristics, eliminating the need for immediate physical inspection by a mechanic.

## 📊 Dataset
The dataset contains approximately 75,000 records of used cars sold in 2020. The target variable is `price`.
Key features include:
* **Categorical:** `Brand`, `model`, `transmission`, `fuelType`.
* **Numerical:** `year`, `mileage`, `tax`, `mpg`, `engineSize`, `paintQuality%`, `previousOwners`.

## 🛠️ Project Pipeline
The solution follows a structured Data Science pipeline:

### 1. Data Cleaning & Preprocessing
* **Typo Correction:** Standardized categorical values (e.g., correcting "TOYOT" to "TOYOTA", handling variations in fuel types).
* **Data Type Conversion:** Converted float columns like `year` and `previousOwners` to integers.
* **Handling Invalid Data:** * Removed the `hasDamage` column (contained only 0 or NaN).
    * Corrected negative values in `mileage` and `engineSize`.
    * Capped variables to realistic bounds (e.g., `year` <= 2020, `paintQuality%` <= 100).
* **Missing Value Imputation:** Used median imputation grouped by Brand/Model for features like `engineSize`, `tax`, and `mpg`.

### 2. Feature Engineering
* **Target Transformation:** Applied `log(price)` transformation to normalize the target distribution and improve linear model performance.
* **New Features:**
    * `car_age`: Calculated as `2020 - year`.
    * `brand_model_avg_price`: Average price per brand/model combination.
    * **Depreciation Metrics:** Created sophisticated features like `km_semielasticity_10k` and `expected_km_discount_pct` to capture the non-linear impact of mileage on price.

### 3. Encoding & Scaling
* **One-Hot Encoding:** Applied to low-cardinality categorical features (`Brand`, `fuelType`, `transmission`).
* **Frequency Encoding:** Applied to the high-cardinality `model` feature to avoid dimensionality explosion.
* **Scaling:** Applied `MinMaxScaler` to all numerical features.

### 4. Feature Selection
A rigorous selection process was applied to reduce dimensionality and overfitting:
* **Filter Methods:** Removed constant variables; analyzed Spearman correlation matrix (removed `mileage` and `year` due to redundancy); Chi-Square test for categorical relevance.
* **Wrapper Methods:** Recursive Feature Elimination (RFE).
* **Embedded Methods:** Lasso Regression and Permutation Importance.
* **Result:** Reduced the feature set by removing low-importance variables (e.g., certain brand dummies, `paintQuality%`).

### 5. Modeling & Optimization
* **Benchmarking:** Tested multiple regressors including:
    * Linear Models: Linear Regression, Ridge, Lasso, ElasticNet.
    * Ensemble Models: HistGradientBoostingRegressor.
* **Hyperparameter Tuning:** Used `RandomizedSearchCV` to optimize the `HistGradientBoostingRegressor`, minimizing the Log-MAE.
* **Final Model:** The optimized **HistGradientBoostingRegressor** was selected for its superior performance (R² ~0.93).

## Results
The final model demonstrates strong predictive power:
* **R² Score:** ~0.93 (Validation)
* **Key Drivers:** Engine size, car age, and the custom mileage depreciation features were among the most influential predictors.
