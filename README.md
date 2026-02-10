# Air Quality Index (AQI) Regression Analysis

## 📌 Objective

The objective of this project is to perform a comprehensive study of different Linear Regression techniques for predicting Air Quality Index (AQI). The analysis includes Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, and Regularization methods (Ridge and Lasso).

This project was completed as part of a laboratory assignment to understand regression modeling, evaluation, and diagnostics in a practical setting.

---

## 📊 Dataset Description

The dataset used in this project is the *Air Quality Data in India (CPCB)* dataset.  
We used the file `city_day.csv`, which contains daily pollutant measurements across different cities.

### Selected Features:
- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

### Target Variable:
- AQI (Air Quality Index)

Missing values were handled by removing rows containing null values to ensure clean model training.

---

## 🔍 Part A: Exploratory Data Analysis (EDA)

The dataset was explored using:
- Summary statistics
- Missing value analysis
- Boxplots (for outlier detection)
- Histograms (to analyze distribution)
- Correlation heatmap

Key observations:
- PM2.5 and PM10 show strong correlation with AQI.
- The dataset contains several extreme values (real pollution spikes).
- Most pollutant variables are right-skewed.
- Multicollinearity exists between some pollutants.

---

## 📈 Part B: Simple Linear Regression

A simple linear regression model was built using PM2.5 as the independent variable and AQI as the target.

Observation:
- The model captured the general trend.
- However, since AQI depends on multiple pollutants, performance was limited compared to multi-feature models.

---

## 📊 Part C: Multiple Linear Regression

A multiple linear regression model was built using six pollutant features.

### Evaluation Metrics:
- MSE
- RMSE
- R² Score

The model achieved a high R² score (~0.89), indicating that nearly 90% of the variance in AQI is explained by the selected pollutants.  
This confirms that AQI depends on a combination of pollutants rather than a single factor.

---

## 📉 Part D: Polynomial Regression

Polynomial regression (degree 2) was applied to capture non-linear relationships between pollutants and AQI.

Comparison with linear regression showed that:
- Polynomial regression slightly improved the fit in some cases.
- The relationship between pollutants and AQI is mostly linear, with mild non-linear effects.

---

## 🛠 Part E: Regularization (Ridge & Lasso)

To address multicollinearity, Ridge and Lasso regression were applied.

- **Ridge Regression** reduced coefficient magnitudes while keeping all features.
- **Lasso Regression** shrank some coefficients toward zero, effectively performing feature selection.

This helped improve model stability and interpretability.

---

## 📌 Part F: Model Diagnostics

Residual analysis was performed to validate regression assumptions.

- Residuals were randomly distributed around zero.
- No strong funnel pattern was observed.
- Residual distribution was approximately normal.

This indicates that the regression assumptions were reasonably satisfied.

---

## 📈 Final Results

- Multiple Linear Regression performed significantly better than Simple Linear Regression.
- Polynomial Regression captured slight non-linear patterns.
- Regularization techniques improved coefficient stability.
- The final model explains a large portion of AQI variation using pollutant data.

---

## 🧠 Conclusion

This project demonstrates how regression techniques can be used to model environmental data effectively. The analysis shows that particulate matter (PM2.5 and PM10) plays a major role in determining AQI levels. 

Regularization methods such as Ridge and Lasso are useful when dealing with correlated features, improving model robustness.

Overall, the study provides practical insight into regression modeling, evaluation metrics, and diagnostic validation.

---

## 🛠 Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📂 Repository Structure
air-quality-index-regression-analysis
│
├── city_day.csv
├── air-quality-index-regression-analysis.ipynb
├── README.md
└── .gitignore
