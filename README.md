# 🌍 Air Quality Prediction Using Regression Models

## 📌 Project Overview

This project applies **Machine Learning Regression techniques** to analyze air pollution data and predict the **Air Quality Index (AQI)** based on various environmental and pollution-related factors. The project also includes Exploratory Data Analysis (EDA), data visualization, model comparison, and diagnostic evaluation.

---

## 🎯 Objective

The main objectives of this project are:

* To analyze the impact of different pollutants on air quality
* To build and compare different regression models
* To evaluate model performance using statistical metrics
* To understand relationships between environmental factors and AQI

---

## 📂 Dataset

The dataset used in this project:
**`updated_pollution_dataset.csv`**

### Features in the dataset:

* Temperature
* Humidity
* PM2.5
* PM10
* NO2
* SO2
* CO
* Proximity to Industrial Areas
* Population Density
* Air Quality (Good, Moderate, Poor)

> 🔹 Since machine learning models require numeric data, the categorical Air Quality labels were converted as follows:
>
> | Air Quality | AQI Value |
> | ----------- | --------- |
> | Good        | 50        |
> | Moderate    | 100       |
> | Poor        | 150       |

---

## ⚙️ Project Workflow

### **Part A: Exploratory Data Analysis (EDA)**

* Loaded dataset and displayed basic information
* Computed summary statistics
* Checked missing values
* Visualized:

  * Feature distributions (histograms)
  * Correlation heatmap

### **Part B: Simple Linear Regression**

* Used **PM2.5** to predict AQI
* Plotted regression line
* Interpreted slope and intercept

### **Part C: Multiple Linear Regression**

* Used multiple features: **PM2.5, PM10, NO2**
* Evaluated using:

  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * R² Score

### **Part D: Polynomial Regression**

* Applied degree-2 polynomial regression
* Compared performance with linear regression

### **Part E: Regularization**

* Applied:

  * Ridge Regression
  * Lasso Regression
* Compared coefficients and feature importance

### **Part F: Model Diagnostics**

* Plotted residuals vs predicted values
* Validated regression assumptions

---

## 📊 Evaluation Metrics Used

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

---

## 🧠 Key Findings

* PM2.5 and PM10 have a strong impact on air quality
* Multiple and Polynomial Regression perform better than Simple Regression
* Ridge and Lasso help reduce overfitting
* Residual analysis confirms model reliability

---

## 🛠️ Technologies Used

* **Python**
* Libraries:

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone <your-repo-link>
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the script:

```bash
python Air_Quality_Regression_Project.py
```

---

## 📈 Future Improvements

* Use real-time air quality sensor data
* Include more pollutants
* Apply Deep Learning models
* Use actual AQI values instead of label mapping

---

## 👨‍💻 Author

**Rajyavardhan Radhey**
CSE-AI Student, CSJMA University, Kanpur

---

## ⭐ Acknowledgment

Thanks to all open-source contributors and datasets that helped in this project.
