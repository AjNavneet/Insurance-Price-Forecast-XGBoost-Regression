# Insurance Pricing Forecast Using XGBoost Regressor

## Project Overview

Insurance companies provide coverage for expenses incurred by policyholders due to damages to health or property. These policies include medical bills, house insurance, motor vehicle insurance, and fire insurance, along with financial losses such as loss of income. Traditional methods of premium calculation are time-consuming and struggle to capture complex data interactions.

---

## Objective

In this project, the goal is to build a machine learning model using XGBoost Regressor to predict healthcare expenses based on features such as age, BMI, smoking, etc. The aim is to help the insurance firm establish accurate premium rates and maximize profits.

---

### Data Description
The dataset contains historical records of 1338 insured customers with the following columns:
- Age: Age of the primary beneficiary.
- Sex: Gender of the primary beneficiary.
- BMI: Body mass index of the primary beneficiary.
- Children: Number of children the primary beneficiary has.
- Smoker: Whether the primary beneficiary smokes.
- Region: The primary beneficiary's residential area in the US.
- Charges: Individual medical costs billed by health insurance.

---

### Tech Stack
- Language: `Python`
- Libraries: `pandas`, `numpy`, `matplotlib`, `plotly`, `statsmodels`, `sklearn`, `xgboost`, `skopt`

---

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Distributions
   - Univariate Analysis
   - Bivariate Analysis
   - Correlation
     - Pearson Correlation
     - Chi-squared Tests
     - ANOVA

2. **Build**
   - Linear regression assumptions
   - Data preprocessing
   - Model training
   - Model evaluation (RMSE)

3. **Improve on the baseline linear model**
   - Introduction to a non-linear model - XGBoost
   - Data preprocessing
   - Using Sklearn's `Pipeline` to optimize the model training process
   - Model evaluation (RMSE)
   - Comparison to the baseline model

4. **Presenting the results to non-technical stakeholders**

---

## Modular Code Overview

- The `lib` folder contains the original ipython notebook.
- The `ml_pipeline` folder contains functions organized into different Python files. `engine.py` calls these functions to run the steps and print the results.
- The `requirements.txt` file lists the required libraries with their respective versions.

---

## Concepts Explored

- Exploratory Data Analysis on Categorical and Continuous Data
- Univariate Data Analysis
- Bivariate Data Analysis
- Correlation Analysis
- Categorical Correlation with Chi-squared
- Correlation between Categorical and Target Variables with ANOVA
- Label Encoding for Categorical Variables
- Understanding Linear Regression Assumptions
- Implementing Linear Regression
- Validating Linear Regression Assumptions
- Implementing XGBoost Regressor
- Building pipelines with Sklearn’s Pipeline operator
- BayesSearchCV for XGBoost Hyperparameter Optimization
- Evaluating Models with Regression Metrics (RMSE)
- Presenting Non-Technical Metrics for Stakeholders

---

## Execution Instructions

1. Create a Python environment using the command 'python3 -m venv myenv'.

2. Activate the environment by running the command 'myenv\Scripts\activate.bat'.

3. Install the required packages using the command 'pip install -r requirements.txt'.

4. Run the project's main script 'engine.py' with the command 'python3 engine.py'.

---
