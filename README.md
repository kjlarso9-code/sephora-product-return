# Sephora Product Return Risk – Final Project

Predicting return risk in beauty products using machine learning and an interactive web app.

## 1. Problem Overview

Beauty retailers like Sephora face high product return rates, which hurt margins and create inventory and logistics issues.  
The goal of this project is to **predict the probability that a product is high return risk** based on characteristics like brand, category, price, review volume, and customer engagement (“loves”).

## 2. Data

- Source: Sephora website product data (`sephora_website_dataset.csv`)
- Each row = a product.
- Key fields used:
  - `brand`
  - `category`
  - `price`
  - `number_of_reviews`
  - `love`
  - `rating` (used to build the target label)

### Target Engineering

- Filtered out products with missing or zero ratings.
- Defined binary target:

  \`high_return_risk = 1\` if **rating < 3.0**, else \`0\`.

## 3. Modeling (Execution)

All model development was done in **Databricks**.

Notebook: `Sephora_Return_Risk_Classification` (link provided in deliverables).

Steps:

1. **Preprocessing**
   - Dropped ID / URL / long-text fields not useful for this first model.
   - Split features into:
     - Numeric: price, review count, loves, etc.
     - Categorical: brand, category, etc.
   - Numeric pipeline: `SimpleImputer(strategy="mean")` → `StandardScaler`.
   - Categorical pipeline: `OneHotEncoder(handle_unknown="ignore")`.

2. **Models Trained**
   - **MLPClassifier (Neural Network)** – several hyperparameter settings.
   - **SVC (Support Vector Classifier)** – linear and RBF kernels, multiple C values.
   - Logged metrics with **MLflow**:
     - Accuracy
     - F1
     - ROC-AUC
   - Logged artifacts:
     - Confusion matrix plots
     - Probability histograms

3. **Tracking with MLflow**
   - Experiment name: `/Users/kjlarso9@asu.edu/SephoraProductReturnRisk_Classification`.
   - Each run includes parameters, metrics, plots, and the saved model.

*(You can add your best F1/accuracy numbers here.)*

## 4. Deployed App (So What?)

The deployed solution is a **Flask web app** hosted on **Render**:

- **App URL:** `<your Render URL here>`
- File: `sephora_product_return_app.py`

### How it works

- On startup, the app:
  - Loads `sephora_website_dataset.csv`
  - Recreates the target label (`high_return_risk`)
  - Trains a simple pipeline model using:
    - Features: `brand`, `category`, `price`, `number_of_reviews`, `love`
    - Preprocessing: imputation, scaling, and one-hot encoding
    - Classifier: `LogisticRegression`

- The home page lets the user input:
  - Brand
  - Category
  - Price (USD)
  - Number of reviews
  - Loves count

- The app outputs:
  - **Prediction:** High vs Low return risk
  - **Probability:** estimated P(high_return_risk = 1)

This demonstrates an end-to-end AI lifecycle:

1. Data prep and feature engineering  
2. Model development and experiment tracking in Databricks (MLflow)  
3. Deployed web app for real-time user interaction

## 5. Running the Project Locally

### Requirements

```bash
pip install -r requirements.txt
