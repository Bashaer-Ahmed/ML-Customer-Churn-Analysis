# Customer Churn Prediction - ML Project

## BUS8405 Machine Learning Assignment

This project implements a comprehensive machine learning solution for predicting customer churn in an online retail environment. The analysis addresses critical business challenges including customer retention, revenue protection, and marketing optimization.

## Project Overview

This ML project analyzes customer behavior patterns to predict churn risk and develop targeted retention strategies. The implementation covers the complete data science pipeline from data analysis to business recommendations.

## Project Structure

```
ML/
├── data/
│   └── raw/
├── src/
│   ├── data_analysis.py
│   ├── model_comparison.py
│   ├── model_training.py
│   └── business_insights.py
├── models/
├── results/
├── requirements.txt
└── main.py
```

## Dataset

- **Source**: Kaggle - Online Retail Customer Churn Dataset
- **Size**: 1000 customers, 15 features
- **Target**: Binary classification (Churn: True/False)

## Key Features

- Customer demographics (Age, Gender, Annual_Income)
- Purchase behavior (Total_Spend, Num_of_Purchases, Years_as_Customer)
- Engagement metrics (Satisfaction_Score, Email_Opt_In, Promotion_Response)
- Service interactions (Num_of_Support_Contacts, Num_of_Returns)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle
2. Run the complete analysis: `python main.py`

## Implementation Results

**Model Performance:**
- Best Algorithm: K-Nearest Neighbors (ROC-AUC: 0.5545)
- Final Training: RandomForest (ROC-AUC: 0.4980, F1: 0.4808)
- Business Impact: 52.6% churn rate identified
- Financial Analysis: $3.28M potential revenue at risk

**Technical Features:**
- Complete ML pipeline implementation
- Hyperparameter optimization
- Cross-validation and proper evaluation
- Comprehensive business analysis
- Professional visualizations

## Project Implementation

This ML project addresses all BUS8405 assignment requirements:

**CLO1 - Dataset Analysis:**
- Comprehensive data exploration and quality assessment
- Feature significance analysis for business context
- Statistical profiling and correlation analysis

**CLO2 - Model Selection:**
- Comparison of 9 different ML algorithms
- Justified model selection based on performance metrics
- Business suitability evaluation

**CLO3 - Model Development:**
- Complete training/validation/testing pipeline
- Hyperparameter tuning and optimization
- Comprehensive performance evaluation

**CLO4 - Business Solutions:**
- Customer segmentation and risk analysis
- Financial impact assessment
- Strategic retention recommendations

## Author
Bashaer Ahmed
BUS8405 Machine Learning Assignment Implementation
