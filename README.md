# Customer Personality Analysis with KMeans and TabNet

## Overview
This project analyzes customer personalities using a hybrid approach: KMeans clustering to identify five personality types and TabNet to predict these clusters for new customers. It includes data preprocessing, feature engineering, visualization, and prediction, executed in a Kaggle environment.

## Functionality
1. **Data Preprocessing**:
   - Loads `/kaggle/input/customer-personality-analysis/marketing_campaign.csv`.
   - Drops `ID`, `Z_CostContact`, `Z_Revenue`, `Dt_Customer`; fills missing `Income` with median.
   - Encodes `Education` and `Marital_Status` with `OneHotEncoder`; scales numerical features.

2. **Feature Engineering**:
   - `TotalChildren`: `Kidhome + Teenhome`.
   - `TotalSpend`: Sum of spending categories.
   - `CampaignResponses`: Sum of campaign acceptances.

3. **Clustering**:
   - Uses KMeans (5 clusters) to group customers.
   - Labels: "Balanced Buyer," "Budget Conscious," "High Spender," "Detached Minimalist," "Campaign Enthusiast."

4. **TabNet Classification**:
   - Trains TabNet to predict KMeans clusters with early stopping.

5. **Visualization**:
   - Bar plots for average `TotalSpend`, `TotalChildren`, `CampaignResponses`, `Recency`, and `Income` by personality.

6. **Prediction**:
   - Predicts personality for new customer data.

## Frameworks and Libraries
- **Python**: Core language (Kaggle Python 3).
- **Pandas**: Data handling (`pd`).
- **NumPy**: Numerical operations (`np`).
- **Scikit-learn**: `train_test_split`, `StandardScaler`, `KMeans`, `classification_report`.
- **Category Encoders**: `OneHotEncoder`.
- **PyTorch TabNet**: `TabNetClassifier`.
- **Matplotlib**: Plotting (`plt`).
- **Seaborn**: Visualization (`sns`).

## Dataset
- Source: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).
- Features: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, spending, campaigns, etc.

## Key Features
- **Hybrid Learning**: KMeans + TabNet.
- **Feature Engineering**: Aggregates for better insights.
- **Visualization**: Interpretable personality profiles.
- **Scalability**: Handles mixed data types.

## Installation
```bash
pip install pytorch-tabnet category_encoders
pip install pandas numpy scikit-learn matplotlib seaborn
