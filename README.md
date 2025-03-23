🧠 Customer Personality Prediction with TabNet
Welcome to the Customer Personality Prediction project! This machine learning pipeline combines clustering and deep tabular modeling to segment customers into behavioral “personalities” using marketing and demographic data. It uses KMeans to define clusters and TabNet (a powerful deep learning model for tabular data) to predict those clusters with high accuracy.

🚀 Features
End-to-End Workflow in Python (ready for Kaggle or Jupyter Notebooks)

📊 Feature Engineering for campaign behavior, total spend, family profile, and more

🤖 KMeans Clustering to define customer "personality" types:

Balanced Buyer

Budget Conscious

High Spender

Detached Minimalist

Campaign Enthusiast

🔥 TabNet Classifier trained to predict cluster using GPU (when available)

📈 Accuracy over 93% on validation data

📦 Input Predictor: Feed new customer data and get their predicted personality

📊 Visualizations for income, spending, recency, kids, and engagement per personality

📁 Data
The dataset used is the Marketing Campaign Dataset:

Source: Kaggle → customer-personality-analysis

Format: marketing_campaign.csv

⚙️ Installation
bash
Copy
Edit
pip install pytorch-tabnet category_encoders scikit-learn pandas numpy matplotlib seaborn
📌 Usage
Load and preprocess the dataset

Engineer features

Run KMeans to create 5 clusters

Train the TabNet model to predict cluster labels

Use the built-in input form to predict personality for new customers

📉 Model Performance
✅ TabNet Accuracy: ~93%

✅ Precision/Recall tracked for each personality

❗Handles missing income, unseen categories, and real-world input edge cases

