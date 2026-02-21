# 🍷 Wine Quality Prediction using Machine Learning

# 📌 Project Overview
This project builds a Machine Learning model to predict wine quality based on physicochemical properties such as acidity, sugar content, pH level, sulphates, and alcohol concentration.

The goal is to simulate a real-world industry use case where wineries can use predictive analytics to improve product consistency, reduce manual quality testing, and optimize production parameters.

# 🎯 Business Objective
Wine quality traditionally depends on expert sensory evaluation. This project demonstrates how machine learning can:

- Predict wine quality automatically
- Reduce reliance on subjective testing
- Provide data-driven production insights
- Identify key chemical factors influencing quality

# 📊 Dataset Information
- Total Samples: 6,497
- Wine Types: Red & White
- Features:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
  - Color (encoded)
- Target Variable: Quality Score

# 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

# 🤖 Machine Learning Model
Model Used: **Random Forest Regressor**

Why Random Forest?
- Handles non-linear relationships
- Reduces overfitting through ensemble learning
- Provides feature importance for interpretability

# 📈 Model Evaluation Metrics
- R² Score
- Root Mean Squared Error (RMSE)

# 🔍 Key Insights
- Alcohol concentration strongly influences wine quality.
- High volatile acidity negatively impacts quality.
- Sulphates and citric acid contribute moderately.
- Machine learning can effectively model chemical-quality relationships.

# 🚀 How to Run the Project

## 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Wine-Quality-Prediction-ML.git

## 2️⃣ Install Dependencies
pip install -r requirements.txt

## 3️⃣ Run Project
python wine_quality_prediction.py
Graphs will be saved inside the outputs/ folder.

📂 Project Structure
Wine-Quality-Prediction-ML/
│
├── wine_quality_prediction.py
├── winequality.csv
├── requirements.txt
├── README.md
└── outputs/

##🔮 Future Improvements

Hyperparameter tuning
Implement XGBoost / LightGBM
Model deployment using Flask or FastAPI
Interactive dashboard integration

##💡 Learning Outcomes

End-to-end ML workflow implementation
Feature importance analysis
Business-oriented data interpretation
Reproducible project structuring
