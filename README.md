# 📈 Time Series Forecasting for Cryptocurrency Price Direction

### Project Overview
This project aims to predict the direction of cryptocurrency price movement using time series forecasting techniques. The goal is to classify whether the price will increase (1) or stay the same/decrease (0) based on historical data. The project involves data preprocessing, feature engineering, exploratory data analysis, model selection, and evaluation.

### Dataset
The project uses two datasets:

**train.csv** – Contains historical cryptocurrency price data for training the model.\
**test.csv** – Used for evaluating the model’s performance.

### Data Preprocessing
* Load and inspect datasets for missing values and outliers.
* Perform time series analysis to understand trends and seasonality.
* Normalize numerical features to ensure consistent model performance.

### Feature Engineering
To enhance predictive performance, the following features were engineered:

* Price Change – Measures absolute price movement.
* Log Return – Captures relative price fluctuations.
* Volatility – Assesses market stability.
* Lag Features – Incorporates past price movements to improve prediction accuracy.
* Rolling Statistics – Calculates moving averages for trend analysis.

### Exploratory Data Analysis (EDA)
* Visualized time series trends in cryptocurrency prices.
* Correlation analysis to retain highly relevant features.
* Distribution analysis of key variables to understand patterns.

### Modeling Approac
Various machine learning models were tested:

* Logistic Regression – Baseline classifier for directional forecasting.
* Random Forest Classifier – Captures complex relationships in the data.
* XGBoost Classifier – Optimized boosting-based model for improved accuracy.

### Hyperparameter Tuning
* Used GridSearchCV and TimeSeriesSplit for optimizing hyperparameters.
* Evaluated models using accuracy, precision, recall, and F1-score.

### Results & Insights
* The XGBoost classifier provided the highest predictive accuracy.
* Feature selection significantly improved model performance.
* Handling class imbalance (if present) was crucial for robust classification.

### Next Steps & Future Improvements
* Experiment with deep learning models like LSTMs or Transformers for sequential data learning.
* Incorporate additional data sources such as trading volume, social media sentiment, or macroeconomic indicators.
* Deploy the model using a real-time cryptocurrency trading API.

### How to Run the Project
Install dependencies:\
```pip install pandas numpy scikit-learn xgboost matplotlib seaborn```

Run the Jupyter Notebook:\
```jupyter notebook FYH_coding_challenge.ipynb```

### Author
Francy Hsu
