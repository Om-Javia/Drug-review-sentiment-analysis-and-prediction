# Drug Review Sentiment Analysis and Prediction

A project to analyze and predict the sentiment of drug reviews using natural language processing (NLP) and machine learning models.

## 📄 Project Summary
This project leverages a dataset containing drug reviews to build sentiment analysis models. The main goal is to preprocess the text data and apply machine learning algorithms to predict whether the sentiment of a new drug review is positive or negative.

## 🗂️ Key Sections

### 1. **Data Loading & Exploration**
- **Objective**: Load the dataset and conduct exploratory data analysis (EDA).
- **Steps**:
  - Data structure overview
  - Visualize key statistics and sentiment distribution

### 2. **Data Preprocessing**
- **Objective**: Prepare the data for model building.
- **Techniques**:
  - Text cleaning (removing stop words, punctuation, etc.)
  - Handling missing values and label encoding

### 3. **Sentiment Analysis**
- **Objective**: Use NLP techniques to derive insights from the reviews.
- **Steps**:
  - Tokenization and vectorization (TF-IDF)
  - Sentiment labeling based on review content

### 4. **Model Building**
- **Objective**: Build machine learning models to predict review sentiment.
- **Models**:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest

### 5. **Model Evaluation**
- **Objective**: Assess the performance of each model.
- **Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix and ROC curve

## 💻 Requirements

Ensure the following libraries are installed before running the code:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk`

### Installation:
Install the required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
