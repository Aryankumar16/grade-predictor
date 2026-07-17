# 🎓 Student Performance Prediction System

## 📖 Overview

The **Student Performance Prediction System** is an end-to-end Machine Learning application that predicts student academic performance using two complementary predictive models. The project combines **classification** and **regression** techniques to provide meaningful insights into a student's academic progress and expected examination outcomes.

The application is built using **Python**, **Scikit-learn**, **XGBoost**, and **Streamlit**, providing an interactive web interface where users can enter student-related information and obtain real-time predictions.

---

## ✨ Features

- 🎓 Predict student performance category (Excellent, Good, Average, Poor)
- 📈 Predict expected examination score (0–100)
- 🌳 Decision Tree Classification Model
- ⚡ XGBoost Regression Model
- 📊 Interactive visualizations using Plotly
- 💻 User-friendly Streamlit interface
- 🚀 Real-time prediction with personalized feedback

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib

---

## 📂 Project Files

```
app.py
requirements.txt
README.md

dt_model.pkl
xgb.pkl
encoders.pkl
features.pkl

student-mat.csv
student_habits_performance.csv

Classification_Model.ipynb
Regression_Model.ipynb
```

---

## 🤖 Machine Learning Models

### Model 1 – Student Performance Classification

**Algorithm:** Decision Tree Classifier

**Input Features**

- Study Time
- Previous Failures
- Activities
- Internet Access
- Absences
- Going Out Frequency
- G1
- G2

**Output**

- Excellent
- Good
- Average
- Poor

**Performance**

- Accuracy: **~85%**
- Cross Validation Accuracy: **~85.09%**

---

### Model 2 – Early Examination Score Prediction

**Algorithm:** XGBoost Regressor

**Input Features**

- Study Hours
- Mental Health Rating
- Exercise Frequency
- Sleep Hours
- Social Media Hours
- Entertainment/Streaming Hours
- Attendance Percentage
- Internet Quality
- Extracurricular Participation

**Engineered Features**

- Distraction Hours
- Study vs Distraction
- Study × Mental Health Interaction

**Performance**

- R² Score: **0.876**
- MAE: **≈ 4.5**
- RMSE: **≈ 5.6**

---

## 📊 Workflow

```
User Input
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ├──────────────┐
      ▼              ▼
Decision Tree    XGBoost
Classifier       Regressor
      │              │
      └──────┬───────┘
             ▼
   Streamlit Web App
             ▼
 Predictions & Visualizations
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Aryankumar16/grade-predictor.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📚 Datasets

### Student Performance Dataset

UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/student+performance

### Student Habits vs Academic Performance Dataset

Kaggle

https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance

---

## 🔮 Future Improvements

- Deep Learning models
- Personalized study recommendations
- Explainable AI
- Student login system
- Academic progress tracking
- Institution-specific datasets

---

## 👨‍💻 Author

**Aryan Kumar**

B.Tech Computer Science & Engineering

Lovely Professional University

---

## 📄 License

This project is intended for educational and academic purposes.
