# trafficprediction

# 🚗 Road Accident Prediction System using Machine Learning & Deep Learning

A machine learning project that predicts the **probability of road accidents** using real-world accident-related features such as weather conditions, road conditions, speed limits, driver information, and traffic control. The project compares the performance of **Random Forest** and **LSTM** models to provide accurate accident risk predictions.

---

# 📌 Project Overview

This project focuses on predicting accident severity by preprocessing accident data, training both traditional Machine Learning and Deep Learning models, and allowing users to interactively enter accident-related details to estimate the probability of an accident.

The system performs:

* Data Cleaning
* Missing Value Imputation
* Feature Encoding
* Feature Scaling
* Model Training
* Model Evaluation
* Model Saving
* Interactive Prediction

---

# ✨ Features

* 📂 Automatic Dataset Loading
* 🧹 Missing Value Handling
* 🔢 Label Encoding for Categorical Data
* 📊 Standard Feature Scaling
* 🌲 Random Forest Classification
* 🧠 LSTM Neural Network
* 📈 ROC-AUC Evaluation
* 📋 Classification Report
* 💾 Model & Scaler Saving
* 🚦 Interactive Prediction System
* ⚠️ High / Low Accident Risk Detection

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Joblib

---

# 📚 Machine Learning Workflow

1. Load Dataset
2. Select Required Features
3. Handle Missing Values
4. Encode Categorical Features
5. Convert Target Variable
6. Split Dataset
7. Scale Features
8. Train Random Forest Model
9. Train LSTM Model
10. Evaluate Models
11. Save Trained Models
12. Predict Accident Probability

---

# 📂 Project Structure

```text
Road-Accident-Prediction/
│
├── accident_prediction_india.csv
├── accident_prediction.py
├── random_forest.pkl
├── lstm_model.h5
├── scaler.save
├── label_encoders.pkl
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Features

### Numerical Features

* Driver Age
* Speed Limit (km/h)

### Categorical Features

* State Name
* Vehicle Type Involved
* Weather Conditions
* Road Condition
* Traffic Control Presence
* Driver Gender
* Alcohol Involvement

### Target Variable

* Accident Severity (Binary Classification)

---

# 🤖 Machine Learning Models

## 🌲 Random Forest

* Ensemble Learning Algorithm
* High Accuracy
* Handles Mixed Data Efficiently
* Generates Accident Probability

---

## 🧠 LSTM (Long Short-Term Memory)

* Deep Learning Model
* Built using TensorFlow/Keras
* Binary Classification
* Predicts Accident Probability

---

# 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Classification Report

---

# 🚦 Interactive Prediction

The application allows users to enter accident-related information such as:

* Driver Age
* Speed Limit
* Weather Condition
* Road Condition
* Vehicle Type
* Driver Gender
* Alcohol Involvement
* Traffic Control Presence
* State Name

The system processes the input and predicts:

* 🌲 Random Forest Accident Probability
* 🧠 LSTM Accident Probability
* 📊 Average Probability
* ⚠️ High Risk or ✅ Low Risk Decision

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/road-accident-prediction.git
```

Move into the project folder

```bash
cd road-accident-prediction
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python accident_prediction.py
```

---

# 📦 Required Libraries

```text
pandas
numpy
scikit-learn
tensorflow
joblib
```

Or install manually:

```bash
pip install pandas numpy scikit-learn tensorflow joblib
```

---

# 📸 Output

The application provides:

* Cleaned Dataset
* Random Forest Performance
* LSTM Performance
* Classification Report
* ROC-AUC Score
* Interactive User Prediction
* Accident Risk Probability
* High/Low Risk Alert

  <img width="1847" height="862" alt="Screenshot 2026-07-01 110619" src="https://github.com/user-attachments/assets/afcd273d-5d37-40fd-983d-e8a9a0850c41" />

  <img width="1026" height="462" alt="Screenshot 2026-07-01 110644" src="https://github.com/user-attachments/assets/ad0231c7-1c82-4a27-9d84-cf13cf4905cb" />
  
<img width="1525" height="870" alt="Screenshot 2026-07-01 110638" src="https://github.com/user-attachments/assets/e9dfcd1f-a6e4-40fb-bb7d-afe5b82dc38a" />



---

# 🎯 Future Improvements

* 🌐 Web Application using Flask/Django
* 📍 Live GPS Integration
* 🗺️ Interactive Accident Heat Maps
* ☁️ Cloud Deployment
* 📱 Mobile Application
* 🤖 Real-time AI Prediction API
* 📊 Dashboard & Analytics
* 🔔 Accident Warning Notifications

---

# 👨‍💻 Developed By

**Divyanshu Negi**

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.
