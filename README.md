# Parkinson's Disease Detection using Machine Learning

## 🧠 Overview
This project is an AI-powered, full-stack web application designed to detect Parkinson’s disease using biomedical voice measurements. 
By analyzing a patient's acoustic features, a trained **Random Forest Classifier** can predict whether the subject is healthy or has Parkinson's disease with high accuracy (~95%).

This repository features a **premium glassmorphic dark-themed UI**, interactive animated confidence charts, and backend generation of feature importance and confusion matrices based on the real UCI ML dataset.

![UI Preview](backend/confusion.png) <!-- Update later to a UI screenshot if needed -->

## ✨ Key Features
- **Accurate Predictions**: Uses 22 distinct voice features (Jitter, Shimmer, HNR, NHR, DFA, etc.) to make diagnoses.
- **Detailed Confidence Scores**: The backend computes class probabilities (e.g. 99% Parkinson's Detected).
- **Premium Frontend UX**: Glassmorphism blocks, gradient accents, fluid micro-animations, and a responsive layout.
- **Demo Mode**: One-click "Demo Values" button loads clinical test data from the original dataset for instant testing.
- **Model Interpretability**: Dynamically generates and serves top 10 Feature Importance bars and a Confusion Matrix graph from the dataset.
- **Printable Clinical Reports**: Print-friendly UI style to export prediction results cleanly.

## 🛠️ Technology Stack
- **Backend Model**: Python, scikit-learn (Random Forest Classifier)
- **Backend API**: Flask, Flask-CORS
- **Data & Viz**: Pandas, NumPy, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (No heavy frameworks required)

## 📊 The Dataset
The model is trained on the [UCI Parkinson’s dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) which contains 195 acoustic voice recordings from 31 people, 23 of whom have Parkinson's disease.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Dipeshrajjoshi/Parkinson-Disease-Detection-using-AIML.git
cd Parkinson-Disease-Detection-using-AIML
```

### 2. Set up the Python Environment
We recommend using a virtual environment.
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors joblib scikit-learn pandas matplotlib seaborn
```

### 3. Run the Backend API
Start the Flask server. It will load the `model.pkl` and `parkinsons.csv` dataset.
```bash
python app.py
```
*The server will run at `http://127.0.0.1:5000`.*

### 4. Launch the Frontend
Simply open `Frontend/index.html` in your favorite web browser.
```bash
# On macOS:
open ../Frontend/index.html
```

## ⚠️ Medical Disclaimer
This tool is built for **educational and academic research purposes only**. It relies entirely on statistical machine learning models and is not a substitute for professional medical diagnosis or clinical judgment. Always consult a qualified healthcare professional.

## 👨‍💻 Author
**Er. Dipesh Raj Joshi**
