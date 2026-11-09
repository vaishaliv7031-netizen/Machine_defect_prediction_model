# Machine Failure Prediction System

## 📌 Project Overview

The **Machine Failure Prediction System** is a Machine Learning-based application designed to predict potential machine failures using operational sensor data.

The system analyzes five important machine parameters: Temperature, Pressure, Vibration Level, Humidity, and Power Consumption.

It uses a trained Scikit-learn classification model to estimate failure probability. FastAPI handles backend requests and input validation, while Streamlit provides an interactive user interface for making predictions.

## 🎯 Project Objectives

- Predict potential machine failures using sensor readings.
- Identify possible machine failure conditions.
- Calculate machine failure probability.
- Validate incoming data before model prediction.
- Provide predictions through an interactive web application.
- Expose machine learning predictions through REST API endpoints.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization and loading |
| FastAPI | Backend API development |
| Pydantic | Input data validation |
| Streamlit | Frontend application |
| Uvicorn | ASGI server |

## 📊 Input Features

The model accepts five sensor parameters:

1. Temperature
2. Pressure
3. Vibration Level
4. Humidity
5. Power Consumption

**Target Variable:** `Failure_Status`

- `0` – No Failure
- `1` – Failure

## 🤖 Machine Learning Workflow

1. Data loading and exploration.
2. Data cleaning and preprocessing.
3. Train-test splitting using stratification.
4. Training and comparison of multiple classification algorithms.
5. Model evaluation using Precision, Recall, F1-score, and Confusion Matrix.
6. Decision threshold analysis.
7. Final model selection and serialization using Joblib.

### Models Explored

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

### Final Selected Model

**Decision Tree Classifier**

The final model uses:

- `max_depth=5`
- `min_samples_leaf=10`
- `class_weight='balanced'`
- `random_state=42`

The preprocessing pipeline includes median imputation for missing values.

The final model artifact stores:
- Trained preprocessing and classification pipeline.
- Prediction threshold: `0.50`.
- Required input feature names.

## ⚙️ Application Architecture

The project follows a frontend-backend architecture:

**1. Streamlit Frontend**
- Collects sensor readings from users.
- Sends prediction requests to FastAPI.
- Displays prediction results and failure probability.

**2. FastAPI Backend**
- Receives incoming HTTP requests.
- Validates sensor data using Pydantic.
- Loads the trained machine learning model.
- Generates predictions.
- Returns results in JSON format.

**3. Machine Learning Model**
- Processes validated sensor readings.
- Calculates failure probability.
- Applies the selected classification threshold.
- Returns the predicted machine status.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks API health |
| POST | `/predict` | Predicts machine failure |

### Example Prediction Request

```json
{
  "Temperature": 0.8,
  "Pressure": 0.88,
  "Vibration_Level": 0.77,
  "Humidity": 0.7,
  "Power_Consumption": 0.125
}
```

### Example API Response

```json
{
  "failure_probability": 0.74,
  "prediction": 1,
  "label": "Failure"
}
```

*Example response format; actual probability depends on the input readings.*

## 📁 Project Structure

```text
Machine_defect_m/
│
├── Machine defect_model/
│   ├── app.py
│   ├── main.py
│   ├── Machine_defect_model.joblib
│   ├── model.ipynb
│   ├── notebook.ipynb
│   └── requirements.txt
│
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone  https://github.com/vaishaliv7031-netizen/Machine_defect_prediction_mode
```

### 2. Navigate to the project directory

```bash
cd Machine_defect_m
```

### 3. Install dependencies

```bash
pip install -r "Machine defect_model/requirements.txt"
```

### 4. Start the FastAPI backend

Navigate to the application folder:

```bash
cd "Machine defect_model"
```

Run:

```bash
uvicorn main:app --reload
```

### 5. Access API documentation

Open:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit frontend

Open another terminal in the application folder and run:

```bash
streamlit run app.py
```

The Streamlit frontend should communicate with the running FastAPI backend.

## 📈 Prediction Output

The application provides:

- Machine failure probability.
- Predicted class (0 or 1).
- Machine status (Failure / No Failure).

## 🔮 Future Improvements

- Deploy the frontend and backend to cloud platforms.
- Integrate real-time IoT sensor data.
- Improve model recall and reduce false predictions.
- Add machine health monitoring dashboards.
- Implement model monitoring and performance tracking.

## 👩‍💻 Author

**Vaishali**

AI/ML | Python | Machine Learning | FastAPI | Streamlit
