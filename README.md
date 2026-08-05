# Diabetes Prediction System

## Project Description

This project predicts whether a person is diabetic using Machine Learning.

## Features

- Data Preprocessing
- Feature Scaling
- Multiple ML Models
- Best Model Selection
- Streamlit Web Application
- Confidence Score

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Dataset

Diabetes Prediction Dataset

## How to Run

### Step 1

Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2

Train the model

```bash
python src/train_model.py
```

This command generates:

- `models/diabetes_model.pkl`

**Note:** The `diabetes_model.pkl` file is not included in this repository because it exceeds GitHub's web upload size limit. Running the above command will automatically generate it.

### Step 3

Run Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

## Author

**Yashaswini Balam**