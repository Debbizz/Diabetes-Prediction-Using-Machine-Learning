# Diabetes Prediction Model

This project trains a machine learning model to predict diabetes based on various health indicators.

## Features
- Preprocessing of numeric and categorical data
- Feature scaling using MinMaxScaler
- One-hot encoding for categorical variables
- Model training and evaluation
- Function to predict diabetes probability from user input

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
Usage
python
Copy
Edit
new_input = {
    'gender': 'Male',
    'age': 45.0,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'current',
    'bmi': 28.5,
    'HbA1c_level': 5.8,
    'blood_glucose_level': 180
}
predict_input(new_input)
Model Output
The model returns a tuple:

0 or 1 (Indicating No Diabetes or Diabetes)
Probability Score
If 0, return False; otherwise, return True.
```
