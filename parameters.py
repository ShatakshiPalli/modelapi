import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

# Load the dataset
with open('final_dataset.json', encoding='utf-8') as f:
    data = json.load(f)

# Extract relevant features and target
def extract_features(patient):
    # Extract medical history conditions
    history = ' '.join([condition['condition'] for condition in patient['medical_history']])
    
    # Extract allergies, ensuring all items are strings
    allergies = ' '.join([item if isinstance(item, str) else '' for item in patient['allergies']])
    
    # Extract immunizations
    immunizations = ' '.join([vaccine['vaccine'] for vaccine in patient['immunizations']])
    
    # Extract lab results
    lab_results = ' '.join([test['test'] for test in patient['lab_results']])
    
    # Extract appointments, including department and reason
    appointments = ' '.join([f"{appt['department']} {appt['reason']}" for appt in patient['appointments']])
    
    return f"{history} {allergies} {immunizations} {lab_results} {appointments}"

X = [extract_features(patient) for patient in data]
y = [patient['specialty'] for patient in data]

# Text Vectorization
vectorizer = TfidfVectorizer()

# Label Encoding for specialties
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)

# Get the unique classes present in the test set
unique_classes = sorted(set(y_test))

# Print the classification report
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=label_encoder.inverse_transform(unique_classes)))

# Calculating F1 Score
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"F1 Score (Macro): {f1_macro}")

f1_weighted = f1_score(y_test, y_pred, average="weighted")
print(f"F1 Score (Weighted): {f1_weighted}")

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculating average error (misclassification rate)
average_error = 1 - accuracy
print(f"Average Error (Misclassification Rate): {average_error}")

# Baseline model: Predicting the majority class for all samples
majority_class = Counter(y_train).most_common(1)[0][0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
print(f"Baseline Accuracy: {baseline_accuracy}")

# Improvement over the baseline
improvement = accuracy - baseline_accuracy
print(f"Improvement over Baseline: {improvement}")

# Predict specialty for new data
def predict_specialty(patient_json):
    patient_features = extract_features(patient_json)
    specialty_encoded = pipeline.predict([patient_features])[0]
    return label_encoder.inverse_transform([specialty_encoded])[0]

# Example of prediction
new_patient = {
    "patient_id": "100023",
    "name": "Michael Lee",
    "age": 28,
    "gender": "Male",
    "address": "5050 Cedar St, Big City, USA",
    "phone": "555-0050",
    "email": "michael.lee@example.com",
    "insurance": {
        "provider": "WellCare",
        "policy_number": "WC789012"
    },
    "medical_history": [
        {
            "condition": "Major Depressive Disorder",
            "diagnosed_date": "2021-11-01",
            "medications": [
                {
                    "name": "Sertraline",
                    "dosage": "50mg",
                    "frequency": "Daily"
                }
            ]
        },
        {
            "condition": "Generalized Anxiety Disorder",
            "diagnosed_date": "2022-03-15",
            "medications": [
                {
                    "name": "Buspirone",
                    "dosage": "10mg",
                    "frequency": "Twice a day"
                }
            ]
        }
    ],
    "allergies": [
        "None"
    ],
    "immunizations": [
        {
            "vaccine": "COVID-19",
            "date": "2023-10-05"
        }
    ],
    "appointments": [
        {
            "date": "2024-08-15",
            "time": "09:00 AM",
            "doctor": "Dr. Clark",
            "department": "Psychiatry",
            "reason": "Medication management and therapy"
        }
    ],
    "lab_results": [
        {
            "test": "Mental Health Assessment",
            "date": "2024-01-10",
            "result": "Moderate symptoms of depression and anxiety",
            "normal_range": "Normal"
        }
    ]
}


predicted_specialty = predict_specialty(new_patient)
print(f"Predicted Specialty: {predicted_specialty}")
