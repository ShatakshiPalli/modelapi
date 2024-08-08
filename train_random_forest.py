# import json
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# # from sklearn.metrics import classification_report, f1_score

# # Load the dataset
# # with open('dataset.json') as f:
# #     data = json.load(f)

# with open('final_dataset.json', encoding='utf-8') as f:
#     data = json.load(f)


# # Extract relevant features and target
# def extract_features(patient):
#     # Extract medical history conditions
#     history = ' '.join([condition['condition'] for condition in patient['medical_history']])
    
#     # Extract allergies, ensuring all items are strings
#     allergies = ' '.join([item if isinstance(item, str) else '' for item in patient['allergies']])
    
#     # Extract immunizations
#     immunizations = ' '.join([vaccine['vaccine'] for vaccine in patient['immunizations']])
    
#     # Extract lab results
#     lab_results = ' '.join([test['test'] for test in patient['lab_results']])
    
#     # Extract appointments, including department and reason
#     appointments = ' '.join([f"{appt['department']} {appt['reason']}" for appt in patient['appointments']])
    
#     return f"{history} {allergies} {immunizations} {lab_results} {appointments}"

# X = [extract_features(patient) for patient in data]
# y = [patient['specialty'] for patient in data]

# # Text Vectorization
# vectorizer = TfidfVectorizer()

# # Label Encoding for specialties
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Build the model pipeline
# pipeline = Pipeline([
#     ('vectorizer', vectorizer),
#     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
# ])

# # Train the model
# pipeline.fit(X_train, y_train)

# # Model Evaluation with classification report
# y_pred = pipeline.predict(X_test)

# # Get the unique classes present in the test set
# unique_classes = sorted(set(y_test))

# # Print the classification report
# # print(classification_report(y_test, y_pred, labels=unique_classes, target_names=label_encoder.inverse_transform(unique_classes)))

# #calculating f1 score
# # f1_macro = f1_score(y_test, y_pred, average="macro")
# # print(f"F1 Score (Macro) : {f1_macro}")

# # f1_weighted = f1_score(y_test, y_pred, average="weighted")
# # print(f"F1 Score (Weighted) : {f1_weighted}")


# # Predict specialty for new data
# def predict_specialty(patient_json):
#     patient_features = extract_features(patient_json)
#     specialty_encoded = pipeline.predict([patient_features])[0]
#     return label_encoder.inverse_transform([specialty_encoded])[0]

# # Example of prediction
# new_patient = {
#     "patient_id": "100026",
#     "name": "Seojin Park",
#     "age": 29,
#     "gender": "Female",
#     "address": "6060 Pine St, Southside, USA",
#     "phone": "555-0100",
#     "email": "seojin.park@example.com",
#     "insurance": {
#         "provider": "HealthCare",
#         "policy_number": "HC345678"
#     },
#     "medical_history": [
#         {
#             "condition": "Anxiety Disorder",
#             "diagnosed_date": "2020-11-20",
#             "medications": [
#                 {
#                     "name": "Sertraline",
#                     "dosage": "50mg",
#                     "frequency": "Once daily"
#                 }
#             ]
#         }
#     ],
#     "allergies": [
#         "None"
#     ],
#     "immunizations": [
#         {
#             "vaccine": "COVID-19",
#             "date": "2023-09-10"
#         }
#     ],
#     "appointments": [
#         {
#             "date": "2024-03-10",
#             "time": "03:00 PM",
#             "doctor": "Dr. Robinson",
#             "department": "Psychiatry",
#             "reason": "Mental health check-up"
#         }
#     ],
#     "lab_results": [
#         {
#             "test": "Blood Test",
#             "date": "2024-02-20",
#             "result": "Normal",
#             "normal_range": "Normal"
#         }
#     ]
# }






# predicted_specialty = predict_specialty(new_patient)
# print(f"Predicted Specialty: {predicted_specialty}")


import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load the dataset
with open('final_dataset.json', encoding='utf-8') as f:
    data = json.load(f)

# Extract relevant features and target
def extract_features(patient):
    history = ' '.join([condition['condition'] for condition in patient['medical_history']])
    allergies = ' '.join([item if isinstance(item, str) else '' for item in patient['allergies']])
    immunizations = ' '.join([vaccine['vaccine'] for vaccine in patient['immunizations']])
    lab_results = ' '.join([test['test'] for test in patient['lab_results']])
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

# Save the trained model pipeline and label encoder
joblib.dump(pipeline, 'model_pipeline.pkl')
with open('label_encoder.json', 'w') as f:
    json.dump({'classes_': label_encoder.classes_.tolist()}, f)
