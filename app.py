import joblib
import json
from flask import Flask, request, jsonify


app = Flask(__name__)

# Load the trained model pipeline and label encoder
model = joblib.load('model_pipeline.pkl')

with open('label_encoder.json') as f:
    label_encoder_data = json.load(f)
label_encoder_classes = label_encoder_data['classes_']

def extract_features(patient):
    history = ' '.join([condition['condition'] for condition in patient['medical_history']])
    allergies = ' '.join([item if isinstance(item, str) else '' for item in patient['allergies']])
    immunizations = ' '.join([vaccine['vaccine'] for vaccine in patient['immunizations']])
    lab_results = ' '.join([test['test'] for test in patient['lab_results']])
    appointments = ' '.join([f"{appt['department']} {appt['reason']}" for appt in patient['appointments']])
    return f"{history} {allergies} {immunizations} {lab_results} {appointments}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient_data = request.get_json()
        patient_features = extract_features(patient_data)
        specialty_encoded = model.predict([patient_features])[0]
        specialty = label_encoder_classes[specialty_encoded]
        return jsonify({'predicted_specialty': specialty})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "Welcome to the Medical Specialty Prediction API!"

if __name__ == "__main__":
    app.run(debug=True)
