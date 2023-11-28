from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load the trained model and vectorizer
model = load('trained_model.joblib')
vec = load('vectorizer.joblib')

# Load the datasets
df_precaution = pd.read_csv('disease_precaution.csv')
df_description = pd.read_csv('disease_description.csv')
df_specialist = pd.read_csv('Doctor_Versus_Disease.csv', encoding='latin1', names=['Disease','Specialist'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    transformed_symptoms = vec.transform([symptoms])
    predicted_disease = model.predict(transformed_symptoms)

    # Fetch doctor recommendation, description, and precautions based on the predicted disease
    result = pd.DataFrame({'Disease': predicted_disease})
    result = result.merge(df_specialist, on="Disease", how="left")
    result = result.merge(df_precaution, on="Disease", how="left")
    result = result.merge(df_description, on="Disease", how="left")

    disease = result['Disease'].iloc[0]
    specialist = result['Specialist'].iloc[0]
    precaution = result['Precaution'].iloc[0] if 'Precaution' in result.columns else "N/A"
    description = result.get('Symptom_Description', ['N/A'])[0]
    return render_template('result.html', disease=disease, specialist=specialist, precaution=precaution, description=description)

if __name__ == '__main__':
    app.run(debug=True)
