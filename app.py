from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning

# Filter warnings to ignore DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the diabetes dataset from CSV
data = pd.read_csv("C:/Users/LENOVO/Desktop/AIOT/diabetes.csv")

# Separate features (X) and target variable (y)
X = data.drop('Outcome', axis=1)  # Assuming 'Outcome' is the column containing the target variable
y = data['Outcome']

# Train your machine learning model
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def login():
    return render_template('loginpage.html')

@app.route('/login_submit', methods=['POST'])
def login_submit():
    # Perform login authentication here (check credentials, etc.)
    # For now, assuming login is successful
    return render_template('index.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            # Extract JSON data from request body
            data = request.json

            # Extract form data from JSON
            pregnancies = data.get('pregnancies')
            glucose = data.get('glucose')
            blood_pressure = data.get('blood_pressure')
            skin_thickness = data.get('skin_thickness')
            insulin = data.get('insulin')
            bmi = data.get('bmi')
            diabetes_pedigree = data.get('diabetesPedigree')
            age = data.get('age')

            # Check if any form field is missing
            if None in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]:
                raise ValueError("Missing form field(s)")

            # Convert form data to floats
            pregnancies = float(pregnancies)
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            diabetes_pedigree = float(diabetes_pedigree)
            age = float(age)

            # Perform prediction
            prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])[0]

            # Convert prediction to Python int
            prediction = int(prediction)

            # Return prediction result as JSON
            return jsonify({'prediction': prediction})
    
        except (KeyError, ValueError) as e:
        # Handle missing or invalid form data
            return jsonify({'error': 'Invalid or missing form data', 'details': str(e)}), 400
        except Exception as e:
        # Print the exact error message to the console for debugging
            print("Prediction error:", e)
        # Return a generic error message with details
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500
    elif request.method == 'GET':
        # Render result.html template with prediction value
        prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)