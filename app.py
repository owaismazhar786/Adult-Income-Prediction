from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('adult_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form.get('age'))
    workclass = request.form.get('workclass')
    fnlwgt = int(request.form.get('fnlwgt'))
    education = request.form.get('education')
    educational_num = int(request.form.get('educational_num'))
    marital_status = request.form.get('marital_status')
    occupation = request.form.get('occupation')
    relationship = request.form.get('relationship')
    race = request.form.get('race')
    gender = request.form.get('gender')
    capital_gain = int(request.form.get('capital_gain'))
    capital_loss = int(request.form.get('capital_loss'))
    hours_per_week = int(request.form.get('hours_per_week'))
    native_country = request.form.get('native_country')

    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'educational-num': [educational_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Make predictions using the pre-trained model
    prediction = model.predict(input_data)

    # Display the predicted income category
    if prediction[0] == 0:
        income_category = '<=50K'
    else:
        income_category = '>50K'

    return render_template('result.html', prediction_text='Predicted income category: {}'.format(income_category))

if __name__ == '__main__':
    app.run(debug=True)
