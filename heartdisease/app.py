from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ['age', 'sex', 'chestpain', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            print("Post cicked")
            # Extract features from form inputs
            input_data = []
            for feature in feature_names:
                val = float(request.form.get(feature))
                input_data.append(val)
                
            # Convert to 2D array, scale, predict
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            print(f"Input data:{input_data}")
            print("Predction:",prediction)
            result = "You may have heart disease." if prediction == 1 else "You are unlikely to have heart disease."
            print(result)
            return render_template('index.html', result=result, inputs=request.form)
        except Exception as e:
            return render_template('index.html', result=f"Error: {e}", inputs=request.form)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# 1 - disease  0 - No Disease