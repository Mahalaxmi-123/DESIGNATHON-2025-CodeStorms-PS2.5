from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user inputs
            air_temp = float(request.form['air_temp'])
            process_temp = float(request.form['process_temp'])
            rotational_speed = float(request.form['rotational_speed'])
            torque = float(request.form['torque'])
            tool_wear = float(request.form['tool_wear'])
            machine_type = request.form['machine_type']

            # One-Hot Encode Machine Type (L, M, H)
            machine_type_encoded = [0, 0]  # Default [0, 0] for 'L'
            if machine_type == 'M':
                machine_type_encoded = [1, 0]  # 'M' => [1, 0]
            elif machine_type == 'H':
                machine_type_encoded = [0, 1]  # 'H' => [0, 1]

            # Prepare input (7 features total)
            input_data = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear] + machine_type_encoded])
            
            # Scale input
            input_scaled = scaler.transform(input_data)

            # 🔹 Fix input shape (Make it a sequence of 50)
            input_reshaped = np.tile(input_scaled, (50, 1))  # Repeat same row 50 times
            input_reshaped = input_reshaped.reshape(1, 50, 7)  # Shape (1, 50, 7)

            # Predict
            prediction = model.predict(input_reshaped)
            mse = np.mean(np.abs(input_reshaped - prediction))

            # Threshold (Use 95th percentile from training)
            threshold = 0.3  # Adjust based on your training analysis
            result = "Failure Expected" if mse > threshold else "No Failure Expected"
        
        except Exception as e:
            result = f"Error: {e}"
        
        return render_template('result.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
