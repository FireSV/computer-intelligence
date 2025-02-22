from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
# Load trained models
models = {
    "SVM": joblib.load("models/SVM.pkl"),
    "Neural Network": joblib.load("models/Neural Network.pkl"),
    "Decision Tree": joblib.load("models/Decision Tree.pkl"),
    "Random Forest": joblib.load("models/Random Forest.pkl"),
    "Gradient Boosting": joblib.load("models/Gradient Boosting.pkl"),
}

# Load preprocessing tools
label_encoders = joblib.load("models/label_encoders.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']  # Adjust based on actual dataset

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Remove 'id' if present
        data.pop('id', None)
        
        # Convert JSON to DataFrame
        if 'Compartments' in data:
            data['Compartments'] = int(data['Compartments'])  # Ensure it's an integer
        test_df = pd.DataFrame([data])

        # Ensure all required columns exist
        required_cols = categorical_cols + numerical_cols
        missing_cols = [col for col in required_cols if col not in test_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Encode categorical data
        for col in categorical_cols:
            if col in label_encoders:
                test_df[col] = test_df[col].astype(str)
                test_df[col] = test_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0)

        # Ensure numerical columns are numeric
        test_df[numerical_cols] = test_df[numerical_cols].apply(pd.to_numeric, errors='coerce')
        
        # Handle missing numerical values
        test_df[numerical_cols] = imputer.transform(test_df[numerical_cols])

        # Scale numerical features
        test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

        # Get predictions
        predictions = {name: model.predict(test_df)[0] for name, model in models.items()}
        ensemble_pred = np.mean(list(predictions.values()))

        # Return results as JSON
        response = {
            "predictions": predictions,
            "ensemble_prediction": ensemble_pred
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
