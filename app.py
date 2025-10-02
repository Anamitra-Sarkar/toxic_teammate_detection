# Needed libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # For handling requests from the browser
import pandas as pd
import joblib
import os

# --- Configuration ---
# Create a 'model' directory in the same folder as app.py
# and place your .joblib file there.
MODEL_DIR = 'model'
MODEL_FILENAME = 'toxic_teammate_model.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load the Model ---
print(f"Loading model from: {MODEL_PATH}")
try:
    loaded_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Make sure the 'model' directory exists and contains the .joblib file.")
    loaded_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# --- Define Feature Columns (CRITICAL!) ---
# These are the columns the model was TRAINED on (AFTER one-hot encoding)
# !! PASTE THE LIST YOU COPIED FROM THE TRAINING SCRIPT OUTPUT HERE !!
# Example - Replace this with your actual list:
TRAINED_COLUMNS = [
    'Missed Meetings (Frequency)_1', 'Missed Meetings (Frequency)_2',
    'Missed Meetings (Frequency)_3', 'Missed Meetings (Frequency)_4',
    'Missed Meetings (Frequency)_5',
    'Deadline Adherence_Always on time', # <<< Add this line
    'Deadline Adherence_Frequently late', 'Deadline Adherence_Sometimes late',
    'Deadline Adherence_Usually on time',
    'Contribution Quality_2', 'Contribution Quality_3',
    'Contribution Quality_4', 'Contribution Quality_5',
    'Responsiveness_2', 'Responsiveness_3', 'Responsiveness_4', 'Responsiveness_5',
    'Communication Respect_2', 'Communication Respect_3',
    'Communication Respect_4', 'Communication Respect_5',
    'Workload Fairness (Perception)_2', 'Workload Fairness (Perception)_3',
    'Workload Fairness (Perception)_4', 'Workload Fairness (Perception)_5',
    'Discussion Participation_2', 'Discussion Participation_3',
    'Discussion Participation_4', 'Discussion Participation_5',
    'Credit Taking_Yes',
    'Conflict/Negativity_2', 'Conflict/Negativity_3',
    'Conflict/Negativity_4', 'Conflict/Negativity_5',
    'Harsh Criticism_2', 'Harsh Criticism_3', 'Harsh Criticism_4', 'Harsh Criticism_5',
    'Rework Required_Yes'
    # IMPORTANT: Double-check if columns like 'Credit Taking_No',
    # 'Rework Required_No', and the '_1' versions for numeric ratings
    # should also be here based on your training script's output columns!
    # If you used drop_first=False during training (implied by your current
    # code not using it), then ALL categories should be present.
]

# These are the ORIGINAL column names expected from the form/JSON
# Must match the keys in the JSON sent by the JavaScript
ORIGINAL_FEATURE_COLUMNS = [
    'Missed Meetings (Frequency)', 'Deadline Adherence',
    'Contribution Quality', 'Responsiveness', 'Communication Respect',
    'Workload Fairness (Perception)', 'Discussion Participation',
    'Credit Taking', 'Conflict/Negativity', 'Harsh Criticism',
    'Rework Required'
]


# --- Define Routes ---

# Route for the main page (serving the HTML)
@app.route('/')
def home():
    # Assumes your HTML file is named 'index.html' and is in a 'templates' folder
    # Create a folder named 'templates' in the same directory as app.py
    # and put your HTML file inside it.
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 1. Get data from the POST request
        data = request.get_json()
        print(f"Received data: {data}") # Log received data

        if not data:
            return jsonify({'error': 'No input data received'}), 400

        # 2. Convert data into a Pandas DataFrame (single row)
        # The keys in 'data' MUST match ORIGINAL_FEATURE_COLUMNS
        # Need to wrap the dict in a list to create a single-row DataFrame
        input_df = pd.DataFrame([data])
        print(f"Input DataFrame:\n{input_df}")

        # 3. Ensure correct data types (COMMENTED OUT / REMOVED)
        # for col in ['Missed Meetings (Frequency)', 'Contribution Quality', 'Responsiveness',
        #             'Communication Respect', 'Workload Fairness (Perception)',
        #             'Discussion Participation', 'Conflict/Negativity', 'Harsh Criticism']:
        #      if col in input_df.columns:
        #         # Convert to object (string) type BEFORE one-hot encoding
        #         input_df[col] = input_df[col].astype(str)


        # 4. Apply One-Hot Encoding (matching the training process)
        # Use the ORIGINAL feature column names here
        # Let get_dummies handle the mixed types in input_df directly
        input_encoded = pd.get_dummies(input_df, columns=ORIGINAL_FEATURE_COLUMNS) # Removed drop_first=True
        print(f"Encoded Input (after get_dummies):\n{input_encoded.columns.tolist()}") # Changed log message slightly


        # 5. Reindex to match the columns the model was trained on (CRITICAL STEP!)
        # This adds any missing columns (with value 0) and ensures the correct order.
        input_reindexed = input_encoded.reindex(columns=TRAINED_COLUMNS, fill_value=0)
        print(f"Reindexed Input Columns:\n{input_reindexed.columns.tolist()}")
        print(f"Reindexed Input Data:\n{input_reindexed}")


        # 6. Make Prediction
        prediction = loaded_model.predict(input_reindexed)
        prediction_proba = loaded_model.predict_proba(input_reindexed) # Get probabilities too

        # Extract the prediction result (it's usually an array)
        result = prediction[0]
        probability = prediction_proba[0] # Probabilities for [class_0, class_1]

        print(f"Raw Prediction: {prediction}")
        print(f"Predicted Class: {result}")
        print(f"Prediction Probabilities: {probability}") # Helps debugging

        # 7. Return the result as JSON
        return jsonify({
            'prediction': result,
            'probability_no': probability[0], # Assuming 'No' is the first class
            'probability_yes': probability[1] # Assuming 'Yes' is the second class
            })

    except KeyError as e:
        print(f"KeyError: {e} - Check if form names match ORIGINAL_FEATURE_COLUMNS")
        return jsonify({'error': f'Missing expected feature in input data: {e}'}), 400
    except Exception as e:
        print(f"Prediction Error: {e}")
        # Log the full traceback for detailed debugging if needed
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on your network (use with caution)
    # Use host='127.0.0.1' (default) for local access only.
    # debug=True automatically restarts the server when code changes (good for development)
    app.run(debug=True, host='0.0.0.0', port=5000)
