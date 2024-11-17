from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Dummy data
data = [
    {
        "capabilities": [1, 0, 1],
        "aspirations": [1, 0],
        "job": "Data Analyst"
    },
    {
        "capabilities": [0, 1, 1],
        "aspirations": [0, 1],
        "job": "Software Developer"
    },
    {
        "capabilities": [1, 1, 0],
        "aspirations": [1, 1],
        "job": "Project Manager"
    },
]

# Prepare the dataset
X = [d["capabilities"] + d["aspirations"]
     for d in data]  # Combine capabilities and aspirations
y = [d["job"] for d in data]

# Train the model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input JSON
    input_data = request.json
    capabilities = input_data.get("capabilities")
    aspirations = input_data.get("aspirations")

    # Validate inputs
    if not capabilities or not aspirations:
        return jsonify({"error":
                        "Capabilities and aspirations are required"}), 400

    # Predict the job
    input_features = np.array([capabilities + aspirations])
    predicted_job = model.predict(input_features)[0]

    return jsonify({"recommended_job": predicted_job})


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
