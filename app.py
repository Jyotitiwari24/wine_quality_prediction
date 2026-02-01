from flask import Flask, request, jsonify, render_template
from src.predict import load_model, predict_quality

app = Flask(__name__)
model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = data['input']
    result = predict_quality(model, input_data)
    return jsonify({'prediction': result})


if __name__ == "__main__":
    app.run(debug=True)
