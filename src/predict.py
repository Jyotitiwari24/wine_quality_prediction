import numpy as np
import pickle


def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_quality(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'


# Example usage
if __name__ == "__main__":
    model = load_model()
    input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5) 
    result = predict_quality(model, input_data)
    print(result)
