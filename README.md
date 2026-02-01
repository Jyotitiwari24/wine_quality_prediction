# Wine Quality Prediction

This project predicts the quality of red wine using a Random Forest Classifier.

## Dataset
The dataset used is `winequality-red.csv` (included in `data/` folder).

## Features
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

## Target
- Quality (Binary: Good >=7, Bad <7)

## Folder Structure
wine-quality-prediction/
│
├── data/ # Dataset
├── src/ # Scripts
├── app.py # Flask API
├── requirements.txt # Python dependencies
├── Dockerfile
├── .gitignore
└── README.md


## How to Run

### Locally
```bash
pip install -r requirements.txt
python src/train_model.py
python app.py
Using Docker
docker build -t wine-quality-app .
docker run -p 5000:5000 wine-quality-app
API Usage
POST request to /predict with JSON:

{
    "input": [7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5]
}
Response:

{
    "prediction": "Good Quality Wine"
}