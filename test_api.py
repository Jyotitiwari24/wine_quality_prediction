import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "input": [7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5]
}

response = requests.post(url, json=data)
print(response.json())
